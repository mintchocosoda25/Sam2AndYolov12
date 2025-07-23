import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Teacher: 전체 SAM2 모델 생성 (이미지 인코더 trunk 포함)
from sam2.build_sam import build_sam2

# Student: trunk만 경량화한 학생 모델 (다중 스케일 출력 유지)
# 내부에서 trunk를 HieraStudentMultiScale로 교체했다고 가정
from build_student_sam2 import build_student_sam2_Tiny

##############################################
# Adaptation Layer (보정 레이어) 정의
##############################################
class ScaleAdaptation(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 단순 1x1 conv; 필요에 따라 더 복잡한 네트워크로 확장 가능
        self.adapt = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.adapt(x)

##############################################
# Dataset 정의: distillTrain 폴더의 이미지들로 학습, distillVal로 검증
##############################################
class DistillImagesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.files = sorted(glob.glob(os.path.join(root_dir, "*.jpg")))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

##############################################
# 파라미터 수 확인 함수 (선택사항)
##############################################
def count_parameters(model):
    total = 0
    sub_counts = {}
    for name, param in model.named_parameters():
        nump = param.numel()
        total += nump
        topmod = name.split(".")[0]
        sub_counts[topmod] = sub_counts.get(topmod, 0) + nump

    print("\n=== Parameter count by top-level module ===")
    for m, c in sub_counts.items():
        print(f"{m:25s} : {c/1e6:.3f} M")
    print(f"TOTAL PARAMS : {total/1e6:.3f} M\n")

##############################################
# Teacher->Student 부분 초기화 (선택사항)
##############################################
def partial_load_trunk(teacher_trunk, student_trunk):
    teacher_sd = teacher_trunk.state_dict()
    student_sd = student_trunk.state_dict()
    
    loaded_keys = []
    skipped_keys = []
    
    for k, v in teacher_sd.items():
        if k in student_sd and student_sd[k].shape == v.shape:
            student_sd[k] = v
            loaded_keys.append(k)
        else:
            skipped_keys.append((k, "Shape mismatch or key not found"))
    
    student_trunk.load_state_dict(student_sd)
    
    print("\n=== Partial Load Summary ===")
    print("Loaded keys:")
    for k in loaded_keys:
        print("   ", k)
    print(f"Total loaded = {len(loaded_keys)}")
    print("Skipped keys:")
    for k, reason in skipped_keys:
        print(f"   {k} -> {reason}")
    print(f"Total skipped = {len(skipped_keys)}\n")

##############################################
# Main Training & Validation 함수
##############################################
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 경로 설정 (원하는 config 및 checkpoint 경로로 수정)
    teacher_config = "configs/sam2.1/sam2.1_hiera_t.yaml"
    teacher_ckpt = "./checkpoints/sam2.1_hiera_tiny.pt"
    # student 모델은 teacher config를 기반으로 생성하되, 내부 trunk는 경량화되어 있음.
    student_config = teacher_config  # config는 teacher와 동일하게 사용
    # 만약 학습 전 checkpoint가 있다면 resume_ckpt에 그 파일 경로를 넣으세요.
    resume_ckpt = "./student_model_distilled_epoch20.pt"  # 예시; 없으면 None

    # 1) Teacher 모델 생성 (전체 SAM2, image_encoder 포함)
    teacher_model = build_sam2(
        config_file=teacher_config,
        ckpt_path=teacher_ckpt,
        device=device,
        mode="eval"
    )
    teacher_model.to(device)
    teacher_model.eval()

    # 2) Student 모델 생성 (trunk만 경량화된 student 모델)
    student_model = build_student_sam2_Tiny(
        teacher_config=student_config,
        teacher_ckpt=teacher_ckpt,
        device=device,
        mode="eval"
    ).to(device)
    # 미리 저장된 student checkpoint가 있으면 로드
    if resume_ckpt is not None and os.path.exists(resume_ckpt):
        print("Resuming student model training from checkpoint:", resume_ckpt)
        sd = torch.load(resume_ckpt, map_location=device, weights_only=True)
        if "model" in sd:
            sd = sd["model"]
        student_model.load_state_dict(sd)
        start_epoch = 0  # 필요시 checkpoint 에서 epoch 정보를 가져옵니다.
        print("Checkpoint loaded.")
    else:
        start_epoch = 0

    student_model.train()

    # (선택사항) 부분 초기화: teacher trunk -> student trunk
    # partial_load_trunk(teacher_model.image_encoder.trunk, student_model.image_encoder.trunk)

    # 3) Multi-scale distillation을 위해, teacher와 student의 trunk 출력은 리스트여야 함
    # (각 scale별로 feature를 반환해야 함)
    channel_projs = None  # Lazy initialization
    adaptation_layer = None  # scale 3 (인덱스 2)에 대해 보정 레이어, lazy 초기화

    # 4) 학생 모델 중 trunk만 학습하도록 freeze (나머지는 고정)
    for name, param in student_model.named_parameters():
        if "image_encoder.trunk" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    trunk_params = [p for p in student_model.parameters() if p.requires_grad]
    print("Trainable trunk params: {:.3f} M".format(sum(p.numel() for p in trunk_params)/1e6))
    count_parameters(student_model)

    # 5) Dataset 및 DataLoader (학습: distillTrain / 검증: distillVal)
    train_transform = T.Compose([
        T.Resize((512,512)),
        T.ToTensor(),
    ])
    train_dataset = DistillImagesDataset("distillTrain", transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

    val_transform = T.Compose([
        T.Resize((512,512)),
        T.ToTensor(),
    ])
    val_dataset = DistillImagesDataset("distillVal", transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    # 6) Optimizer (trunk 파라미터만)
    optimizer = torch.optim.Adam(trunk_params, lr=1e-4)

    total_steps = 0
    EPOCHS = 250
    print("Start Distillation Training...\n")
    for epoch in range(start_epoch, EPOCHS):
        student_model.train()
        running_loss = 0.0
        for step, imgs in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
            imgs = imgs.to(device)
            total_steps += 1

            # Teacher forward: trunk outputs (리스트: 각 스케일별 feature maps)
            with torch.no_grad():
                teacher_feats = teacher_model.image_encoder.trunk(imgs)
            # Student forward: trunk outputs (리스트)
            student_feats = student_model.image_encoder.trunk(imgs)

            # Lazy channel projection: 각 scale별로 teacher와 student의 채널 수가 다르다면 생성
            if channel_projs is None:
                channel_projs = []
                for t_feat, s_feat in zip(teacher_feats, student_feats):
                    in_channels = s_feat.shape[1]
                    out_channels = t_feat.shape[1]
                    if in_channels != out_channels:
                        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1).to(device)
                    else:
                        conv = nn.Identity().to(device)
                    channel_projs.append(conv)

            loss = 0.0
            # 각 스케일별 손실 계산
            for i, (t_feat, s_feat) in enumerate(zip(teacher_feats, student_feats)):
                # 기본적으로 channel_projs를 사용하여 채널 맞추기
                s_feat_proj = channel_projs[i](s_feat)
                if s_feat_proj.shape[-2:] != t_feat.shape[-2:]:
                    s_feat_proj = F.interpolate(s_feat_proj, size=t_feat.shape[-2:], mode="bilinear", align_corners=False)
                # 스케일 3 (인덱스 2)에 대해서는 추가 보정 레이어 적용
                if i == 2:
                    if adaptation_layer is None:
                        # 학생과 teacher의 스케일 3 채널 수 확인 후 adaptation layer 생성
                        student_scale3_channels = s_feat_proj.shape[1]
                        teacher_scale3_channels = t_feat.shape[1]
                        adaptation_layer = ScaleAdaptation(student_scale3_channels, teacher_scale3_channels).to(device)
                        print("Adaptation layer for scale 3 created: {} -> {}".format(student_scale3_channels, teacher_scale3_channels))
                    s_feat_proj = adaptation_layer(s_feat_proj)
                curr_loss = F.mse_loss(s_feat_proj, t_feat)
                loss += curr_loss
                # 개별 손실 출력 (옵션으로 각 스케일별 loss 확인)
                # print(f"Scale {i+1} Loss: {curr_loss.item():.4f}")
            # (옵션) 전체 스케일 loss 평균으로 만들 수도 있음.
            # loss /= len(teacher_feats)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # 검증 단계
        student_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs in val_loader:
                imgs = imgs.to(device)
                teacher_feats = teacher_model.image_encoder.trunk(imgs)
                student_feats = student_model.image_encoder.trunk(imgs)
                batch_loss = 0.0
                for i, (t_feat, s_feat) in enumerate(zip(teacher_feats, student_feats)):
                    s_feat_proj = channel_projs[i](s_feat)
                    if s_feat_proj.shape[-2:] != t_feat.shape[-2:]:
                        s_feat_proj = F.interpolate(s_feat_proj, size=t_feat.shape[-2:], mode="bilinear", align_corners=False)
                    if i == 2 and adaptation_layer is not None:
                        s_feat_proj = adaptation_layer(s_feat_proj)
                    batch_loss += F.mse_loss(s_feat_proj, t_feat)
                val_loss += batch_loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] Summary: Train Loss={avg_train_loss:.4f}, Validation Loss={avg_val_loss:.4f}")

        # 중간 체크포인트 저장 (10 에폭마다 또는 일정 val loss 미만이면 저장)
        if (epoch + 1) % 10 == 0 or avg_val_loss < 0.008:
            save_path = f"student_model_distilled_epoch{epoch+1}.pt"
            state_dict = student_model.state_dict()
            wrapped_ckpt = {"model": state_dict}
            torch.save(wrapped_ckpt, save_path)
            print(f"(Checkpoint) Student model saved to: {save_path}, epoch: {epoch+1}")

    print("\nDistillation Training Done!")
    final_save_path = "student_model_distilled_final2.pt"
    state_dict = student_model.state_dict()
    wrapped_ckpt = {"model": state_dict}
    torch.save(wrapped_ckpt, final_save_path)
    print(f"Final student model saved to: {final_save_path}")

if __name__ == "__main__":
    main()
