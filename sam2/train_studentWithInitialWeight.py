import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

# Teacher
from sam2.build_sam import build_sam2

# Student
from build_student_sam2 import build_student_sam2  # 이미지 인코더만 축소한 student

##########################
# Dataset
##########################
class DistillImagesDataset(Dataset):
    def __init__(self, root_dir="distillData", transform=None):
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

##########################
# Count parameters
##########################
def count_parameters(model):
    """
    모델 파라미터 수(모듈별)와 총합을 출력
    """
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


##########################
# partial_load_trunk
##########################
def partial_load_trunk(teacher_trunk, student_trunk):
    """
    Teacher trunk -> Student trunk 부분 초기화 함수.
    shape가 같은 weight만 복사, 나머지는 그대로 유지.
    """

    teacher_sd = teacher_trunk.state_dict()
    student_sd = student_trunk.state_dict()
    
    loaded_keys = []
    skipped_keys = []
    
    for k, v in teacher_sd.items():
        if k in student_sd:
            if student_sd[k].shape == v.shape:
                # shape match -> copy
                student_sd[k] = v
                loaded_keys.append(k)
            else:
                skipped_keys.append((k, f"Shape mismatch: teacher={v.shape}, student={student_sd[k].shape}"))
        else:
            skipped_keys.append((k, "Key not found in student."))

    student_trunk.load_state_dict(student_sd)

    print("\n=== Partial Load Summary ===")
    print(">>> Loaded keys:")
    for k in loaded_keys:
        print("   ", k)
    print(f"Total loaded = {len(loaded_keys)}")

    print(">>> Skipped keys:")
    for k, reason in skipped_keys:
        print(f"   {k} -> {reason}")
    print(f"Total skipped = {len(skipped_keys)}\n")

    # # copy layer by layer
    # for k, v in teacher_sd.items():
    #     # k 예: "blocks.0.attn.qkv.weight"
    #     if k in student_sd and student_sd[k].shape == v.shape:
    #         print(f"Copying {k} from teacher trunk to student trunk")
    #         student_sd[k] = v
    #     else:
    #         pass  # skip mismatch or missing keys

    # student_trunk.load_state_dict(student_sd)


##########################
# Main
##########################
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Teacher 모델 (full) 로드
    teacher_model = build_sam2(
        config_file="configs/sam2.1/sam2.1_hiera_b+.yaml",
        ckpt_path="./checkpoints/sam2.1_hiera_base_plus.pt",
        device=device,
        mode="eval"
    )

    # 2) Student 모델 (이미지 인코더만 축소) 로드
    #    이 시점에서 student는 random init trunk
    student_model = build_student_sam2(
        teacher_config="configs/sam2.1/sam2.1_hiera_b+.yaml",
        teacher_ckpt="./checkpoints/sam2.1_hiera_base_plus.pt",
        device=device,
        mode="eval"
    ).to(device)

    # ▷ 부분 초기화: teacher trunk -> student trunk
    partial_load_trunk(teacher_model.image_encoder.trunk,
                       student_model.image_encoder.trunk)

    # 나머지 세팅
    student_model.train()

    print(">>> Student model parameter info BEFORE freezing trunk <<<")
    count_parameters(student_model)

    # 3) Dataset & Loader
    transform = T.Compose([
        T.Resize((512,512)),
        T.ToTensor(),
    ])
    dataset = DistillImagesDataset("distillData", transform=transform)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)

    # 4) Freeze everything except trunk
    for name, param in student_model.named_parameters():
        if "image_encoder.trunk" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    trunk_params = [p for p in student_model.parameters() if p.requires_grad]
    print(f"Trainable trunk params: {sum(p.numel() for p in trunk_params)/1e6:.3f} M")

    optimizer = torch.optim.Adam(trunk_params, lr=1e-4)

    total_steps = 0
    print("Start Distillation Training...\n")

    # 학습 epoch
    EPOCHS = 50
    for epoch in range(EPOCHS):
        for step, imgs in enumerate(loader):
            imgs = imgs.to(device)
            total_steps += 1

            # teacher forward
            with torch.no_grad():
                t_out = teacher_model.image_encoder(imgs)
                t_feats = t_out["vision_features"].float()  # [B,cT,H',W']

            # student forward
            s_out = student_model.image_encoder(imgs)
            s_feats = s_out["vision_features"].float()    # [B,cS,H',W']

            # 채널 mismatch -> 1x1 conv
            if s_feats.shape[1] != t_feats.shape[1]:
                if not hasattr(main, 'channel_proj'):
                    import torch.nn as nn
                    main.channel_proj = nn.Conv2d(s_feats.shape[1], t_feats.shape[1], 1).to(device)
                s_feats = main.channel_proj(s_feats)

            # 해상도 mismatch -> interpolate
            if s_feats.shape[-2:] != t_feats.shape[-2:]:
                s_feats = torch.nn.functional.interpolate(
                    s_feats, size=t_feats.shape[-2:], mode="bilinear"
                )

            # Loss = MSE
            loss = torch.nn.functional.mse_loss(s_feats, t_feats)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                print(f"Epoch [{epoch}/{EPOCHS}], Step [{step}/{len(loader)}], "
                      f"Global Step={total_steps}, Loss={loss.item():.4f}")

    print("\nDistillation done!")

    # 5) 모델 저장
    save_path = "student_model_distilled.pt"
    torch.save(student_model.state_dict(), save_path)
    print(f"Student model saved to: {save_path}")

if __name__=="__main__":
    main()
