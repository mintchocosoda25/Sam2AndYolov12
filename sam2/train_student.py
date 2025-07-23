import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

# Teacher
from sam2.build_sam import build_sam2

# Student
from build_student_sam2 import build_student_sam2

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

def count_parameters(model):
    """
    모델 파라미터 수(모듈별)와 총합을 출력
    """
    total = 0
    sub_counts = {}
    for name, param in model.named_parameters():
        # 굳이 requires_grad 검사를 안 해도 전체 파라미터가 확인됨.
        # if not param.requires_grad:
        #     continue
        nump = param.numel()
        total += nump
        topmod = name.split(".")[0]
        sub_counts[topmod] = sub_counts.get(topmod, 0) + nump

    print("\n=== Parameter count by top-level module ===")
    for m, c in sub_counts.items():
        print(f"{m:25s} : {c/1e6:.3f} M")
    print(f"TOTAL PARAMS : {total/1e6:.3f} M\n")
    
    


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Student 모델 로드
    student_model = build_student_sam2(
        teacher_config="configs/sam2.1/sam2.1_hiera_b+.yaml",
        teacher_ckpt="./checkpoints/sam2.1_hiera_base_plus.pt",
        device=device,
        mode="eval"
    ).to(device)

    student_model.train()

    print(">>> Student model parameter info BEFORE freezing trunk <<<")
    count_parameters(student_model)

    # 2) Teacher 모델 로드
    teacher_model = build_sam2(
        config_file="configs/sam2.1/sam2.1_hiera_b+.yaml",
        ckpt_path="./checkpoints/sam2.1_hiera_base_plus.pt",
        device=device,
        mode="eval"
    )
    


    # 3) Transform
    transform = T.Compose([
        T.Resize((512,512)),
        T.ToTensor(),
    ])
    dataset = DistillImagesDataset("distillData", transform=transform)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)

    # Freeze everything except trunk
    for name, param in student_model.named_parameters():
        if "image_encoder.trunk" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # 다시 파라미터 확인 (학습 대상만 세고 싶다면 requires_grad=True만 센다)
    trunk_params = [p for p in student_model.parameters() if p.requires_grad]
    print(">>> Student trunk parameter info (requires_grad=True) <<<")
    total_trunk = sum(p.numel() for p in trunk_params)
    print(f"Trunk trainable params: {total_trunk/1e6:.3f} M")

    # Optimizer
    optimizer = torch.optim.Adam(trunk_params, lr=1e-4)
    
    total_steps = 0
    print("Start Distillation Training...\n")

    for epoch in range(50):
        for step, imgs in enumerate(loader):
            imgs = imgs.to(device)
            total_steps += 1

            # Teacher forward
            with torch.no_grad():
                t_out = teacher_model.image_encoder(imgs)
                t_feats = t_out["vision_features"].float()  # [B, cT, H', W']

            # Student forward
            s_out = student_model.image_encoder(imgs)
            s_feats = s_out["vision_features"].float()     # [B, cS, H', W']

            # 1) 채널 수 불일치 시 1x1 conv
            if s_feats.shape[1] != t_feats.shape[1]:
                if not hasattr(main, 'channel_proj'):
                    import torch.nn as nn
                    main.channel_proj = nn.Conv2d(s_feats.shape[1], t_feats.shape[1], 1).to(device)
                s_feats = main.channel_proj(s_feats)

            # 2) 해상도 불일치 시 interpolate
            if s_feats.shape[-2:] != t_feats.shape[-2:]:
                s_feats = torch.nn.functional.interpolate(s_feats, size=t_feats.shape[-2:], mode="bilinear")

            # 3) MSE
            loss = torch.nn.functional.mse_loss(s_feats, t_feats)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                print(f"Epoch [{epoch}/10], Step [{step}/{len(loader)}], "
                      f"Global Step={total_steps}, Loss={loss.item():.4f}")

    print("\nDistillation done!")

    # 4) 모델 저장
    save_path = "student_model_distilled.pt"
    torch.save(student_model.state_dict(), save_path)
    print(f"Student model saved to: {save_path}")

if __name__=="__main__":
    main()
