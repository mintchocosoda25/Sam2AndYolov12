import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

os.environ["CUDA_VISIBLE_DEVICES"]="2"

# Teacher
from sam2.build_sam import build_sam2

# Student
from build_student_sam2 import build_student_sam2_Tiny  # 이미지 인코더만 축소한 student

##########################
# Dataset
##########################
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

##########################
# Count parameters
##########################
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

##########################
# Partial load for trunk weights
##########################
def partial_load_trunk(teacher_trunk, student_trunk):
    teacher_sd = teacher_trunk.state_dict()
    student_sd = student_trunk.state_dict()
    
    loaded_keys = []
    skipped_keys = []
    
    for k, v in teacher_sd.items():
        if k in student_sd:
            if student_sd[k].shape == v.shape:
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

##########################
# Main Training & Validation
##########################
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Teacher 모델 로드 (전체 SAM2 모델)
    teacher_model = build_sam2(
        config_file="configs/sam2.1/sam2.1_hiera_t.yaml",
        ckpt_path="./checkpoints/sam2.1_hiera_tiny.pt",
        device=device,
        mode="eval"
    )
    
    # 2) Student 모델 로드 (이미지 인코더만 축소한 student)
    student_model = build_student_sam2_Tiny(
        teacher_config="configs/sam2.1/sam2.1_hiera_t.yaml",
        teacher_ckpt="./checkpoints/sam2.1_hiera_tiny.pt",
        device=device,
        mode="eval"
    ).to(device)
    
    # 부분 초기화: teacher trunk -> student trunk
    partial_load_trunk(teacher_model.image_encoder.trunk,
                       student_model.image_encoder.trunk)

    
    # 미리 저장된 student checkpoint가 있다면 로드 (계속 학습 가능)
    checkpoint_path = "student_model_distilled.pt"
    if os.path.exists(checkpoint_path):
        print(f"Loading pre-saved student checkpoint from {checkpoint_path} ...")
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        student_model.load_state_dict(state_dict)

    
    # Freeze 모든 파라미터 중 trunk만 학습하도록 설정
    for name, param in student_model.named_parameters():
        if "image_encoder.trunk" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    trunk_params = [p for p in student_model.parameters() if p.requires_grad]
    print(">>> Trainable trunk params: {:.3f} M".format(sum(p.numel() for p in trunk_params)/1e6))
    
    # 학습 전 파라미터 수 확인
    print(">>> Student model parameter info BEFORE training <<<")
    count_parameters(student_model)
    
    # 3) Transform, Dataset, DataLoader for Training
    train_transform = T.Compose([
        T.Resize((512,512)),
        T.ToTensor(),
    ])
    train_dataset = DistillImagesDataset("distillTrain", transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True, num_workers=4)
    
    # Validation dataset: distillVal
    val_transform = T.Compose([
        T.Resize((512,512)),
        T.ToTensor(),
    ])
    val_dataset = DistillImagesDataset("distillVal", transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=24, shuffle=False, num_workers=4)
    
    # 4) Optimizer (trunk 파라미터만)
    optimizer = torch.optim.Adam(trunk_params, lr=1e-4)
    
    total_steps = 0
    EPOCHS = 200
    print("Start Distillation Training...\n")
    
    for epoch in range(EPOCHS):
        student_model.train()
        running_loss = 0.0
        for step, imgs in enumerate(train_loader):
            imgs = imgs.to(device)
            total_steps += 1

            # Teacher forward
            with torch.no_grad():
                t_out = teacher_model.image_encoder(imgs)
                t_feats = t_out["vision_features"].float()  # [B, cT, H, W]

            # Student forward
            s_out = student_model.image_encoder(imgs)
            s_feats = s_out["vision_features"].float()    # [B, cS, H, W]

            # 채널 불일치 -> 1x1 conv
            if s_feats.shape[1] != t_feats.shape[1]:
                if not hasattr(main, 'channel_proj'):
                    import torch.nn as nn
                    main.channel_proj = nn.Conv2d(s_feats.shape[1], t_feats.shape[1], 1).to(device)
                s_feats = main.channel_proj(s_feats)
            
            # 해상도 불일치 -> interpolate
            if s_feats.shape[-2:] != t_feats.shape[-2:]:
                s_feats = torch.nn.functional.interpolate(s_feats, size=t_feats.shape[-2:], mode="bilinear")
            
            loss = torch.nn.functional.mse_loss(s_feats, t_feats)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if step % 50 == 0:
                print(f"Train: Epoch [{epoch+1}/{EPOCHS}], Step [{step}/{len(train_loader)}], Global Step={total_steps}, Loss={loss.item():.4f}")
        
        # 학습 epoch 끝난 후 검증 진행
        student_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_imgs in val_loader:
                val_imgs = val_imgs.to(device)
                t_out = teacher_model.image_encoder(val_imgs)
                t_feats = t_out["vision_features"].float()
                
                s_out = student_model.image_encoder(val_imgs)
                s_feats = s_out["vision_features"].float()
                
                if s_feats.shape[1] != t_feats.shape[1]:
                    s_feats = main.channel_proj(s_feats)
                if s_feats.shape[-2:] != t_feats.shape[-2:]:
                    s_feats = torch.nn.functional.interpolate(s_feats, size=t_feats.shape[-2:], mode="bilinear")
                
                batch_loss = torch.nn.functional.mse_loss(s_feats, t_feats)
                val_loss += batch_loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] Summary: Train Loss={avg_train_loss:.4f}, Validation Loss={avg_val_loss:.4f}\n")

        if avg_val_loss < 0.002:
            save_path = "student_model_distill_real.pt"
            torch.save(student_model.state_dict(), save_path)
            print(f"Student model saved to: {save_path}")

        if epoch%20==0:
            save_path = f"student_model_distill_epoch{epoch}.pt"
            torch.save(student_model.state_dict(), save_path)
            print(f"(BackUp)Student model saved to: {save_path}, epoch:{epoch}")

    
    print("Distillation done!")
    
    # 5) 모델 저장
    #save_path = "student_model_distilled.pt"
    #torch.save(student_model.state_dict(), save_path)
    #print(f"Student model saved to: {save_path}")

if __name__=="__main__":
    main()
