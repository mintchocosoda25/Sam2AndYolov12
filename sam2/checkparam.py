import torch
from sam2.build_sam import build_sam2

# 경로 지정
config_file = "configs/sam2.1/sam2.1_hiera_t.yaml"
ckpt_path = "./checkpoints/sam2.1_hiera_tiny.pt"

# 모델 로드 (teacher 모델로 사용 가능)
model = build_sam2(
    config_file=config_file,
    ckpt_path=ckpt_path,
    device="cpu",   # 또는 "cuda"
    mode="eval"
)


def count_parameters(model):
    total = 0
    sub_counts = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        n = param.numel()
        total += n
        top = name.split('.')[0]
        sub_counts[top] = sub_counts.get(top, 0) + n

    print("▶ 구조별 파라미터 수:")
    for k, v in sub_counts.items():
        print(f"{k:25s}: {v / 1e6:.2f}M")
    print(f"\n▶ 전체 파라미터 수: {total / 1e6:.2f}M")

# 실행
count_parameters(model)