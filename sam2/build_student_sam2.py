# file: build_student_sam2.py

import torch
from sam2.build_sam import build_sam2
from Student_hiera import HieraStudentMinimal

def build_student_sam2(
    teacher_config="configs/sam2.1/sam2.1_hiera_b+.yaml",
    teacher_ckpt="./checkpoints/sam2.1_hiera_base_plus.pt",
    device="cuda",
    mode="eval",
):
    """
    1) Teacher config로 SAM2 모델 생성
    2) 이미지 인코더의 trunk만 HieraStudentMinimal로 교체
    3) MemoryAttention, MemoryEncoder, MaskDecoder 등은 그대로
    4) 반환된 모델 = 'Student', 이미지 인코더만 작아짐
    """
    # 1) Teacher config 기반 모델 생성
    student_model = build_sam2(
        config_file=teacher_config,
        ckpt_path=teacher_ckpt,
        device=device,
        mode=mode,
        apply_postprocessing=True,  # 원하는 대로
    )

    # 2) trunk 교체
    #   - student_model.image_encoder.trunk -> HieraStudentMinimal(...)
    new_trunk = HieraStudentMinimal( )
    student_model.image_encoder.trunk = new_trunk

    # 만약 neck의 channel_list가 trunk.channel_list와 호환되는지 점검 필요
    # teacher neck.d_model=256 등은 그대로
    # scalp=1인지 0인지도 teacher랑 동일한지 보고 조정 가능

    return student_model


def build_student_sam2_Tiny(
    teacher_config="configs/sam2.1/sam2.1_hiera_t.yaml",
    teacher_ckpt="./checkpoints/sam2.1_hiera_tiny.pt",
    device="cuda",
    mode="eval",
):
    """
    1) Teacher config로 SAM2 모델 생성
    2) 이미지 인코더의 trunk만 HieraStudentMinimal로 교체
    3) MemoryAttention, MemoryEncoder, MaskDecoder 등은 그대로
    4) 반환된 모델 = 'Student', 이미지 인코더만 작아짐
    """
    # 1) Teacher config 기반 모델 생성
    student_model = build_sam2(
        config_file=teacher_config,
        ckpt_path=teacher_ckpt,
        device=device,
        mode=mode,
        apply_postprocessing=True,  # 원하는 대로
    )

    # 2) trunk 교체
    #   - student_model.image_encoder.trunk -> HieraStudentMinimal(...)
    new_trunk = HieraStudentMinimal( 
        embed_dim=96,)
    student_model.image_encoder.trunk = new_trunk

    # 만약 neck의 channel_list가 trunk.channel_list와 호환되는지 점검 필요
    # teacher neck.d_model=256 등은 그대로
    # scalp=1인지 0인지도 teacher랑 동일한지 보고 조정 가능

    return student_model
