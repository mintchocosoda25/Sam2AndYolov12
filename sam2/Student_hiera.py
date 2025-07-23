# file: my_student_trunk.py
import torch
from torch import nn
from sam2.modeling.backbones.hieradet import Hiera
class HieraStudentMinimal(Hiera):
    """
    Hiera 기반으로 stage/block 수, head_mul 등을 줄여 파라미터를 낮춘 예시.
    Teacher와 neck 호환 위해, 최종 embed_dim이 teacher와 비슷해지도록
    dim_mul / stages 조절 가능.
    """
    
    def __init__(
        self,
        embed_dim=96,             # 초기 채널 (teacher와 동일하면 neck 호환 쉬움)
        num_heads=1,             # 초기 head 수
        drop_path_rate=0.1,
        q_pool=3,
        q_stride=(2,2),
        stages=(2,2,4,1),        # block 개수 축소
        dim_mul=2.0,             # stage 넘어갈 때 채널 배율
        head_mul=1.0,            
        window_spec=(8,4,14,7),
        global_att_blocks=(3, 5, 7),  
        window_pos_embed_bkg_spatial_size=(7, 7),
        weights_path=None,
        return_interm_layers=True,
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            drop_path_rate=drop_path_rate,
            q_pool=q_pool,
            q_stride=q_stride,
            stages=stages,
            dim_mul=dim_mul,
            head_mul=head_mul,
            window_spec=window_spec,
            global_att_blocks=global_att_blocks,
            weights_path=weights_path,
            return_interm_layers=return_interm_layers,
        )


