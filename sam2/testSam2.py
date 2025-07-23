# eval_student.py
import os, json, csv, time, cv2, numpy as np, torch
from tqdm import tqdm
from pycocotools import mask as mask_utils
from sam2.build_sam import build_sam2_video_predictor

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# ── 0. 경로 / 모델 설정 ─────────────────────────────────────────
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
VIDEO_ROOT    = "./videos"          # 50개 비디오가 있는 폴더
GT_ROOT       = "./soccer_gt"             # 앞서 만든 GT‑json 들이 있는 곳
STUDENT_CKPT  = "./checkpoints/sam2.1_hiera_tiny.pt"
STUDENT_CFG   = "./configs/sam2.1/sam2.1_hiera_t.yaml" 
OUT_CSV       = "eval_results.csv"

# model_cfg = "./configs/sam2.1/sam2_t_t.yaml"
# sam2_checkpoint = "./checkpoints/student_model_distilled_final.pt"

# "./configs/sam2.1/sam2.1_hiera_t.yaml" 
# "./checkpoints/sam2.1_hiera_tiny.pt"


# ── 1. student predictor (Triton/compile OFF) ─────────────────
predictor = build_sam2_video_predictor(
    STUDENT_CFG,
    STUDENT_CKPT,
    device=DEVICE,
    vos_optimized=False,
    hydra_overrides_extra=["++model.compile_image_encoder=false"],
).eval()

# ── 2. 유틸 함수 ───────────────────────────────────────────────
def rle_to_mask(seg, h, w):
    """
    seg : dict  {"counts": str | list, "size":[h,w]}
          str  "xxxx..."
    """
    if isinstance(seg, str):
        seg = {"counts": seg.encode("ascii"), "size": [h, w]}
    elif isinstance(seg["counts"], str):
        seg = {"counts": seg["counts"].encode("ascii"), "size": seg["size"]}
    return mask_utils.decode(seg).astype(np.uint8)

def mask_centroid(mask):
    ys, xs = np.nonzero(mask)
    return (float(xs.mean()), float(ys.mean())) if xs.size else None

def iou_score(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return 0.0 if union == 0 else inter / union

# ── 3. 평가 루프 ───────────────────────────────────────────────
rows = [("video", "mIoU", "presence_acc", "high_iou_ratio", "runtime_sec")]
VIDEO_LIST = sorted([d for d in os.listdir(VIDEO_ROOT)
                     if os.path.isdir(os.path.join(VIDEO_ROOT, d))])[:100]

for vid in tqdm(VIDEO_LIST, desc="Videos"):
    gt_path = os.path.join(GT_ROOT, f"{vid}.json")
    if not os.path.isfile(gt_path):
        print(f"[WARN] {vid}: GT json missing → skip")
        continue

    with open(gt_path) as f:
        gt = json.load(f)

    gt_imgs = sorted(gt["images"], key=lambda x: x["id"])
    gt_ann  = {a["image_id"]: a for a in gt["annotations"]}
    num_f   = len(gt_imgs)
    h, w    = gt_imgs[0]["height"], gt_imgs[0]["width"]

    # ── 3‑1. 첫 프레임 centroid 클릭 ───────────────────────────
    mask0_gt = rle_to_mask(gt_ann[0]["segmentation"], h, w)
    cent = mask_centroid(mask0_gt)
    if cent is None:                       # GT가 비어 있으면 skip
        print(f"[WARN] {vid}: frame‑0 GT empty → skip")
        continue
    cx, cy = cent

    # predictor 초기화
    vdir  = os.path.join(VIDEO_ROOT, vid)
    state = predictor.init_state(video_path=vdir)

    _, _, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=state,
        frame_idx=0,
        obj_id=1,                          # 단일 객체 id=1
        points=np.array([[cx, cy]], np.float32),
        labels=np.array([1], np.int32),
    )

    # ── 3‑2. 메트릭 초기화 ────────────────────────────────────
    iou_sum = pres_correct = high_iou_cnt = 0
    start_t = time.time()

    # 첫 프레임 평가
    pred0 = (out_mask_logits[0, 0] > 0).cpu().numpy()
    iou0  = iou_score(mask0_gt, pred0)
    iou_sum       += iou0
    high_iou_cnt  += iou0 >= 0.90
    pres_correct  += int((mask0_gt.sum() > 0) == (pred0.sum() > 0))

    # ── 3‑3. 나머지 프레임 propagate & 평가 ──────────────────
    for f_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
        obj_ids_np = np.atleast_1d(np.asarray(obj_ids).astype(int))
        hit = np.nonzero(obj_ids_np == 1)[0]
        if hit.size:
            pred_mask = (mask_logits[hit[0], 0] > 0).cpu().numpy()
        else:                               # 객체가 사라진 경우
            pred_mask = np.zeros((h, w), np.uint8)

        gt_mask = rle_to_mask(gt_ann[f_idx]["segmentation"], h, w)
        iou = iou_score(gt_mask, pred_mask)
        iou_sum      += iou
        high_iou_cnt += iou >= 0.90
        pres_correct += int((gt_mask.sum() > 0) == (pred_mask.sum() > 0))

    runtime = time.time() - start_t
    mIoU            = iou_sum / num_f
    presence_acc    = pres_correct / num_f
    high_iou_ratio  = high_iou_cnt / num_f

    rows.append((vid,
                 f"{mIoU:.4f}",
                 f"{presence_acc:.4f}",
                 f"{high_iou_ratio:.4f}",
                 f"{runtime:.2f}"))

    print(f"{vid:15s} | mIoU {mIoU:.4f} | Acc {presence_acc:.4f} | ≥0.9 {high_iou_ratio:.4f} | {runtime:.2f}s")

# ── 4. CSV 저장 ───────────────────────────────────────────────
with open(OUT_CSV, "w", newline="") as f:
    csv.writer(f).writerows(rows)
print(f"\n완료! 결과 → {OUT_CSV}")
