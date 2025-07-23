# eval_yolov12_ball.py
import os, json, csv, time, cv2, numpy as np, torch
from tqdm import tqdm
from pycocotools import mask as mask_utils
from ultralytics import YOLO  # YOLO‑v12n 을 그대로 로드 가능

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# ── 0. 경로 설정 ───────────────────────────────────────────────
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
VIDEO_ROOT    = "../sam2/videos"          # videos/<video>/00000.jpg …
GT_ROOT       = "../sam2/soccer_gt"      # pseudo_gt/<video>.json
YOLO_WEIGHTS  = "./runs/segment/train/weights/best.pt"
OUT_CSV       = "eval_yolo_soccer.csv"
RESULT_DIR    = "results1"           # 시각화 결과를 저장할 폴더
VIDEO_LIST    = sorted([d for d in os.listdir(VIDEO_ROOT)
                        if os.path.isdir(os.path.join(VIDEO_ROOT, d))])[:100]

BALL_CLASS_ID = 0  # COCO sports ball

# ── 1. YOLO 모델 로드 ──────────────────────────────────────────
# model = YOLO("yolo12n-seg.yaml").load(YOLO_WEIGHTS)
model = YOLO("runs/segment/train/weights/best.pt")
model.to(DEVICE)
model.overrides["conf"] = 0.25  # confidence threshold (원하면 조정)
# 모델 호출 시 'classes' 매개변수를 통해 BALL_CLASS_ID만 예측하도록 설정

# ── 2. 유틸 함수 ───────────────────────────────────────────────
def rle_to_mask(seg, h, w):
    if isinstance(seg, str):
        seg = {"counts": seg.encode("ascii"), "size": [h, w]}
    elif isinstance(seg["counts"], str):
        seg = {"counts": seg["counts"].encode("ascii"), "size": seg["size"]}
    return mask_utils.decode(seg).astype(np.uint8)

def iou_score(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return 0.0 if union == 0 else inter / union

def masks_from_yolo(result, h, w):
    """
    result.masks.data  : (N, H', W') float32, 값 0~1
    -> 스포츠 볼에 해당하는 마스크들의 합집합 반환 (없으면 zeros)
    
    - 모델 예측 시 'classes' 매개변수로 BALL_CLASS_ID만 예측하므로 
      별도 클래스 필터링은 필요하지 않음.
    - 예측 마스크의 해상도가 GT 이미지 해상도와 다르면 재조정.
    """
    if result.masks is None:
        return np.zeros((h, w), np.uint8)
    masks = result.masks.data.cpu().numpy()  # (N, H', W')
    if masks.size == 0:
        return np.zeros((h, w), np.uint8)
    union = np.any(masks > 0.5, axis=0).astype(np.uint8)
    # 예측 마스크 크기가 GT와 다르면 재조정 (cv2.resize: (width, height))
    if union.shape != (h, w):
         union = cv2.resize(union, (w, h), interpolation=cv2.INTER_NEAREST)
    return union

# ── 3. 평가 루프 ───────────────────────────────────────────────
rows = [("video", "mIoU", "presence_acc", "high_iou_ratio", "runtime_sec")]

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

    # ── 메트릭 초기화 ─────────────────────────────────────────
    iou_sum = pres_correct = high_iou_cnt = 0
    start_t = time.time()

    # ── 프레임별 추론 ─────────────────────────────────────────
    vdir = os.path.join(VIDEO_ROOT, vid)
    frame_files = sorted(os.listdir(vdir))

    for idx, fname in enumerate(frame_files):
        img_path = os.path.join(vdir, fname)
        # 모델 호출 시, classes 매개변수를 이용하여 BALL_CLASS_ID(32)만 예측함
        res = model(img_path, task="segment", verbose=False, device=DEVICE, classes=[BALL_CLASS_ID])[0]

        pred_mask = masks_from_yolo(res, h, w)
        gt_mask = rle_to_mask(gt_ann[idx]["segmentation"], h, w)

        iou = iou_score(gt_mask, pred_mask)
        iou_sum      += iou
        high_iou_cnt += iou >= 0.90
        pres_correct += int((gt_mask.sum() > 0) == (pred_mask.sum() > 0))

        # ── 30 프레임마다 시각화 결과 저장 ──────────────────────────
        if idx % 30 == 0:
            # 원본 이미지 로드
            orig_img = cv2.imread(img_path)
            if orig_img is not None:
                # 예측 마스크: 빨간색, GT 마스크: 초록색 오버레이
                pred_overlay = np.zeros_like(orig_img)
                pred_overlay[pred_mask > 0] = (0, 0, 255)  # Red
                gt_overlay = np.zeros_like(orig_img)
                gt_overlay[gt_mask > 0] = (0, 255, 0)       # Green

                # 원본 이미지에 오버레이 반영 (알파 블렌딩)
                vis_img = cv2.addWeighted(orig_img, 1.0, pred_overlay, 0.5, 0)
                vis_img = cv2.addWeighted(vis_img, 1.0, gt_overlay, 0.5, 0)

                # 영상 별 결과 폴더 생성 및 이미지 저장
                save_dir = os.path.join(RESULT_DIR, vid)
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, fname)
                cv2.imwrite(save_path, vis_img)

    runtime        = time.time() - start_t
    mIoU           = iou_sum / num_f
    presence_acc   = pres_correct / num_f
    high_iou_ratio = high_iou_cnt / num_f

    rows.append((vid,
                 f"{mIoU:.4f}",
                 f"{presence_acc:.4f}",
                 f"{high_iou_ratio:.4f}",
                 f"{runtime:.2f}"))

    print(f"{vid:15s} | mIoU {mIoU:.4f} | Acc {presence_acc:.4f} "
          f"| ≥0.9 {high_iou_ratio:.4f} | {runtime:.2f}s")

# ── 4. CSV 저장 ───────────────────────────────────────────────
with open(OUT_CSV, "w", newline="") as f:
    csv.writer(f).writerows(rows)
print(f"\n완료! 결과 → {OUT_CSV}")
