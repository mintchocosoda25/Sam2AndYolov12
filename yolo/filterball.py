import os
import shutil

# 기존 데이터셋 디렉토리
DATASET_DIR = "datasets/coco"  # 사용자 환경에 맞게 조정
LABELS_DIR_TRAIN = os.path.join(DATASET_DIR, "labels/train2017")
IMAGES_DIR_TRAIN = os.path.join(DATASET_DIR, "images/train2017")

# sports ball 클래스 ID (COCO에서 보통 32)
SPORTS_BALL_ID = 32

# 결과물 저장 디렉토리(새로운 커스텀 데이터셋)
CUSTOM_DATASET_DIR = "datasets/coco_sportsball"
CUSTOM_LABELS_TRAIN = os.path.join(CUSTOM_DATASET_DIR, "labels/train2017")
CUSTOM_IMAGES_TRAIN = os.path.join(CUSTOM_DATASET_DIR, "images/train2017")

os.makedirs(CUSTOM_LABELS_TRAIN, exist_ok=True)
os.makedirs(CUSTOM_IMAGES_TRAIN, exist_ok=True)

# 새로운 train 리스트를 만들어 저장할 파일
train_list_path = os.path.join(CUSTOM_DATASET_DIR, "train2017_sportsball.txt")
train_list_file = open(train_list_path, "w")

label_files = sorted(os.listdir(LABELS_DIR_TRAIN))
for label_name in label_files:
    if not label_name.endswith(".txt"):
        continue

    label_path = os.path.join(LABELS_DIR_TRAIN, label_name)
    with open(label_path, "r") as lf:
        lines = lf.readlines()

    # 스포츠 볼 ID만 골라내기
    filtered_lines = []
    for line in lines:
        if line.strip():
            tokens = line.strip().split()
            class_id = int(tokens[0])  # 기존 클래스 ID
            if class_id == SPORTS_BALL_ID:
                # 새로운 라벨에서는 클래스 ID를 0으로 매핑: sports ball 한 개만 남음
                tokens[0] = "0"  
                new_line = " ".join(tokens) + "\n"
                filtered_lines.append(new_line)

    # 필터링 결과가 하나 이상 있으면 해당 이미지, 라벨파일 복사
    if len(filtered_lines) > 0:
        # 원본 이미지 파일명: 라벨파일명과 동일(.txt -> .jpg/.png)
        # ultralytics 구조: 000000000001.txt -> 000000000001.jpg
        base_name = os.path.splitext(label_name)[0]  # 000000000001
        possible_exts = [".jpg", ".png", ".jpeg"]    # Coco는 주로 .jpg

        src_img_path = None
        for ext in possible_exts:
            candidate = os.path.join(IMAGES_DIR_TRAIN, base_name + ext)
            if os.path.isfile(candidate):
                src_img_path = candidate
                break

        if src_img_path is None:
            # 이미지가 없으면 스킵
            continue

        # 새 라벨 경로
        dst_label_path = os.path.join(CUSTOM_LABELS_TRAIN, label_name)
        # 새 이미지 경로
        dst_img_path = os.path.join(CUSTOM_IMAGES_TRAIN, os.path.basename(src_img_path))

        # 라벨파일 생성
        with open(dst_label_path, "w") as lf:
            lf.writelines(filtered_lines)

        # 이미지 복사
        shutil.copy2(src_img_path, dst_img_path)

        # 새 train2017_sportsball.txt에 이미지 절대경로(or 상대경로) 기록
        # (Ultralytics는 txt 파일 내에 이미지 경로를 기록해둠)
        train_list_file.write(dst_img_path + "\n")

train_list_file.close()
print(f"Done. Filtered train labels & images saved to {CUSTOM_DATASET_DIR}")
print(f"New train list saved to {train_list_path}")
