import pandas as pd
import shutil
from pathlib import Path

# === 1. 설정 ===

# 파일 경로 설정
csv_path = "data/bbox_with_bme_label.csv"   # split 정보가 들어 있는 CSV 파일 경로
source_root = Path("data/axSpA_classification_full_slice")  # 원본 환자별 폴더가 있는 경로
train_target_root = Path("bme_classification/dataset/train_fullslice")
test_target_root = Path("bme_classification/dataset/test_fullslice")

# === 2. 데이터 불러오기 ===

# bbox_with_bme_label.csv 불러오기
df = pd.read_csv(csv_path)
df['patient_id'] = df['patient_id'].astype(str).str.zfill(7)  # ID 포맷 맞추기

# patient_id, split만 뽑고 중복 제거
patient_split_df = df[['patient_id', 'split']].drop_duplicates()

# === 3. 타겟 폴더 생성 ===

train_target_root.mkdir(parents=True, exist_ok=True)
test_target_root.mkdir(parents=True, exist_ok=True)

# === 4. 환자별로 복사하기 ===

for idx, row in patient_split_df.iterrows():
    patient_id = row['patient_id']
    split = row['split'].lower()  # 소문자로 변환 (train/test)

    source_path = source_root / patient_id

    if split == 'train':
        target_path = train_target_root / patient_id
    elif split == 'test':
        target_path = test_target_root / patient_id
    else:
        print(f"❗ Warning: Unknown split '{split}' for patient {patient_id}")
        continue  # 알 수 없는 split이면 건너뜀

    # 복사 (타겟에 이미 존재하면 건너뜀)
    if source_path.exists():
        if not target_path.exists():
            shutil.copytree(source_path, target_path)
            print(f"✅ Copied {patient_id} to {split}")
        else:
            print(f"⚡ {patient_id} already exists in {split}, skipped.")
    else:
        print(f"❗ Source folder not found for patient {patient_id}")

print("\n🎯 모든 환자 폴더 복사가 완료되었습니다.")
