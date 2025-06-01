# onbid_map_round_train.py

import os
import shutil
import stat
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from huggingface_hub import HfApi, Repository

# 환경 변수에서 Hugging Face 토큰 읽기
HF_REPO_NAME = "asteroidddd/onbid-map-round"
HF_TOKEN     = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("환경 변수 HF_TOKEN이 설정되어 있지 않습니다.")

# 이 스크립트의 경로
SCRIPT_PATH = os.path.abspath(__file__)

def rm_readonly(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)

def main():

    # 데이터 불러오기
    df = None

    # 라벨 인코딩 & 빈도 ≤ 10인 클래스 제거
    le_label = LabelEncoder()
    df["낙찰차수_LE"] = le_label.fit_transform(df["낙찰차수"])
    counts = df["낙찰차수_LE"].value_counts()
    rare = counts[counts <= 10].index.tolist()
    df = df[~df["낙찰차수_LE"].isin(rare)].reset_index(drop=True)

    # 날짜 파생 변수 생성
    df["최초입찰_연도"] = df["최초입찰시기"].dt.year
    df["최초입찰_월"]   = df["최초입찰시기"].dt.month
    df["최초입찰_일"]   = df["최초입찰시기"].dt.day
    df["최초입찰_요일"] = df["최초입찰시기"].dt.weekday
    df = df.drop(columns=["최초입찰시기"])

    # 피처/타깃 분리
    X = df[["대분류", "중분류", "기관",
            "최초입찰_연도", "최초입찰_월", "최초입찰_일", "최초입찰_요일",
            "1차최저입찰가"]]
    y = df["낙찰차수_LE"]

    # 전처리 + 모델 파이프라인
    cat_cols = ["대분류", "중분류", "기관"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("ohe", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ],
        remainder="passthrough"
    )
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(eval_metric="mlogloss", random_state=42))
    ])

    # 학습
    pipeline.fit(X, y)

    # 파이프라인 & 라벨 인코더 저장
    os.makedirs("output", exist_ok=True)
    pipeline_path = "output/auction_pipeline.pkl"
    label_path    = "output/label_encoder.pkl"
    joblib.dump(pipeline, pipeline_path)
    joblib.dump(le_label, label_path)

    # requirements.txt 작성
    deps = ["pandas", "scikit-learn", "xgboost", "joblib", "huggingface_hub"]
    with open("requirements.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(deps))

    # Hugging Face 레포 생성 시도
    api = HfApi()
    try:
        api.create_repo(repo_id=HF_REPO_NAME, token=HF_TOKEN)
    except:
        pass

    # 로컬에 레포 클론 (기존 삭제 시 read-only 처리)
    local_dir = "hf_repo"
    if os.path.isdir(local_dir):
        shutil.rmtree(local_dir, onerror=rm_readonly)
    repo = Repository(local_dir=local_dir, clone_from=HF_REPO_NAME, use_auth_token=HF_TOKEN)

    # 파일 복사
    for src in [SCRIPT_PATH, "requirements.txt", pipeline_path, label_path]:
        dst = os.path.join(local_dir, os.path.basename(src))
        shutil.copy(src, dst)

    # 커밋 및 푸시
    repo.git_add(auto_lfs_track=True)
    repo.git_commit("Add trained pipeline + preprocessing code")
    repo.git_push()

if __name__ == "__main__":
    main()
