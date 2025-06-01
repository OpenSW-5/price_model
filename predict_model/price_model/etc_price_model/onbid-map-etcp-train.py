# onbid-map-etcp-train.py

import os
import shutil
import stat
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from huggingface_hub import HfApi, Repository

# 차수별 하이퍼파라미터 설정
ORDER_PARAMS = {
    1: {
        "objective": "reg:squarederror",
        "max_depth": 4,
        "learning_rate": 0.10,
        "n_estimators": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
    },
    2: {
        "objective": "reg:squarederror",
        "max_depth": 5,
        "learning_rate": 0.05,
        "n_estimators": 120,
        "subsample": 0.85,
        "colsample_bytree": 0.9,
        "random_state": 42,
    },
    3: {
        "objective": "reg:squarederror",
        "max_depth": 6,
        "learning_rate": 0.03,
        "n_estimators": 150,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": 42,
    },
    4: {
        "objective": "reg:squarederror",
        "max_depth": 7,
        "learning_rate": 0.02,
        "n_estimators": 180,
        "subsample": 0.9,
        "colsample_bytree": 0.95,
        "random_state": 42,
    },
    5: {
        "objective": "reg:squarederror",
        "max_depth": 8,
        "learning_rate": 0.01,
        "n_estimators": 200,
        "subsample": 0.95,
        "colsample_bytree": 0.95,
        "random_state": 42,
    },
}

# 환경 변수에서 Hugging Face 토큰 읽기
HF_REPO_NAME = "asteroidddd/onbid-map-etcp"
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

    # '자동차' 대분류 행 제거
    if "대분류" in df.columns:
        df = df[~(df["대분류"] == "자동차")].reset_index(drop=True)

    # '낙찰차수' 컬럼을 정수형으로 변환하고, 5 이상은 5로 통일
    df["낙찰차수"] = df["낙찰차수"].astype(int).apply(lambda x: x if x < 5 else 5)

    # 차수별 모델 학습 & 저장
    for order in [1, 2, 3, 4, 5]:

        # 해당 차수 데이터만 필터링
        subset = df[df["낙찰차수"] == order].copy().reset_index(drop=True)
        if subset.empty:
            print(f"차수 {order} 데이터가 없습니다. 건너뜁니다.")
            continue

        # 날짜 컬럼(datetime) 파생변수 생성
        if "최초입찰시기" in subset.columns:
            subset["최초입찰시기"] = pd.to_datetime(subset["최초입찰시기"])
            subset["최초입찰_연도"] = subset["최초입찰시기"].dt.year
            subset["최초입찰_월"] = subset["최초입찰시기"].dt.month
            subset["최초입찰_일"] = subset["최초입찰시기"].dt.day
            subset["최초입찰_요일"] = subset["최초입찰시기"].dt.weekday
            subset = subset.drop(columns=["최초입찰시기"])
        else:
            raise KeyError("최초입찰시기 컬럼이 데이터프레임에 없습니다.")

        # 사용할 피처 컬럼 결정
        base_cols = ["대분류", "중분류", "기관"]
        date_cols = ["최초입찰_연도", "최초입찰_월", "최초입찰_일", "최초입찰_요일"]
        if order == 1:
            bid_cols = ["1차최저입찰가"]
        elif order == 2:
            bid_cols = ["1차최저입찰가", "2차최저입찰가"]
        elif order == 3:
            bid_cols = ["1차최저입찰가", "2차최저입찰가", "3차최저입찰가"]
        elif order == 4:
            bid_cols = ["1차최저입찰가", "2차최저입찰가", "3차최저입찰가", "4차최저입찰가"]
        else:  # order == 5
            bid_cols = ["1차최저입찰가", "2차최저입찰가", "3차최저입찰가", "4차최저입찰가", "5차최저입찰가"]

        feature_cols = base_cols + date_cols + bid_cols
        X = subset[feature_cols].copy()
        y = subset["낙찰가율_최초최저가기준"].copy()

        # 전처리 + 모델 파이프라인 정의
        preprocessor = ColumnTransformer(
            transformers=[("ohe", OneHotEncoder(handle_unknown="ignore"), base_cols)],
            remainder="passthrough"
        )

        # 차수별 파라미터를 꺼내서 XGBRegressor 생성
        params = ORDER_PARAMS.get(order)
        model = XGBRegressor(**params)
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", model)
        ])

        # 전체 데이터로 학습
        pipeline.fit(X, y)
        print(f"차수 {order} 모델 학습 완료 (params: {params})")

        # 모델 저장
        output_dir = f"output/order{order}"
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(pipeline, os.path.join(output_dir, "pipeline.pkl"))
        print(f"  → pipeline.pkl 저장: {output_dir}/pipeline.pkl")

    # requirements.txt 작성
    deps = ["pandas", "scikit-learn", "xgboost", "joblib", "huggingface_hub"]
    with open("requirements.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(deps))

    # 레포 생성 시도
    api = HfApi()
    try:
        api.create_repo(repo_id=HF_REPO_NAME, token=HF_TOKEN)
    except Exception:
        pass

    # 로컬에 레포 클론 (기존 디렉토리 삭제 포함)
    local_dir = "hf_repo"
    if os.path.isdir(local_dir):
        shutil.rmtree(local_dir, onerror=rm_readonly)
    repo = Repository(local_dir=local_dir, clone_from=HF_REPO_NAME, use_auth_token=HF_TOKEN)

    # output/order{차수} 내 파일을 hf_repo/models_by_order/order{차수} 폴더로 복사
    for order in [1, 2, 3, 4, 5]:
        src_dir = f"output/order{order}"
        if not os.path.isdir(src_dir):
            continue

        dst_dir = os.path.join(local_dir, "models_by_order", f"order{order}")
        os.makedirs(dst_dir, exist_ok=True)

        src_file = os.path.join(src_dir, "pipeline.pkl")
        if os.path.isfile(src_file):
            shutil.copy(src_file, os.path.join(dst_dir, "pipeline.pkl"))

    # 스크립트 파일 및 requirements.txt도 함께 복사
    for src in [SCRIPT_PATH, "requirements.txt"]:
        dst = os.path.join(local_dir, os.path.basename(src))
        shutil.copy(src, dst)

    # 커밋 및 푸시
    repo.git_add(auto_lfs_track=True)
    repo.git_commit("Add trained pipelines for orders 1~5 (개별 파라미터) + training script")
    repo.git_push()
    print("Hugging Face Hub에 모델 업로드 완료")

if __name__ == "__main__":
    main()
