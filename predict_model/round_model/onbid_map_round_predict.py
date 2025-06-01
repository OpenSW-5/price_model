# onbid_map_round_predict.py

import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# 경매 데이터프레임 입력 후 파이프라인 예측 수행 클래스: 인스턴스 생성시 모델 로드
class RoundPredictor:

    # 생성자: 모델 로드
    def __init__(
        self,
        repo_id: str = "asteroidddd/onbid-map-round",
        pipeline_filename: str = "auction_pipeline.pkl",
        label_encoder_filename: str = "label_encoder.pkl"
    ):
        
        # 모델 파일 다운로드
        pipeline_path = hf_hub_download(repo_id=repo_id, filename=pipeline_filename)
        label_path = hf_hub_download(repo_id=repo_id, filename=label_encoder_filename)

        # 모델 로드
        self.pipeline = joblib.load(pipeline_path)
        self.label_encoder = joblib.load(label_path)

    # 예측: 데이터프레임 입력 후 차수 반환
    def predict(self, df: pd.DataFrame) -> list:

        # 입력이 Series(한 행)인 경우, DataFrame으로 변환
        if isinstance(df, pd.Series):
            df = pd.DataFrame([df])

        # 원본 df를 변경하지 않도록 복사본 생성
        df_new = df.copy()

        # 날짜 컬럼을 datetime으로 변환 후 파생변수 생성
        df_new["최초입찰시기"] = pd.to_datetime(df_new["최초입찰시기"])
        df_new["최초입찰_연도"] = df_new["최초입찰시기"].dt.year
        df_new["최초입찰_월"] = df_new["최초입찰시기"].dt.month
        df_new["최초입찰_일"] = df_new["최초입찰시기"].dt.day
        df_new["최초입찰_요일"] = df_new["최초입찰시기"].dt.weekday

        # 학습 시 사용한 컬럼 순서대로 선택
        feature_cols = [
            "대분류", "중분류", "기관",
            "최초입찰_연도", "최초입찰_월", "최초입찰_일", "최초입찰_요일",
            "1차최저입찰가"
        ]
        X_new = df_new[feature_cols]

        # 예측
        y_pred_le = self.pipeline.predict(X_new)
        y_pred_labels = self.label_encoder.inverse_transform(y_pred_le)
        y_pred_ints = [int(label) for label in y_pred_labels]

        return y_pred_ints