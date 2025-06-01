# onbid_map_carp_predict.py

import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# 경매 데이터프레임 입력 후 파이프라인 예측 수행 클래스: 인스턴스 생성시 모델 로드
class CarPricePredictor:

    # 생성자: 모델 로드
    def __init__(
        self,
        order: int,
        repo_id: str = "asteroidddd/onbid-map-carp",
        model_subpath: str = "models_by_order",
        pipeline_filename: str = "pipeline.pkl"
    ):

        # order를 1~5로 제한
        if order < 1:
            order = 1
        elif order > 5:
            order = 5
        self.order = order

        # 모델 파일 경로 생성 및 다운로드
        hf_path = f"{model_subpath}/order{order}/{pipeline_filename}"
        local_pipeline_path = hf_hub_download(repo_id=repo_id, filename=hf_path)

        # 파이프라인 로드
        self.pipeline = joblib.load(local_pipeline_path)

    # 예측: DataFrame 또는 Series 입력 후 낙찰가율을 담은 Series 반환
    def predict(self, df: pd.DataFrame) -> pd.Series:

        # 입력이 Series(한 행)인 경우, DataFrame으로 변환
        if isinstance(df, pd.Series):
            df = pd.DataFrame([df])

        # 원본 df를 변경하지 않도록 복사본 생성
        df_new = df.copy()

        # "최초입찰시기" 컬럼이 datetime이 아니면 변환
        if not pd.api.types.is_datetime64_any_dtype(df_new["최초입찰시기"]):
            df_new["최초입찰시기"] = pd.to_datetime(df_new["최초입찰시기"])

        # 날짜 컬럼으로부터 파생 변수 생성
        df_new["최초입찰_연도"] = df_new["최초입찰시기"].dt.year
        df_new["최초입찰_월"] = df_new["최초입찰시기"].dt.month
        df_new["최초입찰_일"] = df_new["최초입찰시기"].dt.day
        df_new["최초입찰_요일"] = df_new["최초입찰시기"].dt.weekday

        # 피처 컬럼 결정: 대분류, 중분류, 기관 + 날짜 파생 변수 + 해당 차수까지의 최저입찰가
        base_cols = ["대분류", "중분류", "소분류", "제조사", "차종", "기관"]
        date_derived_cols = ["최초입찰_연도", "최초입찰_월", "최초입찰_일", "최초입찰_요일"]

        if self.order == 1:
            bid_cols = ["1차최저입찰가"]
        elif self.order == 2:
            bid_cols = ["1차최저입찰가", "2차최저입찰가"]
        elif self.order == 3:
            bid_cols = ["1차최저입찰가", "2차최저입찰가", "3차최저입찰가"]
        elif self.order == 4:
            bid_cols = ["1차최저입찰가", "2차최저입찰가", "3차최저입찰가", "4차최저입찰가"]
        else:  # order == 5
            bid_cols = ["1차최저입찰가", "2차최저입찰가", "3차최저입찰가", "4차최저입찰가", "5차최저입찰가"]

        feature_cols = base_cols + date_derived_cols + bid_cols

        # 필수 컬럼 누락 검사
        missing = [c for c in feature_cols if c not in df_new.columns]
        if missing:
            raise ValueError(f"필수 컬럼 누락: {missing}")

        # 모델 입력용 DataFrame 생성
        X_new = df_new[feature_cols].copy()

        # 예측
        preds = self.pipeline.predict(X_new)

        # 원래 인덱스를 유지하면서 결과를 Series로 반환
        return pd.Series(preds, index=df.index, name="predicted_낙찰가율_최초최저가기준")
