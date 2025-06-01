# onbid_map_prob_predict.py

import pandas as pd
import joblib
import numpy as np
from huggingface_hub import hf_hub_download

# 누적확률(CDF) 계산 헬퍼 함수
def _get_cdf_from_dict(model_dict, value: float) -> float:
    x_min = model_dict['x_min']
    x_max = model_dict['x_max']
    x_range = model_dict['x_range']
    cdf = model_dict['cdf']

    if value <= x_min:
        return 0.0
    if value >= x_max:
        return 1.0

    idx = np.searchsorted(x_range, value, side='right') - 1
    return float(cdf[idx])


# 앙상블 확률 계산 헬퍼 함수
def _compute_ensemble_prob(
    overall_dict: dict,
    major_dict: dict,
    minor_dict: dict,
    major: str,
    minor: str,
    value: float,
    w_all: float,
    w_major: float,
    w_minor: float
) -> float:
    # 전체 모델에서의 CDF
    p_all = _get_cdf_from_dict(overall_dict, value)

    # 대분류 모델
    if major in major_dict:
        p_major = _get_cdf_from_dict(major_dict[major], value)
    else:
        p_major = p_all

    # 중분류 모델
    if minor in minor_dict:
        p_minor = _get_cdf_from_dict(minor_dict[minor], value)
    else:
        p_minor = p_all

    total_w = w_all + w_major + w_minor
    return (p_all * w_all + p_major * w_major + p_minor * w_minor) / total_w


# KDE 기반 확률 예측 클래스
class ProbPredictor:
    # 생성자
    def __init__(
        self,
        repo_id: str = "asteroidddd/onbid-map-prob",
        model_subpath: str = "models",
        weights: tuple = (1.0, 1.0, 1.0)
    ):

        # Hugging Face 허브에서 모델 파일 다운로드 (public repo이므로 토큰 불필요)
        overall_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{model_subpath}/overall_dict.pkl"
        )
        major_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{model_subpath}/major_dict.pkl"
        )
        minor_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{model_subpath}/minor_dict.pkl"
        )

        # 로컬로 가져온 pickle 파일을 로드
        self.overall_dict = joblib.load(overall_path)
        self.major_dict = joblib.load(major_path)
        self.minor_dict = joblib.load(minor_path)

        # 가중치 튜플 (w_all, w_major, w_minor)
        self.w_all, self.w_major, self.w_minor = weights

    # 예측
    def predict(self, df: pd.DataFrame) -> pd.Series:
        # Series 형태로 들어오면 DataFrame으로 변환
        if isinstance(df, pd.Series):
            df = pd.DataFrame([df])

        # 입력 DataFrame 복사
        df_new = df.copy()

        # 필수 컬럼 검사
        required_cols = ["대분류", "중분류", "낙찰가율_최초최저가기준"]
        missing = [c for c in required_cols if c not in df_new.columns]
        if missing:
            raise ValueError(f"필수 컬럼 누락: {missing}")

        # 결과를 담을 리스트
        probs = []
        for idx, row in df_new.iterrows():
            major = str(row["대분류"])
            minor = str(row["중분류"])
            value = float(row["낙찰가율_최초최저가기준"])

            prob = _compute_ensemble_prob(
                overall_dict=self.overall_dict,
                major_dict=self.major_dict,
                minor_dict=self.minor_dict,
                major=major,
                minor=minor,
                value=value,
                w_all=self.w_all,
                w_major=self.w_major,
                w_minor=self.w_minor
            )
            probs.append(prob)

        return pd.Series(probs, index=df_new.index, name="prob_낙찰가율_최초최저가기준")
