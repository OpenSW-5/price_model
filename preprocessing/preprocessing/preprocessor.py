# preprocessor.py

# 라이브러리 로드
import pandas as pd
import numpy as np

# 기본 경로 설정
_IN_KEYWORDS_CSV_PATH = r"./preprocessing/in_keywords.csv"
_MAP_KEYWORDS_CSV_PATH = r"./preprocessing/map_keywords.csv"

# 기본 전처리: 날짜·카테고리 정리, 결측/비공개 처리, 낙찰여부 생성, 중복 제거, 금액 float 변환
def _basic_preprocess(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    # 개찰일시 타입 변환
    df["개찰일시"] = pd.to_datetime(df["개찰일시"], format="%Y-%m-%d %H:%M", errors="coerce")
    df = df.sort_values("개찰일시").reset_index(drop=True)

    # 카테고리 처리
    df[["대분류", "중분류"]] = df["카테고리"].str.strip("[]").str.split(" / ", expand=True)

    # 칼럼 이름 변경
    df = df.drop(columns=["카테고리", "낙찰가율(%)"]).rename(columns={"최저입찰가 (예정가격)(원)": "최저입찰가", "낙찰가(원)": "낙찰가", "기관/담당부점": "기관"})

    # 결측 처리
    df[["최저입찰가", "낙찰가"]] = df[["최저입찰가", "낙찰가"]].replace("-", np.nan)

    # 비공개 제외
    df = df[df["최저입찰가"].notna() & df["중분류"].notna() & (df["최저입찰가"] != "비공개")]

    # 낙찰 여부 처리
    df["낙찰여부"] = df["입찰결과"].apply(lambda x: 1 if "낙찰" in str(x) else (0 if "유찰" in str(x) else np.nan))
    df = df[df["낙찰여부"].notna()].copy()
    df["낙찰여부"] = df["낙찰여부"].astype(int)
    df.drop(columns=["입찰결과"], inplace=True)

    # 낙찰가 처리
    dup = df.groupby("일련번호")["낙찰가"].apply(lambda x: x.notna().sum())
    dup = dup[dup > 1].index
    df = df[~df["일련번호"].isin(dup)].copy()

    # 타입 변환
    for col in ["최저입찰가", "낙찰가"]:
        df[col] = df[col].replace(",", "", regex=True).astype(float)

    return df

# 임시
def aggregate_group(df: pd.DataFrame, max_rounds: int = 3) -> pd.DataFrame:

    # 낙찰된 건
    successful_bids = df[df["낙찰가"].notna()]["일련번호"].unique()
    df = df.sort_values("개찰일시").reset_index(drop=True)

    row = {
        "일련번호": df.loc[0, "일련번호"],
        "대분류": df.loc[0, "대분류"],
        "중분류": df.loc[0, "중분류"],
        "물건정보": df.loc[0, "물건정보"],
        "낙찰가": df["낙찰가"].dropna().unique()[0] if df["낙찰가"].notna().any() else pd.NA,
    }

    bid_ended = False
    bid_prices = []

    row["최초입찰시기"] = df["개찰일시"].min()

    bid_success_row = df[df["낙찰여부"] == 1]
    if not bid_success_row.empty:
        row["낙찰개찰일시"] = bid_success_row["개찰일시"].values[0]
        row["낙찰차수"] = bid_success_row.index[0] + 1
    else:
        row["낙찰개찰일시"] = pd.NaT
        row["낙찰차수"] = pd.NA

    for _, r in df.iterrows():
        if bid_ended:
            bid_prices.append(pd.NA)
            continue
        bid_prices.append(r["최저입찰가"])
        if r["낙찰여부"] == 1:
            bid_ended = True

    for i in range(max_rounds):
        col_name = f"{i+1}차최저입찰가"
        row[col_name] = bid_prices[i] if i < len(bid_prices) else pd.NA

    row["최종최저입찰가"] = next((p for p in reversed(bid_prices) if pd.notna(p)), pd.NA)

    if row["일련번호"] in successful_bids:
        row["최종유찰여부"] = 0
    else:
        return None


# 일련번호 기준 집계: 일련번호별 라운드별 최저입찰가, 낙찰 정보, 최종유찰여부, 낙찰가율 계산
def _aggregate_summary(df: pd.DataFrame, max_rounds: int = 5) -> pd.DataFrame:

    def calc_rate(nak, base):
        if pd.isna(nak) or pd.isna(base):
            return pd.NA
        return nak / base * 100 if base else np.inf

    summary_rows = []
    sold_serials = df[df["낙찰가"].notna()]["일련번호"].unique()

    for serial_no, g in df.groupby("일련번호"):
        g = g.sort_values("개찰일시").reset_index(drop=True)
        first = g.iloc[0]

        # 낙찰 행 찾기
        win_idxs = g.index[g["낙찰여부"] == 1].tolist()
        if win_idxs:
            win_idx = win_idxs[0]
            win_row = g.loc[win_idx]
            낙찰개찰일시 = win_row["개찰일시"]
            낙찰차수 = win_idx + 1
            낙찰가 = win_row["낙찰가"]
        else:
            낙찰개찰일시, 낙찰차수, 낙찰가 = pd.NA, pd.NA, pd.NA

        # 차수별 최저입찰가 리스트 생성 (낙찰 이후는 NaN 유지)
        bid_prices: list[float | pd.NAType] = []
        bid_ended = False
        for _, r in g.iterrows():
            if bid_ended:
                bid_prices.append(pd.NA)
            else:
                bid_prices.append(r["최저입찰가"])
                if r["낙찰여부"] == 1:
                    bid_ended = True

        # 결과 행 구성
        row = {
            "일련번호": serial_no,
            "대분류": first["대분류"],
            "중분류": first["중분류"],
            "물건정보": first["물건정보"],
            "기관": first["기관"],
            "최초입찰시기": g["개찰일시"].min(),
            "낙찰개찰일시": 낙찰개찰일시,
            "낙찰차수": 낙찰차수,
            "낙찰가": 낙찰가,
        }

        # 차수별 최저입찰가 컬럼
        for i in range(1, max_rounds + 1):
            row[f"{i}차최저입찰가"] = bid_prices[i - 1] if i - 1 < len(bid_prices) else pd.NA

        # 최종최저입찰가: 마지막 유효값
        row["최종최저입찰가"] = next((p for p in reversed(bid_prices) if pd.notna(p)), pd.NA)

        # 최종유찰여부
        row["최종유찰여부"] = 0 if serial_no in sold_serials else 1

        # 낙찰가율
        row["낙찰가율_최초최저가기준"] = calc_rate(row["낙찰가"], row["1차최저입찰가"])
        row["낙찰가율_최종최저가기준"] = calc_rate(row["낙찰가"], row["최종최저입찰가"])

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)

# 타입 정리 및 낙찰가율 필터 적용
def _convert_types_and_filter(df_summary: pd.DataFrame, max_rounds: int = 5, min_rate: float = 10, max_rate: float = 1000) -> pd.DataFrame:

    # 타입 변환
    num_cols = ["낙찰가"] + [f"{i}차최저입찰가" for i in range(1, max_rounds + 1)] + ["최종최저입찰가", "낙찰가율_최초최저가기준", "낙찰가율_최종최저가기준"]
    for col in num_cols:
        if col in df_summary.columns:
            df_summary[col] = pd.to_numeric(df_summary[col], errors="coerce")
    if "낙찰개찰일시" in df_summary.columns:
        df_summary["낙찰개찰일시"] = pd.to_datetime(df_summary["낙찰개찰일시"], errors="coerce")
    
    # 낙찰가율 필터 적용
    mask = (
        (df_summary["낙찰가율_최초최저가기준"] > min_rate) &
        (df_summary["낙찰가율_최초최저가기준"] < max_rate) &
        (df_summary["낙찰가율_최종최저가기준"] > min_rate) &
        (df_summary["낙찰가율_최종최저가기준"] < max_rate)
    )
    return df_summary[mask].reset_index(drop=True)

# 기관 처리
def _classify_org(df_fillted: pd.DataFrame, df_in_name: str, df_map_name: str) -> pd.DataFrame:

    # csv 파일 읽기
    df_in = pd.read_csv(df_in_name, encoding="utf-8-sig")
    df_map = pd.read_csv(df_map_name, encoding="utf-8-sig")

    # 리스트로 변환
    map_list = list(df_map.to_dict(orient="records"))
    in_list = list(df_in.to_dict(orient="records"))

    # 데이터프레임 복사
    df_plot = df_fillted.copy()
    df_plot = df_plot[df_plot["기관"].notna()].copy()

    # 분류 함수
    def classify(org_name: str) -> str:
        # map_keywords: org_name이 keyword와 정확히 일치할 때
        for row in map_list:
            if org_name.strip() == row["keyword"]:
                return row["group"]
        # in_keywords: org_name에 keyword가 부분 포함되어 있을 때
        for row in in_list:
            if row["keyword"] in org_name:
                return row["group"]
        # 어느 경우에도 해당하지 않으면 "기타"
        return "기타"

    # 기관 처리
    df_plot["기관"] = df_plot["기관"].apply(classify)
    
    return df_plot


# 전체 전처리 파이프라인
def preprocessor(df: pd.DataFrame, max_rounds: int = 5, min_rate: float = 10, max_rate: float = 1000) -> pd.DataFrame:

    df = _basic_preprocess(df)
    df = _aggregate_summary(df, max_rounds=max_rounds)
    df = _convert_types_and_filter(df, max_rounds=max_rounds, min_rate=min_rate, max_rate=max_rate)
    df = _classify_org(df, _IN_KEYWORDS_CSV_PATH, _MAP_KEYWORDS_CSV_PATH)
    return df
