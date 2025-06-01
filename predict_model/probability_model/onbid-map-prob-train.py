# onbid-map-prob-train.py

import os
import shutil
import stat
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
import joblib
from huggingface_hub import HfApi, Repository

# 환경 변수에서 Hugging Face 토큰 읽기
HF_REPO_NAME = "asteroidddd/onbid-map-prob"
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("환경 변수 HF_TOKEN이 설정되어 있지 않습니다.")

# KDE 학습용 함수 정의
def train_kde_models(df,
                     car_df,
                     value_col='낙찰가율_최초최저가기준',
                     major_col='대분류',
                     minor_col='중분류',
                     bandwidth=2.0,
                     num_grid=1000,
                     margin=10):

    # 전체 데이터 KDE 학습
    values_all = df[value_col].dropna().values.reshape(-1, 1)
    x_all_min = values_all.min() - margin
    x_all_max = values_all.max() + margin
    x_all = np.linspace(x_all_min, x_all_max, num_grid).reshape(-1, 1)

    kde_all = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    kde_all.fit(values_all)

    log_pdf_all = kde_all.score_samples(x_all)
    pdf_all = np.exp(log_pdf_all)
    dx_all = (x_all[1, 0] - x_all[0, 0])
    cdf_all = np.cumsum(pdf_all) * dx_all

    overall_dict = {
        'kde': kde_all,
        'x_range': x_all.flatten(),
        'cdf': cdf_all,
        'x_min': x_all_min,
        'x_max': x_all_max,
    }

    # 대분류별 KDE 학습: 자동차 car_df 사용
    major_dict = {}
    for major_cat, group in df.groupby(major_col):

        # 만약 대분류가 '자동차'라면 car_df 사용
        if str(major_cat) == '자동차':
            vals = car_df[value_col].dropna().values.reshape(-1, 1)
        else:
            vals = group[value_col].dropna().values.reshape(-1, 1)

        if len(vals) < 2:
            major_dict[major_cat] = overall_dict
            continue

        x_min = vals.min() - margin
        x_max = vals.max() + margin
        x_range = np.linspace(x_min, x_max, num_grid).reshape(-1, 1)

        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
        kde.fit(vals)

        log_pdf = kde.score_samples(x_range)
        pdf = np.exp(log_pdf)
        dx = x_range[1, 0] - x_range[0, 0]
        cdf = np.cumsum(pdf) * dx

        major_dict[major_cat] = {
            'kde': kde,
            'x_range': x_range.flatten(),
            'cdf': cdf,
            'x_min': x_min,
            'x_max': x_max,
        }

    # 중분류별 KDE 학습: 자동차 car_df 사용
    minor_dict = {}
    for minor_cat, group in df.groupby(minor_col):
        vals = group[value_col].dropna().values.reshape(-1, 1)
        if len(vals) < 2:
            minor_dict[minor_cat] = overall_dict
            continue

        x_min = vals.min() - margin
        x_max = vals.max() + margin
        x_range = np.linspace(x_min, x_max, num_grid).reshape(-1, 1)

        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
        kde.fit(vals)

        log_pdf = kde.score_samples(x_range)
        pdf = np.exp(log_pdf)
        dx = x_range[1, 0] - x_range[0, 0]
        cdf = np.cumsum(pdf) * dx

        minor_dict[minor_cat] = {
            'kde': kde,
            'x_range': x_range.flatten(),
            'cdf': cdf,
            'x_min': x_min,
            'x_max': x_max,
        }

    return overall_dict, major_dict, minor_dict

def rm_readonly(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)

# 메인
def main():
    # 데이터 불러오기
    df = None
    car_df = None

    # KDE 모델 학습
    overall_model, major_models, minor_models = train_kde_models(
        df=df,
        car_df=car_df,
        value_col='낙찰가율_최초최저가기준',
        major_col='대분류',
        minor_col='중분류',
        bandwidth=2.0,
        num_grid=1000,
        margin=10
    )

    # KDE 모델 저장
    os.makedirs("output/kde_models", exist_ok=True)
    joblib.dump(overall_model, "output/kde_models/overall_dict.pkl")
    joblib.dump(major_models,  "output/kde_models/major_dict.pkl")
    joblib.dump(minor_models,  "output/kde_models/minor_dict.pkl")
    print("KDE 모델 파일 저장 완료: output/kde_models/overall_dict.pkl, major_dict.pkl, minor_dict.pkl")

    # requirements.txt 작성
    deps = ["numpy", "pandas", "scikit-learn", "joblib", "huggingface_hub"]
    with open("requirements.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(deps))

    # Hugging Face 레포 생성 및 클론
    api = HfApi()
    try:
        api.create_repo(repo_id=HF_REPO_NAME, token=HF_TOKEN)
    except Exception:
        pass  # 이미 레포가 존재하면 무시

    local_dir = "hf_repo"
    if os.path.isdir(local_dir):
        shutil.rmtree(local_dir, onerror=rm_readonly)
    repo = Repository(local_dir=local_dir, clone_from=HF_REPO_NAME, use_auth_token=HF_TOKEN)

    # 모델 파일 및 스크립트 복사
    dst_models_dir = os.path.join(local_dir, "models")
    os.makedirs(dst_models_dir, exist_ok=True)

    for fname in ["overall_dict.pkl", "major_dict.pkl", "minor_dict.pkl"]:
        src = os.path.join("output/kde_models", fname)
        if os.path.isfile(src):
            shutil.copy(src, os.path.join(dst_models_dir, fname))

    # 스크립트 파일 및 requirements.txt 복사
    script_name = os.path.basename(__file__)
    shutil.copy(__file__, os.path.join(local_dir, script_name))
    shutil.copy("requirements.txt", os.path.join(local_dir, "requirements.txt"))

    # 커밋 및 푸시
    repo.git_add(auto_lfs_track=True)
    repo.git_commit("Add KDE ensemble models (overall, major, minor) + training script")
    repo.git_push()
    print("Hugging Face Hub에 KDE 모델 업로드 완료")

if __name__ == "__main__":
    main()
