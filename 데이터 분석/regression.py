import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform

# 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')  # Windows에서 글씨 설정
elif platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')    # macOS에서 설정
else:
    plt.rc('font', family='NanumGothic')     # 리눅스 등..

# 마이너스 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

# CSV 호출(데이터 합치기 위해)
years = range(2015, 2021)
dfs = []

for year in years:
    filename = f'{year}.csv'
    temp_df = pd.read_csv(filename, encoding='utf-8')
    temp_df['연도'] = year
    dfs.append(temp_df)

# 데이터프레임 통합
all_data = pd.concat(dfs, ignore_index=True)

# 컬럼 이름 지정
all_data.columns = ['일련번호','카테고리','물건정보','최저입찰가 (예정가격)(원)',
                    '낙찰가(원)','낙찰가율(%)','입찰결과','개찰일시','미지정']

# 숫자를 가져오는데 방해되는 요소 제거
def clean_numeric(val):
    if pd.isna(val):
        return None
    val = str(val).replace(',', '').replace('%', '').replace('\r', '').replace('\n', '').replace('이상', '')
    try:
        return float(val)
    except ValueError:
        return None

all_data['낙찰가율(%)'] = all_data['낙찰가율(%)'].apply(clean_numeric)
all_data['최저입찰가 (예정가격)(원)'] = all_data['최저입찰가 (예정가격)(원)'].replace(['비공개', '-'], None).apply(clean_numeric)
all_data['낙찰가(원)'] = all_data['낙찰가(원)'].replace('-', None).apply(clean_numeric)


# 낙찰만 필터링 + '낙찰후취소' 제거
df = all_data[all_data['입찰결과'].str.contains('낙찰') & ~all_data['입찰결과'].str.contains('후취소')]
df = df.dropna(subset=['낙찰가(원)'])

# 💡 물건 키워드 추출 함수
def extract_key(text):
    return text.strip() if isinstance(text, str) else '기타'


# 물건키 추출
df['물건키'] = df['물건정보'].apply(extract_key)

# 예측 함수: 최빈 낙찰가율 사용
def 예측_낙찰가(물건정보, 최저입찰가):
    key = extract_key(물건정보)
    해당물건 = df[df['물건키'] == key]

    if 해당물건.empty:
        return None

    # 평균 낙찰가율 (NaN 제거 후 계산)
    유효_낙찰가율 = 해당물건['낙찰가율(%)'].dropna()
    if 유효_낙찰가율.empty:
        return None

    avg_rate = 유효_낙찰가율.mean()
    
    # 평균이 NaN이면 예측 불가
    if pd.isna(avg_rate):
        return None

    예측값 = 최저입찰가 * (avg_rate / 100)
    return round(예측값, 0) if 예측값 > 0 else None


# 히스토그램 함수 추가
def plot_낙찰가율_히스토그램(df, 물건정보):
    key = extract_key(물건정보)
    subset = df[df['물건키'] == key]

    if subset.empty:
        print(f"[오류] '{key}'에 해당하는 데이터가 없습니다.")
        return

    mean_val = subset['낙찰가율(%)'].mean()

    plt.figure(figsize=(10, 6))
    plt.hist(subset['낙찰가율(%)'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'평균: {mean_val:.2f}%')
    plt.title(f"'{key}' 물건의 낙찰가율 분포", fontsize=15)
    plt.xlabel('낙찰가율(%)', fontsize=12)
    plt.ylabel('건수', fontsize=12)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()



# 대입 테스트
test_물건 = '현대자동차 중형승합 이-에어로타운 5899cc'
test_최저입찰가 = 8830000
예측가 = 예측_낙찰가(test_물건, test_최저입찰가)

print(f"{test_물건}의 예측 낙찰가: {예측가:,.0f}원")

# 히스토그램 출력
plot_낙찰가율_히스토그램(df, test_물건)
