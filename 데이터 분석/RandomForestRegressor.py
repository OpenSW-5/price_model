import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.font_manager as fm
import platform

# 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')
else:
    plt.rc('font', family='NanumGothic')

plt.rcParams['axes.unicode_minus'] = False

# CSV 불러오기 및 통합
years = range(2015, 2021)
dfs = []
for year in years:
    filename = f'{year}.csv'
    temp_df = pd.read_csv(filename, encoding='utf-8')
    temp_df['연도'] = year
    dfs.append(temp_df)

all_data = pd.concat(dfs, ignore_index=True)

# 컬럼 이름 지정
all_data.columns = ['일련번호','카테고리','물건정보','최저입찰가 (예정가격)(원)',
                    '낙찰가(원)','낙찰가율(%)','입찰결과','개찰일시','미지정']

# 숫자 클렌징 함수
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

# 낙찰된 항목만 필터링
df = all_data[all_data['입찰결과'].str.contains('낙찰') & ~all_data['입찰결과'].str.contains('후취소')]
df = df.dropna(subset=['낙찰가(원)', '낙찰가율(%)', '최저입찰가 (예정가격)(원)', '물건정보'])

# 물건 키워드 생성
def extract_key(text):
    return text.strip() if isinstance(text, str) else '기타'

df['물건키'] = df['물건정보'].apply(extract_key)

# Label Encoding
le = LabelEncoder()
df['물건코드'] = le.fit_transform(df['물건키'])

# 학습 데이터 구성
X = df[['최저입찰가 (예정가격)(원)', '물건코드']]
y = df['낙찰가율(%)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 트리 기반 회귀 모델 학습
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 예측 및 평가
y_pred = model.predict(X_test)
print(f"트리 기반 모델 MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"트리 기반 모델 R²: {r2_score(y_test, y_pred):.2f}")

# 예측 함수
def 회귀_예측_낙찰가(물건정보, 최저입찰가):
    key = extract_key(물건정보)
    try:
        물건코드 = le.transform([key])[0]
    except ValueError:
        return None  # 학습되지 않은 물건 키

    input_df = pd.DataFrame([[최저입찰가, 물건코드]], columns=['최저입찰가 (예정가격)(원)', '물건코드'])
    예측율 = model.predict(input_df)[0]

    if 예측율 <= 0 or pd.isna(예측율):
        return None

    return round(최저입찰가 * (예측율 / 100), 0)

# 히스토그램 함수
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

# 테스트 예시
test_물건 = '현대자동차 중형승합 이-에어로타운 5899cc'
test_최저입찰가 = 8830000

예측가 = 회귀_예측_낙찰가(test_물건, test_최저입찰가)

if 예측가:
    print(f"[트리 기반 모델] {test_물건}의 예측 낙찰가: {예측가:,.0f}원")
else:
    print(f"[트리 기반 모델] 예측 불가.")

# 히스토그램 출력
plot_낙찰가율_히스토그램(df, test_물건)