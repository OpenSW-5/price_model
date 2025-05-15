from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb

# 분류 모델 인스턴스 딕셔너리 반환 함수
def get_models(num_pos: int, num_neg: int):
    scale_pos_weight = num_neg / num_pos if num_pos > 0 else 1  # 0으로 나누는 경우 방지

    return {
        'sgd': SGDClassifier(loss='log_loss', penalty='elasticnet', class_weight='balanced', random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        'svc': SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42),
        'xgboost': xgb.XGBClassifier(
            n_estimators=100,
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight,
            random_state=42
        ),
        'lightgbm': lgb.LGBMClassifier(n_estimators=100, class_weight='balanced', verbose=-1, random_state=42),
        'mlp': MLPClassifier(
            hidden_layer_sizes=(512, 256, 128, 64, 32, 32, 16),
            activation='relu',
            solver='adam',
            learning_rate_init=0.001,
            alpha=1e-4,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            max_iter=100000,
            tol=1e-4,
            random_state=42
        ),
    }

# 파이프라인 생성 함수
def make_classification_pipeline(selected_models, preprocessor, num_pos: int, num_neg: int):
    """
    selected_models: str 또는 list[str]
    preprocessor: sklearn-compatible transformer (예: ColumnTransformer)
    num_pos, num_neg: 클래스 비율을 위한 양성/음성 샘플 수
    """
    models = get_models(num_pos, num_neg)

    if isinstance(selected_models, list):
        estimators = [(name, models[name]) for name in selected_models]
        clf = VotingClassifier(estimators=estimators, voting='soft')
        return Pipeline([
            ('preprocess', preprocessor),
            ('voting', clf)
        ])
    else:
        return Pipeline([
            ('preprocess', preprocessor),
            ('model', models[selected_models])
        ])
