from sklearn.linear_model import ElasticNet, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb

# 회귀 모델 딕셔너리 반환 함수
def get_regression_models():
    return {
        'elasticnet': ElasticNet(random_state=42),
        'bayesian_ridge': BayesianRidge(),
        'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'svr': SVR(kernel='rbf'),
        'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=42, eval_metric='rmse'),
        'lightgbm': lgb.LGBMRegressor(n_estimators=100, verbose=-1, random_state=42),
        'mlp': MLPRegressor(
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

# 회귀용 파이프라인 생성 함수
def make_regression_pipeline(selected_models, preprocessor):
    """
    selected_models: str 또는 list[str]
    preprocessor: sklearn-compatible transformer
    """
    models = get_regression_models()

    if isinstance(selected_models, list):
        estimators = [(name, models[name]) for name in selected_models]
        return Pipeline([
            ('preprocess', preprocessor),
            ('voting', VotingRegressor(estimators=estimators))
        ])
    else:
        return Pipeline([
            ('preprocess', preprocessor),
            ('model', models[selected_models])
        ])
