import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


class MLModelManager:
    # Класс для управления всеми ML моделями

    def __init__(self):
        # === КЛАССИФИКАТОРЫ (предсказывают направление: рост/падение) ===
        self.classifiers = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest (Clf)': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
            'XGBoost (Clf)': xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42,
                                               use_label_encoder=False, eval_metric='logloss', verbosity=0),
            'LightGBM (Clf)': lgb.LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42,
                                                 verbose=-1),
            'CatBoost (Clf)': CatBoostClassifier(iterations=100, depth=4, learning_rate=0.05, random_seed=42,
                                                 verbose=False),
            'MLP (Clf)': MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
                                       max_iter=200, random_state=42, early_stopping=True,
                                       validation_fraction=0.1, verbose=False)
        }

        # === РЕГРЕССОРЫ (предсказывают доходность в процентах) ===
        self.regressors = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.01),
            'Random Forest (Reg)': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
            'XGBoost (Reg)': XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42,
                                          verbosity=0),
            'LightGBM (Reg)': LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42,
                                            verbose=-1),
            'CatBoost (Reg)': CatBoostRegressor(iterations=100, depth=4, learning_rate=0.05, random_seed=42,
                                                verbose=False),
            'MLP (Reg)': MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
                                      max_iter=200, random_state=42, early_stopping=True,
                                      validation_fraction=0.1, verbose=False)
        }

        # Здесь будут храниться результаты
        self.classification_results = {}
        self.regression_results = {}
        self.best_classifier_name = None
        self.best_regressor_name = None
        self.best_classifier = None
        self.best_regressor = None

    def train_classifiers(self, X_train, y_train):
        # Обучаем все классификаторы на тренировочных данных
        for name, model in self.classifiers.items():
            model.fit(X_train, y_train)

    def train_regressors(self, X_train, y_train):
        # Обучаем все регрессоры на тренировочных данных
        for name, model in self.regressors.items():
            model.fit(X_train, y_train)

    def evaluate_classifiers(self, X_train, X_test, y_train, y_test):
        # Оцениваем все классификаторы и выбираем лучший

        # Шаг 1: обучаем модели
        self.train_classifiers(X_train, y_train)

        # Шаг 2: делаем предсказания
        y_pred_dict = {}
        y_proba_dict = {}

        for name, model in self.classifiers.items():
            y_pred_dict[name] = model.predict(X_test)

            # Получаем вероятности (нужны для ROC-AUC)
            if hasattr(model, 'predict_proba'):
                y_proba_dict[name] = model.predict_proba(X_test)[:, 1]
            else:
                y_proba_dict[name] = model.predict(X_test)

        # Шаг 3: вычисляем метрики для каждой модели
        results = {}
        for name in self.classifiers.keys():
            accuracy = accuracy_score(y_test, y_pred_dict[name])
            precision = precision_score(y_test, y_pred_dict[name], zero_division=0)
            recall = recall_score(y_test, y_pred_dict[name], zero_division=0)
            f1 = f1_score(y_test, y_pred_dict[name], zero_division=0)

            # ROC-AUC может быть не определён, если все предсказания одинаковые
            roc_auc = 0.5
            if len(np.unique(y_proba_dict[name])) > 1:
                roc_auc = roc_auc_score(y_test, y_proba_dict[name])

            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc
            }

        self.classification_results = results

        # Шаг 4: выбираем лучший классификатор по ROC-AUC
        best_roc = -1
        best_name = None
        for name, metrics in results.items():
            if metrics['roc_auc'] > best_roc:
                best_roc = metrics['roc_auc']
                best_name = name

        self.best_classifier_name = best_name
        self.best_classifier = self.classifiers[best_name]

        return results, y_pred_dict, y_proba_dict

    def evaluate_regressors(self, X_train, X_test, y_train, y_test):
        # Оцениваем все регрессоры и выбираем лучший

        # Шаг 1: обучаем модели
        self.train_regressors(X_train, y_train)

        # Шаг 2: делаем предсказания и вычисляем метрики
        results = {}
        y_pred_dict = {}

        for name, model in self.regressors.items():
            y_pred = model.predict(X_test)
            y_pred_dict[name] = y_pred

            # Переводим доходность в проценты для наглядности
            y_test_pct = y_test * 100
            y_pred_pct = y_pred * 100

            mae = mean_absolute_error(y_test_pct, y_pred_pct)
            rmse = np.sqrt(mean_squared_error(y_test_pct, y_pred_pct))

            results[name] = {
                'MAE': mae,
                'RMSE': rmse
            }

        self.regression_results = results

        # Шаг 3: выбираем лучший регрессор по RMSE (чем меньше, тем лучше)
        best_rmse = float('inf')
        best_name = None
        for name, metrics in results.items():
            if metrics['RMSE'] < best_rmse:
                best_rmse = metrics['RMSE']
                best_name = name

        self.best_regressor_name = best_name
        self.best_regressor = self.regressors[best_name]

        return results, y_pred_dict

    def predict_direction(self, X_last):
        # Предсказывает направление (1 - рост, 0 - падение)
        return self.best_classifier.predict(X_last)[0]

    def predict_return(self, X_last):
        # Предсказывает доходность (в долях, например 0.05 = 5%)
        return self.best_regressor.predict(X_last)[0]