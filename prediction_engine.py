import pandas as pd
import numpy as np
import traceback
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from feature_engineer import FeatureEngineer
from ml_models import MLModelManager
from time_series_models import TimeSeriesModels
from data_cleaner import DataCleaner


class StockPredictionEngine:
    # Основной движок прогнозирования

    def __init__(self, df_prices, df_fund):
        # Сохраняем входные данные
        self.df_prices = df_prices
        self.df_fund = df_fund

        # Создаём вспомогательные объекты
        self.feature_engineer = FeatureEngineer()
        self.ml_manager = MLModelManager()
        self.time_series = TimeSeriesModels()
        self.scaler = StandardScaler()

        # Флаги и результаты
        self.is_trained = False
        self.trained_forecast_days = 5
        self.classification_results = None
        self.regression_results = None
        self.best_classifier_name = None
        self.best_regressor_name = None
        self.status_callback = None
        self.y_test = None
        self.y_proba_dict = None
        self.time_series_results = None

    def set_status_callback(self, callback):
        # Устанавливает функцию для отображения статуса в интерфейсе
        self.status_callback = callback

    def update_status(self, message):
        # Отображает сообщение о статусе (только в интерфейсе)
        if self.status_callback:
            self.status_callback(message)

    def prepare_data(self, forecast_days=5):
        # Подготавливает данные для обучения: создаёт признаки и целевые переменные

        all_data = []  # Список для данных всех компаний
        tickers = self.df_prices['ticker'].unique()
        total = len(tickers)

        # Перебираем все тикеры
        for i, ticker in enumerate(tickers):
            self.update_status(f"Обработка {ticker} ({i + 1}/{total})...")

            # Берём данные только для текущего тикера
            company_data = self.df_prices[self.df_prices['ticker'] == ticker].copy()

            # Создаём признаки
            company_data = self.feature_engineer.create_features(company_data, forecast_days)

            # Если данные успешно обработаны
            if company_data is not None:
                # Добавляем фундаментальные показатели, если есть
                if self.df_fund is not None:
                    fund_row = self.df_fund[self.df_fund['ticker'] == ticker]
                    if len(fund_row) > 0:
                        pe_val = DataCleaner.clean_numeric_string(fund_row.iloc[0].get('pe', np.nan))
                        company_data['pe'] = pe_val

                all_data.append(company_data)

        # Проверяем, что есть хоть какие-то данные
        if not all_data:
            return None, None, None, None, None, None

        # Объединяем данные всех компаний в один DataFrame
        df = pd.concat(all_data, ignore_index=True)

        # Выделяем признаки и целевые переменные
        X = df[self.feature_engineer.feature_cols].values
        y_direction = df['target_direction'].values  # Направление
        y_return = df['target_return'].values  # Доходность

        # Заменяем пропуски в признаках на нули
        for col in range(X.shape[1]):
            col_data = X[:, col]
            if np.isnan(col_data).any():
                col_data = np.nan_to_num(col_data, nan=0)
                X[:, col] = col_data

        return X, y_direction, y_return, df, self.feature_engineer.feature_cols

    def train(self, forecast_days=5):
        # Обучает все модели (классификаторы, регрессоры, ARIMA/SARIMA)

        try:
            self.trained_forecast_days = forecast_days

            # Шаг 1: подготовка данных
            self.update_status("Подготовка данных...")
            X, y_direction, y_return, df, feature_cols = self.prepare_data(forecast_days)

            # Проверяем, что данных достаточно
            if X is None or len(X) < 100:
                self.update_status("ОШИБКА: Недостаточно данных для обучения!")
                return False

            # Шаг 2: масштабирование признаков
            self.update_status("Масштабирование признаков...")
            X = self.scaler.fit_transform(X)

            # Шаг 3: разделение на обучающую и тестовую выборки (80% / 20%)
            split_idx = int(len(X) * 0.8)
            X_train = X[:split_idx]
            X_test = X[split_idx:]
            y_dir_train = y_direction[:split_idx]
            y_dir_test = y_direction[split_idx:]
            y_ret_train = y_return[:split_idx]
            y_ret_test = y_return[split_idx:]

            # Шаг 4: обучение классификаторов (для направления)
            self.update_status("Обучение классификаторов (6 моделей) для направления...")
            self.classification_results, y_pred_dict, self.y_proba_dict = self.ml_manager.evaluate_classifiers(
                X_train, X_test, y_dir_train, y_dir_test)
            self.best_classifier_name = self.ml_manager.best_classifier_name

            self.y_test = y_dir_test

            # Шаг 5: обучение регрессоров (для доходности)
            self.update_status("Обучение регрессоров (8 моделей) для прогноза доходности...")
            self.regression_results, y_pred_dict_reg = self.ml_manager.evaluate_regressors(
                X_train, X_test, y_ret_train, y_ret_test)
            self.best_regressor_name = self.ml_manager.best_regressor_name

            # Шаг 6: обучение временных моделей (ARIMA/SARIMA)
            self.update_status("Обучение временных моделей (ARIMA/SARIMA)...")

            # Для ARIMA используем логарифмическую доходность (более стабильна)
            y_log_return = np.log1p(y_return)
            y_log_train = y_log_return[:split_idx]
            y_log_test = y_log_return[split_idx:]
            self.time_series_results = self.evaluate_time_series(y_log_train, y_log_test)

            self.is_trained = True
            self.update_status("Обучение завершено!")

            return self.classification_results, self.regression_results, self.time_series_results

        except Exception as e:
            self.update_status(f"Ошибка: {str(e)[:50]}")
            traceback.print_exc()
            return False

    def evaluate_time_series(self, y_train, y_test):
        # Оценивает ARIMA и SARIMA модели

        results = {}

        # Проверяем, что данных достаточно
        if len(y_train) < 30:
            results['ARIMA'] = {'MAE': np.nan, 'RMSE': np.nan}
            results['SARIMA'] = {'MAE': np.nan, 'RMSE': np.nan}
            return results

        # === ARIMA ===
        try:
            # Обучаем модель
            self.time_series.fit_arima(y_train)
            # Делаем прогноз
            y_pred_arima = self.time_series.predict_arima(len(y_test))

            # Конвертируем логарифмическую доходность обратно в проценты
            y_test_return = (np.exp(y_test) - 1) * 100
            y_pred_return = (np.exp(y_pred_arima) - 1) * 100

            # Отбрасываем выбросы (доходность более 30% считаем выбросом)
            mask = np.abs(y_test_return) < 30
            y_test_clean = y_test_return[mask]
            y_pred_clean = y_pred_return[mask]

            if len(y_test_clean) > 0:
                results['ARIMA'] = {
                    'MAE': mean_absolute_error(y_test_clean, y_pred_clean),
                    'RMSE': np.sqrt(mean_squared_error(y_test_clean, y_pred_clean)),
                }
            else:
                results['ARIMA'] = {'MAE': np.nan, 'RMSE': np.nan}
        except Exception as e:
            results['ARIMA'] = {'MAE': np.nan, 'RMSE': np.nan}

        # === SARIMA ===
        try:
            self.time_series.fit_sarima(y_train)
            y_pred_sarima = self.time_series.predict_sarima(len(y_test))

            y_test_return = (np.exp(y_test) - 1) * 100
            y_pred_return = (np.exp(y_pred_sarima) - 1) * 100

            mask = np.abs(y_test_return) < 30
            y_test_clean = y_test_return[mask]
            y_pred_clean = y_pred_return[mask]

            if len(y_test_clean) > 0:
                results['SARIMA'] = {
                    'MAE': mean_absolute_error(y_test_clean, y_pred_clean),
                    'RMSE': np.sqrt(mean_squared_error(y_test_clean, y_pred_clean)),
                }
            else:
                results['SARIMA'] = {'MAE': np.nan, 'RMSE': np.nan}
        except Exception as e:
            results['SARIMA'] = {'MAE': np.nan, 'RMSE': np.nan}

        return results

    def predict(self, ticker, forecast_days):
        # Прогнозирует для конкретного тикера

        # Если модель ещё не обучена или горизонт изменился, переобучаем
        if not self.is_trained or forecast_days != self.trained_forecast_days:
            result = self.train(forecast_days)
            if not result:
                return None

        # Загружаем исторические данные для тикера
        hist_data = self.df_prices[self.df_prices['ticker'] == ticker].sort_values('date')
        if len(hist_data) == 0:
            return None

        # Дополняем пропущенные даты и интерполируем цены
        min_date = hist_data['date'].min()
        max_date = hist_data['date'].max()
        all_dates = pd.date_range(start=min_date, end=max_date, freq='B')
        hist_data = hist_data.set_index('date').reindex(all_dates)
        hist_data['price'] = hist_data['price'].interpolate(method='linear').ffill().bfill()
        hist_data = hist_data.reset_index()
        hist_data = hist_data.rename(columns={'index': 'date'})

        # Текущая цена и дата
        current_price = hist_data['price'].iloc[-1]
        last_date = hist_data['date'].iloc[-1]

        # Создаём признаки для текущего момента
        features_df = self.feature_engineer.create_features(hist_data.copy(), forecast_days)
        if features_df is None:
            return None

        # Берём последнюю строку признаков
        last_row = features_df.iloc[-1]

        # Формируем вектор признаков
        feature_vector = []
        for col in self.feature_engineer.feature_cols:
            # Проверяем, что колонка существует
            if col in last_row.index:
                feature_vector.append(last_row[col])
        feature_vector = np.nan_to_num(feature_vector, nan=0)

        # Масштабируем
        X_last = np.array(feature_vector).reshape(1, -1)
        X_last = self.scaler.transform(X_last)

        # Делаем предсказания
        pred_direction = self.ml_manager.predict_direction(X_last)  # Направление
        pred_return_ml = self.ml_manager.predict_return(X_last)  # Доходность от ML

        # Прогноз от ARIMA/SARIMA
        log_return_series = np.log(hist_data['price'] / hist_data['price'].shift(1)).dropna()
        arima_forecast_return, sarima_forecast_return = self.predict_regression(log_return_series, forecast_days)

        # Суммарная доходность за период
        pred_return_arima = 0
        if len(arima_forecast_return) > 0:
            pred_return_arima = np.exp(np.sum(arima_forecast_return)) - 1

        pred_return_sarima = 0
        if len(sarima_forecast_return) > 0:
            pred_return_sarima = np.exp(np.sum(sarima_forecast_return)) - 1

        # Собираем все прогнозы доходности
        all_returns = [pred_return_ml, pred_return_arima, pred_return_sarima]

        # Оставляем только корректные (не NaN и не слишком большие)
        valid_returns = []
        for r in all_returns:
            if not np.isnan(r) and abs(r) < 0.2:
                valid_returns.append(r)

        # Усредняем
        if len(valid_returns) > 0:
            pred_return = sum(valid_returns) / len(valid_returns)
        else:
            pred_return = 0

        # Ограничиваем максимальное изменение 10%
        if pred_return > 0.1:
            pred_return = 0.1
        if pred_return < -0.1:
            pred_return = -0.1

        # Финальная цена и направление
        predicted_price = current_price * (1 + pred_return)
        final_direction = 1
        if pred_return <= 0:
            final_direction = 0

        # Прогнозы цен по дням для графика
        arima_prices = [current_price]
        sarima_prices = [current_price]

        for ret in arima_forecast_return:
            arima_prices.append(arima_prices[-1] * np.exp(ret))
        for ret in sarima_forecast_return:
            sarima_prices.append(sarima_prices[-1] * np.exp(ret))

        # Убираем начальную цену из списка
        if len(arima_prices) > 1:
            arima_prices = arima_prices[1:]
        else:
            arima_prices = []

        if len(sarima_prices) > 1:
            sarima_prices = sarima_prices[1:]
        else:
            sarima_prices = []

        return {
            'current_price': current_price,
            'predicted_price': predicted_price,
            'predicted_return_ml': pred_return_ml * 100,
            'predicted_return_arima': pred_return_arima * 100,
            'predicted_return_sarima': pred_return_sarima * 100,
            'expected_return': pred_return,
            'direction': 'Рост' if final_direction == 1 else 'Падение',
            'last_date': last_date,
            'hist_data': hist_data,
            'forecast_days': forecast_days,
            'best_classifier': self.best_classifier_name,
            'best_regressor': self.best_regressor_name,
            'classification_results': self.classification_results,
            'regression_results': self.regression_results,
            'time_series_results': self.time_series_results,
            'arima_forecast_prices': arima_prices,
            'sarima_forecast_prices': sarima_prices,
            'y_test': self.y_test,
            'y_proba_dict': self.y_proba_dict
        }

    def predict_regression(self, y_series, steps):
        # Делает прогноз ARIMA и SARIMA
        self.time_series.fit_arima(y_series)
        arima_forecast = self.time_series.predict_arima(steps)

        self.time_series.fit_sarima(y_series)
        sarima_forecast = self.time_series.predict_sarima(steps)

        return arima_forecast, sarima_forecast