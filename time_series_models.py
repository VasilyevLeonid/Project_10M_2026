import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

# Отключаем предупреждения для чистоты вывода
warnings.filterwarnings('ignore')


class TimeSeriesModels:
    # Класс для работы с моделями временных рядов ARIMA и SARIMA

    def __init__(self):
        # Сохраняем обученные модели
        self.arima_model = None
        self.sarima_model = None

        # Порядки моделей (упрощённые для лучшей сходимости)
        self.arima_order = (1, 1, 1)  # (p, d, q) - авторегрессия, разность, скользящее среднее
        self.sarima_order = (1, 0, 1)  # (p, d, q) для несезонной части
        self.sarima_seasonal_order = (1, 0, 1, 5)  # (P, D, Q, s) для сезонной части

    def fit_arima(self, y):
        # Обучает ARIMA модель на данных y
        try:
            # Убираем пропущенные значения
            y_clean = y[~np.isnan(y)]

            # Проверяем, что данных достаточно
            if len(y_clean) < 30:
                return False

            # Создаём и обучаем модель
            self.arima_model = ARIMA(y_clean, order=self.arima_order)
            self.arima_model = self.arima_model.fit(method_kwargs={'maxiter': 500, 'disp': False})
            return True
        except Exception as e:
            return False

    def fit_sarima(self, y):
        # Обучает SARIMA модель на данных y
        try:
            y_clean = y[~np.isnan(y)]

            if len(y_clean) < 30:
                return False

            # Создаём и обучаем модель с сезонностью
            self.sarima_model = SARIMAX(y_clean, order=self.sarima_order,
                                        seasonal_order=self.sarima_seasonal_order,
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)
            self.sarima_model = self.sarima_model.fit(disp=False, maxiter=500)
            return True
        except Exception as e:
            return False

    def predict_arima(self, steps):
        # Делает прогноз на steps шагов вперёд
        try:
            # Если модель не обучена, возвращаем нули
            if self.arima_model is None:
                return np.zeros(steps)

            forecast = self.arima_model.forecast(steps=steps)
            # Ограничиваем прогноз, чтобы избежать взрывного роста
            forecast = np.clip(forecast, -0.1, 0.1)
            return forecast
        except:
            return np.zeros(steps)

    def predict_sarima(self, steps):
        # Делает прогноз на steps шагов вперёд
        try:
            if self.sarima_model is None:
                return np.zeros(steps)

            forecast = self.sarima_model.forecast(steps=steps)
            forecast = np.clip(forecast, -0.1, 0.1)
            return forecast
        except:
            return np.zeros(steps)