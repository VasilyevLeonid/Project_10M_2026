import pandas as pd
import numpy as np


class FeatureEngineer:
    # Класс для создания признаков из исторических данных

    def __init__(self):
        # Здесь будет список названий всех признаков
        self.feature_cols = None

    def create_features(self, company_data, forecast_days):
        # Создаёт технические индикаторы и целевые переменные

        # Проверяем, что данных достаточно для анализа
        if len(company_data) < 100:
            return None

        # Сортируем данные по дате
        company_data = company_data.sort_values('date').copy()

        # Дополняем пропущенные даты (только рабочие дни)
        min_date = company_data['date'].min()
        max_date = company_data['date'].max()
        all_dates = pd.date_range(start=min_date, end=max_date, freq='B')

        company_data = company_data.set_index('date')
        company_data = company_data.reindex(all_dates)
        # Заполняем пропуски цен линейной интерполяцией
        company_data['price'] = company_data['price'].interpolate(method='linear').ffill().bfill()

        # Снова проверяем, что после интерполяции данных достаточно
        if len(company_data) < 100:
            return None

        # Возвращаем индекс как обычную колонку
        company_data = company_data.reset_index()
        company_data = company_data.rename(columns={'index': 'date'})

        # === СКОЛЬЗЯЩИЕ СРЕДНИЕ ===
        for window in [5, 10, 20]:
            company_data[f'ma_{window}'] = company_data['price'].rolling(window).mean()
            company_data[f'std_{window}'] = company_data['price'].rolling(window).std()
            company_data[f'price_ma_ratio_{window}'] = company_data['price'] / (company_data[f'ma_{window}'] + 1e-6)

        # === МОМЕНТУМ (изменение цены за период) ===
        for period in [5, 10, 20]:
            company_data[f'momentum_{period}'] = company_data['price'].pct_change(period)

        # === ДОХОДНОСТЬ И ВОЛАТИЛЬНОСТЬ ===
        company_data['returns'] = company_data['price'].pct_change()
        for window in [5, 10, 20]:
            company_data[f'volatility_{window}'] = company_data['returns'].rolling(window).std()

        # === RSI (Индекс относительной силы) ===
        delta = company_data['price'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-6)
        company_data['rsi'] = 100 - (100 / (1 + rs))

        # === MACD (сходимость/расхождение скользящих средних) ===
        ema_12 = company_data['price'].ewm(span=12, adjust=False).mean()
        ema_26 = company_data['price'].ewm(span=26, adjust=False).mean()
        company_data['macd'] = ema_12 - ema_26
        company_data['macd_signal'] = company_data['macd'].ewm(span=9, adjust=False).mean()
        company_data['macd_diff'] = company_data['macd'] - company_data['macd_signal']

        # === BOLLINGER BANDS (полосы Боллинджера) ===
        company_data['bb_middle'] = company_data['price'].rolling(window=20).mean()
        bb_std = company_data['price'].rolling(window=20).std()
        company_data['bb_upper'] = company_data['bb_middle'] + 2 * bb_std
        company_data['bb_lower'] = company_data['bb_middle'] - 2 * bb_std
        company_data['bb_width'] = (company_data['bb_upper'] - company_data['bb_lower']) / (
                    company_data['bb_middle'] + 1e-6)
        company_data['bb_position'] = (company_data['price'] - company_data['bb_lower']) / (
                    company_data['bb_upper'] - company_data['bb_lower'] + 1e-6)

        # === ВРЕМЕННЫЕ ПРИЗНАКИ ===
        company_data['day_of_week'] = company_data['date'].dt.weekday
        company_data['is_monday'] = (company_data['date'].dt.weekday == 0).astype(int)
        company_data['is_friday'] = (company_data['date'].dt.weekday == 4).astype(int)
        company_data['month'] = company_data['date'].dt.month

        # === ATR (средний истинный диапазон) ===
        company_data['high'] = company_data['price'] * 1.005
        company_data['low'] = company_data['price'] * 0.995
        company_data['tr'] = np.maximum(
            company_data['high'] - company_data['low'],
            np.maximum(abs(company_data['high'] - company_data['price'].shift(1)),
                       abs(company_data['low'] - company_data['price'].shift(1)))
        )
        company_data['atr_14'] = company_data['tr'].rolling(14).mean()
        company_data['atr_ratio'] = company_data['atr_14'] / (company_data['price'] + 1e-6)

        # === ЦЕЛЕВЫЕ ПЕРЕМЕННЫЕ (то, что будем предсказывать) ===
        # Направление: 1 - рост, 0 - падение
        company_data['target_direction'] = (company_data['price'].shift(-forecast_days) > company_data['price']).astype(
            int)

        # Процентная доходность за период
        company_data['target_return'] = (company_data['price'].shift(-forecast_days) - company_data['price']) / \
                                        company_data['price']

        # Логарифмическая доходность (для ARIMA/SARIMA)
        company_data['target_log_return'] = np.log(company_data['price'].shift(-forecast_days) / company_data['price'])
        company_data['log_price'] = np.log(company_data['price'])

        # Убираем бесконечности и заполняем пропуски
        company_data = company_data.replace([np.inf, -np.inf], np.nan)
        company_data = company_data.ffill().bfill().fillna(0)

        # Удаляем строки, где нет целевого значения
        company_data = company_data[company_data['target_direction'].notna()]

        # Проверяем, что после очистки осталось достаточно данных
        if len(company_data) < 50:
            return None

        # Список колонок, которые НЕ нужно использовать как признаки
        exclude_cols = ['date', 'price', 'target_direction', 'target_return', 'target_log_return',
                        'log_price', 'pe', 'high', 'low', 'tr', 'cap']

        # Сохраняем список признаков (все числовые колонки, кроме исключённых)
        self.feature_cols = []
        for col in company_data.columns:
            # Проверяем, что колонка не в списке исключений
            if col not in exclude_cols:
                # Проверяем, что колонка числовая
                if company_data[col].dtype in ['float64', 'int64']:
                    self.feature_cols.append(col)

        return company_data