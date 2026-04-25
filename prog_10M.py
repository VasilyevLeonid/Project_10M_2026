import sys
import pandas as pd
import numpy as np
import sqlite3
import traceback
import re
from datetime import datetime
from PyQt5 import QtCore, QtGui, QtWidgets
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score
import lightgbm as lgb
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import warnings

warnings.filterwarnings('ignore')

# Настройка русских шрифтов
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


class Database:
    def __init__(self, db_name="forecasts.db"):
        self.db_name = db_name
        self.create_table()

    def create_table(self):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS forecasts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                company_name TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                forecast_price REAL,
                expected_return REAL,
                direction TEXT,
                created_date TEXT NOT NULL,
                status TEXT DEFAULT 'active'
            )
        ''')
        conn.commit()
        conn.close()

    def save_forecast(self, ticker, company_name, timeframe, forecast_price=None, expected_return=None, direction=None):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        created_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute('''
            INSERT INTO forecasts (ticker, company_name, timeframe, forecast_price, expected_return, direction, created_date, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (ticker, company_name, timeframe, forecast_price, expected_return, direction, created_date, 'active'))
        conn.commit()
        forecast_id = cursor.lastrowid
        conn.close()
        return forecast_id

    def get_all_forecasts(self):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute(
            'SELECT id, ticker, company_name, timeframe, forecast_price, expected_return, direction, created_date FROM forecasts WHERE status = "active" ORDER BY id ASC')
        forecasts = cursor.fetchall()
        conn.close()
        return forecasts

    def delete_forecast(self, forecast_id):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM forecasts WHERE id = ?', (forecast_id,))
        conn.commit()
        conn.close()


def clean_numeric_string(value):
    """Очистка строковых числовых значений от лишних символов"""
    if pd.isna(value) or value == "" or value is None:
        return np.nan

    if isinstance(value, (int, float)):
        return float(value)

    # Преобразуем в строку
    str_val = str(value)

    # Удаляем пробелы и заменяем запятую на точку
    str_val = str_val.strip().replace(',', '.')

    # Удаляем всё кроме цифр, точки и минуса
    str_val = re.sub(r'[^\d.\-]', '', str_val)

    # Удаляем лишние точки
    parts = str_val.split('.')
    if len(parts) > 2:
        str_val = parts[0] + '.' + ''.join(parts[1:])

    try:
        return float(str_val)
    except:
        return np.nan


class StockPredictionEngine:
    """Движок для прогнозирования цен акций"""

    def __init__(self, df_prices, df_fund):
        self.df_prices = df_prices
        self.df_fund = df_fund
        self.models = {'return': None, 'direction': None}
        self.scaler = None
        self.feature_cols = None
        self.fundamental_cache = {}
        self.is_trained = False
        self.trained_forecast_days = 5

        # Создаём кэш фундаментальных данных с очисткой значений
        if self.df_fund is not None:
            for _, row in self.df_fund.iterrows():
                ticker = row.get('ticker', '')
                if ticker:
                    cap_val = row.get('Капитализация', np.nan)
                    pe_val = row.get('Цена/прибыль', np.nan)

                    # Очищаем значения
                    cap_val = clean_numeric_string(cap_val)
                    pe_val = clean_numeric_string(pe_val)

                    self.fundamental_cache[ticker] = {
                        'cap': cap_val,
                        'pe': pe_val
                    }

    def get_fundamental_features(self, ticker):
        if ticker in self.fundamental_cache:
            return self.fundamental_cache[ticker]['cap'], self.fundamental_cache[ticker]['pe']
        return np.nan, np.nan

    def create_features(self, ticker, forecast_days):
        """Создание признаков для обучения"""
        company_data = self.df_prices[self.df_prices['ticker'] == ticker].copy()

        if len(company_data) < 100:
            return None

        # Базовые индикаторы
        for window in [5, 10, 20, 50]:
            company_data[f'ma_{window}'] = company_data['price'].rolling(window).mean()
            company_data[f'std_{window}'] = company_data['price'].rolling(window).std()
            company_data[f'price_ma_ratio_{window}'] = company_data['price'] / (company_data[f'ma_{window}'] + 1e-6)

        for period in [3, 5, 10, 20]:
            company_data[f'momentum_{period}'] = company_data['price'].pct_change(period)

        company_data['returns'] = company_data['price'].pct_change()
        for window in [5, 10, 20]:
            company_data[f'volatility_{window}'] = company_data['returns'].rolling(window).std()

        # RSI
        delta = company_data['price'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-6)

        if len(company_data) >= 14:
            company_data['rsi'] = 100 - (100 / (1 + rs))
        else:
            company_data['rsi'] = 50

        # MACD
        ema_12 = company_data['price'].ewm(span=12, adjust=False).mean()
        ema_26 = company_data['price'].ewm(span=26, adjust=False).mean()
        company_data['macd'] = ema_12 - ema_26
        company_data['macd_signal'] = company_data['macd'].ewm(span=9, adjust=False).mean()
        company_data['macd_diff'] = company_data['macd'] - company_data['macd_signal']

        # Bollinger Bands
        company_data['bb_middle'] = company_data['price'].rolling(window=20).mean()
        bb_std = company_data['price'].rolling(window=20).std()
        company_data['bb_upper'] = company_data['bb_middle'] + 2 * bb_std
        company_data['bb_lower'] = company_data['bb_middle'] - 2 * bb_std
        company_data['bb_width'] = (company_data['bb_upper'] - company_data['bb_lower']) / (
                    company_data['bb_middle'] + 1e-6)
        company_data['bb_position'] = (company_data['price'] - company_data['bb_lower']) / (
                    company_data['bb_upper'] - company_data['bb_lower'] + 1e-6)

        # Календарные признаки
        company_data['day_of_week'] = company_data['date'].dt.weekday
        company_data['is_monday'] = (company_data['date'].dt.weekday == 0).astype(int)
        company_data['is_friday'] = (company_data['date'].dt.weekday == 4).astype(int)
        company_data['month'] = company_data['date'].dt.month
        company_data['quarter'] = (company_data['date'].dt.month - 1) // 3 + 1

        # Фундаментальные признаки
        cap, pe = self.get_fundamental_features(ticker)
        company_data['cap'] = cap
        company_data['pe'] = pe

        # Целевая переменная
        company_data['target_price'] = company_data['price'].shift(-forecast_days)
        company_data['target_return'] = (company_data['target_price'] / company_data['price']) - 1
        company_data['target_direction'] = (company_data['target_price'] > company_data['price']).astype(int)

        # Заполняем NaN (ИСПРАВЛЕНО ДЛЯ PANDAS 2.0+)
        company_data = company_data.replace([np.inf, -np.inf], np.nan)
        company_data = company_data.ffill().bfill().fillna(0)

        # Оставляем только строки с целевой переменной
        company_data = company_data[company_data['target_price'].notna()]

        if len(company_data) < 50:
            return None

        return company_data

    def create_prediction_features(self, ticker):
        """Создание признаков для прогноза"""
        company_data = self.df_prices[self.df_prices['ticker'] == ticker].copy()

        if len(company_data) < 100:
            return None

        for window in [5, 10, 20, 50]:
            company_data[f'ma_{window}'] = company_data['price'].rolling(window).mean()
            company_data[f'std_{window}'] = company_data['price'].rolling(window).std()
            company_data[f'price_ma_ratio_{window}'] = company_data['price'] / (company_data[f'ma_{window}'] + 1e-6)

        for period in [3, 5, 10, 20]:
            company_data[f'momentum_{period}'] = company_data['price'].pct_change(period)

        company_data['returns'] = company_data['price'].pct_change()
        for window in [5, 10, 20]:
            company_data[f'volatility_{window}'] = company_data['returns'].rolling(window).std()

        delta = company_data['price'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-6)

        if len(company_data) >= 14:
            company_data['rsi'] = 100 - (100 / (1 + rs))
        else:
            company_data['rsi'] = 50

        ema_12 = company_data['price'].ewm(span=12, adjust=False).mean()
        ema_26 = company_data['price'].ewm(span=26, adjust=False).mean()
        company_data['macd'] = ema_12 - ema_26
        company_data['macd_signal'] = company_data['macd'].ewm(span=9, adjust=False).mean()
        company_data['macd_diff'] = company_data['macd'] - company_data['macd_signal']

        company_data['bb_middle'] = company_data['price'].rolling(window=20).mean()
        bb_std = company_data['price'].rolling(window=20).std()
        company_data['bb_upper'] = company_data['bb_middle'] + 2 * bb_std
        company_data['bb_lower'] = company_data['bb_middle'] - 2 * bb_std
        company_data['bb_width'] = (company_data['bb_upper'] - company_data['bb_lower']) / (
                    company_data['bb_middle'] + 1e-6)
        company_data['bb_position'] = (company_data['price'] - company_data['bb_lower']) / (
                    company_data['bb_upper'] - company_data['bb_lower'] + 1e-6)

        company_data['day_of_week'] = company_data['date'].dt.weekday
        company_data['is_monday'] = (company_data['date'].dt.weekday == 0).astype(int)
        company_data['is_friday'] = (company_data['date'].dt.weekday == 4).astype(int)
        company_data['month'] = company_data['date'].dt.month
        company_data['quarter'] = (company_data['date'].dt.month - 1) // 3 + 1

        cap, pe = self.get_fundamental_features(ticker)
        company_data['cap'] = cap
        company_data['pe'] = pe

        # Заполняем NaN (ИСПРАВЛЕНО ДЛЯ PANDAS 2.0+)
        company_data = company_data.replace([np.inf, -np.inf], np.nan)
        company_data = company_data.ffill().bfill().fillna(0)

        return company_data

    def train(self, forecast_days=5):
        """Обучение модели на указанное количество дней"""
        try:
            self.trained_forecast_days = forecast_days
            all_data = []
            tickers = self.df_prices['ticker'].unique()

            print(f"\n{'=' * 60}")
            print(f"Начинаем обучение на {len(tickers)} тикерах для прогноза на {forecast_days} дней...")
            print(f"{'=' * 60}")

            successful_tickers = 0

            for ticker in tickers:
                try:
                    company_data = self.create_features(ticker, forecast_days)
                    if company_data is not None:
                        successful_tickers += 1
                        print(f"  ✓ {ticker}: добавлено {len(company_data)} примеров")
                        for idx, row in company_data.iterrows():
                            row_data = {
                                'target_return': row['target_return'],
                                'target_direction': row['target_direction'],
                                'ma_5': row['ma_5'], 'ma_10': row['ma_10'], 'ma_20': row['ma_20'],
                                'ma_50': row['ma_50'],
                                'std_5': row['std_5'], 'std_10': row['std_10'], 'std_20': row['std_20'],
                                'price_ma_ratio_5': row['price_ma_ratio_5'],
                                'price_ma_ratio_10': row['price_ma_ratio_10'],
                                'price_ma_ratio_20': row['price_ma_ratio_20'],
                                'price_ma_ratio_50': row['price_ma_ratio_50'],
                                'momentum_3': row['momentum_3'], 'momentum_5': row['momentum_5'],
                                'momentum_10': row['momentum_10'], 'momentum_20': row['momentum_20'],
                                'volatility_5': row['volatility_5'], 'volatility_10': row['volatility_10'],
                                'volatility_20': row['volatility_20'],
                                'rsi': row['rsi'], 'macd': row['macd'], 'macd_signal': row['macd_signal'],
                                'macd_diff': row['macd_diff'], 'bb_width': row['bb_width'],
                                'bb_position': row['bb_position'],
                                'day_of_week': row['day_of_week'], 'is_monday': row['is_monday'],
                                'is_friday': row['is_friday'], 'month': row['month'], 'quarter': row['quarter'],
                                'cap': row['cap'], 'pe': row['pe'],
                            }
                            all_data.append(row_data)
                    else:
                        print(f"  ✗ {ticker}: недостаточно данных для обучения")
                except Exception as e:
                    print(f"  ✗ {ticker}: ошибка - {str(e)}")

            print(f"\n{'=' * 60}")
            print(f"Успешно обработано тикеров: {successful_tickers} из {len(tickers)}")
            print(f"Всего собрано записей: {len(all_data)}")
            print(f"{'=' * 60}")

            if len(all_data) < 100:
                print("ОШИБКА: Недостаточно данных для обучения!")
                return False

            # Создаем DataFrame
            print("\nСоздание DataFrame...")
            df = pd.DataFrame(all_data)
            print(f"  Формат DataFrame: {df.shape}")

            # Подготовка признаков
            exclude_cols = ['target_return', 'target_direction']
            self.feature_cols = [c for c in df.columns if c not in exclude_cols]
            print(f"  Количество признаков: {len(self.feature_cols)}")

            # Заполнение пропусков
            print("  Заполнение пропусков...")
            for col in self.feature_cols:
                if df[col].isna().any():
                    median_val = df[col].median()
                    if pd.isna(median_val) or np.isinf(median_val):
                        median_val = 0
                    df[col] = df[col].fillna(median_val)

                # Принудительно преобразуем в числовой тип
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

            # Масштабирование
            print("  Масштабирование признаков...")
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(df[self.feature_cols])
            y_return = df['target_return'].values
            y_direction = df['target_direction'].values

            print(f"  Размер X: {X.shape}")
            print(
                f"  Распределение направлений: Рост = {(y_direction == 1).sum()}, Падение = {(y_direction == 0).sum()}")

            # Разделение на train/test
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_return_train, y_return_test = y_return[:split_idx], y_return[split_idx:]
            y_direction_train, y_direction_test = y_direction[:split_idx], y_direction[split_idx:]

            print(f"  Обучающая выборка: {len(X_train)} примеров")
            print(f"  Тестовая выборка: {len(X_test)} примеров")

            # Обучение модели регрессии
            print("\nОбучение модели регрессии (LightGBM)...")
            self.models['return'] = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.05,
                random_state=42,
                verbose=-1
            )
            self.models['return'].fit(X_train, y_return_train)

            # Обучение модели классификации
            print("Обучение модели классификации (LightGBM)...")
            self.models['direction'] = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                random_state=42,
                verbose=-1
            )
            self.models['direction'].fit(X_train, y_direction_train)

            # Оценка
            y_return_pred = self.models['return'].predict(X_test)
            y_direction_pred = self.models['direction'].predict(X_test)

            r2 = r2_score(y_return_test, y_return_pred)
            acc = accuracy_score(y_direction_test, y_direction_pred)

            self.is_trained = True
            print(f"\n{'=' * 60}")
            print(f"ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
            print(f"  R² (доходность): {r2:.4f}")
            print(f"  Accuracy (направление): {acc:.4f}")
            print(f"  Размер выборки: {len(df)} записей")
            print(f"{'=' * 60}\n")

            return {'r2': r2, 'accuracy': acc, 'samples': len(df)}

        except Exception as e:
            print(f"\nКРИТИЧЕСКАЯ ОШИБКА ПРИ ОБУЧЕНИИ: {str(e)}")
            traceback.print_exc()
            return False

    def predict(self, ticker, forecast_days):
        """Прогноз на указанное количество дней"""
        if not self.is_trained:
            print("Модель не обучена!")
            return None

        print(f"\n--- Прогноз для {ticker} на {forecast_days} дней ---")

        # Если прогноз на другое количество дней, переобучаем модель
        if forecast_days != self.trained_forecast_days:
            print(f"Переобучение модели для прогноза на {forecast_days} дней...")
            metrics = self.train(forecast_days)
            if not metrics:
                return None

        hist_data = self.df_prices[self.df_prices['ticker'] == ticker].sort_values('date')
        if len(hist_data) == 0:
            print(f"Нет исторических данных для {ticker}")
            return None

        current_price = hist_data['price'].iloc[-1]
        last_date = hist_data['date'].iloc[-1]
        print(f"Текущая цена: {current_price}, дата: {last_date}")

        company_features = self.create_prediction_features(ticker)
        if company_features is None or len(company_features) == 0:
            print(f"Не удалось создать признаки для {ticker}")
            return None

        last_row = company_features.iloc[-1]

        feature_vector = []
        for feat in self.feature_cols:
            if feat in last_row.index:
                val = last_row[feat]
                if pd.isna(val):
                    val = 0
                feature_vector.append(val)
            else:
                feature_vector.append(0)

        X_last = np.array(feature_vector).reshape(1, -1)
        X_last = self.scaler.transform(X_last)

        pred_return = self.models['return'].predict(X_last)[0]
        pred_direction = self.models['direction'].predict(X_last)[0]
        pred_proba = self.models['direction'].predict_proba(X_last)[0, 1]

        predicted_price = current_price * (1 + pred_return)
        print(
            f"Прогноз: цена={predicted_price:.2f}, изменение={pred_return:.2%}, направление={'Рост' if pred_direction == 1 else 'Падение'}")

        return {
            'current_price': current_price,
            'predicted_price': predicted_price,
            'expected_return': pred_return,
            'direction': 'Рост' if pred_direction == 1 else 'Падение',
            'confidence': pred_proba if pred_direction == 1 else 1 - pred_proba,
            'last_date': last_date,
            'hist_data': hist_data,
            'forecast_days': forecast_days
        }


class ForecastResultWin(QtWidgets.QWidget):
    """Окно с результатами прогноза и графиками"""

    def __init__(self, ticker, company_name, forecast_result, db):
        super().__init__()
        self.ticker = ticker
        self.company_name = company_name
        self.forecast_result = forecast_result
        self.db = db
        self.setWindowTitle(f"Результат прогноза - {ticker}")
        self.setGeometry(200, 100, 1200, 800)

        self.setStyleSheet("""
            QWidget {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1a1a2e, stop:1 #16213e);
            }
            QLabel {
                color: white;
            }
            QPushButton {
                background-color: #e94560;
                color: white;
                border-radius: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #ff6b6b;
            }
            QTabWidget::pane {
                background-color: transparent;
                border: none;
            }
            QTabBar::tab {
                background-color: #0f3460;
                color: white;
                padding: 10px;
                margin: 5px;
                border-radius: 10px;
            }
            QTabBar::tab:selected {
                background-color: #e94560;
            }
        """)

        layout = QtWidgets.QVBoxLayout()

        # Заголовок
        title = QtWidgets.QLabel(f"📊 ПРОГНОЗ ДЛЯ {ticker}")
        title.setFont(QtGui.QFont("Arial", 18, QtGui.QFont.Bold))
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet("color: #FF9800; padding: 20px;")
        layout.addWidget(title)

        # Название компании
        company = QtWidgets.QLabel(company_name[:60])
        company.setAlignment(QtCore.Qt.AlignCenter)
        company.setStyleSheet("color: #00ff88; font-size: 14px; padding: 5px;")
        layout.addWidget(company)

        # Вкладки с графиками
        tabs = QtWidgets.QTabWidget()

        # Вкладка с результатами
        result_tab = self.create_result_tab()
        tabs.addTab(result_tab, "📊 Результаты")

        # Вкладка с графиком цены
        price_tab = self.create_price_chart_tab()
        tabs.addTab(price_tab, "📈 График цены")

        # Вкладка с индикаторами
        indicators_tab = self.create_indicators_tab()
        tabs.addTab(indicators_tab, "📉 Технические индикаторы")

        layout.addWidget(tabs)

        # Кнопки
        btn_layout = QtWidgets.QHBoxLayout()

        save_btn = QtWidgets.QPushButton("💾 Сохранить прогноз")
        save_btn.setFixedSize(200, 45)
        save_btn.setFont(QtGui.QFont("Arial", 12, QtGui.QFont.Bold))
        save_btn.setStyleSheet("background-color: #FF9800;")
        save_btn.clicked.connect(self.save_forecast)
        btn_layout.addWidget(save_btn)

        close_btn = QtWidgets.QPushButton("❌ Закрыть")
        close_btn.setFixedSize(120, 45)
        close_btn.clicked.connect(self.close)
        btn_layout.addWidget(close_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def create_result_tab(self):
        """Создание вкладки с результатами"""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        # Карточка с результатами
        card = QtWidgets.QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: rgba(15, 52, 96, 0.8);
                border-radius: 15px;
                padding: 30px;
                margin: 20px;
            }
        """)
        card_layout = QtWidgets.QVBoxLayout(card)

        # Текущая цена
        current = QtWidgets.QLabel(f"💰 Текущая цена: ${self.forecast_result['current_price']:.2f}")
        current.setFont(QtGui.QFont("Arial", 16))
        current.setAlignment(QtCore.Qt.AlignCenter)
        card_layout.addWidget(current)

        # Прогноз
        forecast = QtWidgets.QLabel(f"🔮 Прогнозируемая цена: ${self.forecast_result['predicted_price']:.2f}")
        forecast.setFont(QtGui.QFont("Arial", 18, QtGui.QFont.Bold))
        forecast.setAlignment(QtCore.Qt.AlignCenter)
        forecast.setStyleSheet("color: #FF9800; padding: 10px;")
        card_layout.addWidget(forecast)

        # Изменение
        change = self.forecast_result['expected_return'] * 100
        change_text = f"📈 Ожидаемое изменение: {change:+.2f}%"
        change_color = "#00ff88" if change > 0 else "#ff6b6b"
        change_label = QtWidgets.QLabel(change_text)
        change_label.setFont(QtGui.QFont("Arial", 14))
        change_label.setAlignment(QtCore.Qt.AlignCenter)
        change_label.setStyleSheet(f"color: {change_color}; padding: 5px;")
        card_layout.addWidget(change_label)

        # Направление
        direction_text = f"🎯 Направление: {self.forecast_result['direction']}"
        direction_label = QtWidgets.QLabel(direction_text)
        direction_label.setFont(QtGui.QFont("Arial", 14))
        direction_label.setAlignment(QtCore.Qt.AlignCenter)
        card_layout.addWidget(direction_label)

        # Уверенность
        confidence = self.forecast_result['confidence'] * 100
        confidence_label = QtWidgets.QLabel(f"🎲 Уверенность: {confidence:.1f}%")
        confidence_label.setFont(QtGui.QFont("Arial", 14))
        confidence_label.setAlignment(QtCore.Qt.AlignCenter)
        card_layout.addWidget(confidence_label)

        # Горизонт прогноза
        days = self.forecast_result['forecast_days']
        horizon_label = QtWidgets.QLabel(f"📆 Горизонт прогноза: {days} дней")
        horizon_label.setFont(QtGui.QFont("Arial", 12))
        horizon_label.setAlignment(QtCore.Qt.AlignCenter)
        horizon_label.setStyleSheet("color: #888; padding: 5px;")
        card_layout.addWidget(horizon_label)

        layout.addWidget(card)
        layout.addStretch()

        return tab

    def create_price_chart_tab(self):
        """Создание вкладки с графиком цены и прогноза"""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        # Создаём фигуру matplotlib
        figure = Figure(figsize=(10, 6), dpi=100)
        canvas = FigureCanvas(figure)
        toolbar = NavigationToolbar(canvas, self)

        layout.addWidget(toolbar)
        layout.addWidget(canvas)

        # Строим график
        hist_data = self.forecast_result['hist_data']
        forecast_days = self.forecast_result['forecast_days']
        predicted_price = self.forecast_result['predicted_price']
        pred_return = self.forecast_result['expected_return']
        pred_direction = 1 if self.forecast_result['direction'] == 'Рост' else 0

        plot_days = min(100, len(hist_data))
        plot_data = hist_data.tail(plot_days)

        ax = figure.add_subplot(111)
        ax.plot(plot_data['date'], plot_data['price'], 'b-', linewidth=2, label='Исторические данные')

        last_date = plot_data['date'].iloc[-1]
        last_price = plot_data['price'].iloc[-1]

        # Генерируем даты для прогноза (только рабочие дни)
        forecast_dates = []
        current_date = last_date
        while len(forecast_dates) < forecast_days:
            current_date = current_date + pd.Timedelta(days=1)
            if current_date.weekday() < 5:
                forecast_dates.append(current_date)

        # Детерминированное линейное изменение (без шума)
        forecast_prices = []
        current = last_price
        daily_return = pred_return / forecast_days if forecast_days > 0 else 0

        for i in range(len(forecast_dates)):
            current = current * (1 + daily_return)
            forecast_prices.append(current)

        ax.plot(forecast_dates, forecast_prices, 'r--', linewidth=2, marker='o', markersize=6, label='Прогноз')
        ax.axvline(x=last_date, color='gray', linestyle='--', alpha=0.7, label='Текущий момент')

        # Зона неопределенности РАСТЕТ с каждым днем
        uncertainty = 0.03 * (1 + np.linspace(0, 0.5, len(forecast_prices)))
        upper = [p * (1 + u) for p, u in zip(forecast_prices, uncertainty)]
        lower = [p * (1 - u) for p, u in zip(forecast_prices, uncertainty)]
        ax.fill_between(forecast_dates, lower, upper, alpha=0.2, color='red', label='Зона неопределенности')

        ax.set_title(f'{self.ticker} - Прогноз цены на {forecast_days} дней', fontsize=14, fontweight='bold')
        ax.set_xlabel('Дата')
        ax.set_ylabel('Цена ($)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        direction_symbol = '▲' if pred_direction == 1 else '▼'
        direction_color = 'green' if pred_direction == 1 else 'red'
        ax.annotate(f'{direction_symbol} ${predicted_price:.2f} ({pred_return:.1%})',
                    xy=(forecast_dates[-1] if forecast_dates else last_date,
                        forecast_prices[-1] if forecast_prices else last_price),
                    xytext=(10, 20), textcoords='offset points',
                    fontsize=11, color=direction_color, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        figure.tight_layout()
        canvas.draw()

        return tab

    def create_indicators_tab(self):
        """Создание вкладки с техническими индикаторами"""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        # Создаём фигуру matplotlib
        figure = Figure(figsize=(10, 8), dpi=100)
        canvas = FigureCanvas(figure)
        toolbar = NavigationToolbar(canvas, self)

        layout.addWidget(toolbar)
        layout.addWidget(canvas)

        # Строим графики индикаторов
        hist_data = self.forecast_result['hist_data']
        plot_days = min(100, len(hist_data))
        plot_data = hist_data.tail(plot_days).copy()

        ax1 = figure.add_subplot(311)
        ax2 = figure.add_subplot(312)
        ax3 = figure.add_subplot(313)

        # Цена и скользящие средние
        ax1.plot(plot_data['date'], plot_data['price'], 'b-', linewidth=1.5, label='Цена')
        ma_20 = plot_data['price'].rolling(20).mean()
        ma_50 = plot_data['price'].rolling(50).mean()
        ax1.plot(plot_data['date'], ma_20, 'orange', linewidth=1.5, label='MA 20')
        ax1.plot(plot_data['date'], ma_50, 'purple', linewidth=1.5, label='MA 50')
        ax1.set_title('Цена и скользящие средние', fontsize=10)
        ax1.set_ylabel('Цена ($)')
        ax1.legend(loc='best', fontsize=8)
        ax1.grid(True, alpha=0.3)

        # RSI
        delta = plot_data['price'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-6)
        rsi = 100 - (100 / (1 + rs))

        ax2.plot(plot_data['date'], rsi, 'purple', linewidth=1.5, label='RSI')
        ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Перекупленность')
        ax2.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Перепроданность')
        ax2.set_title('RSI (Индекс относительной силы)', fontsize=10)
        ax2.set_ylabel('RSI')
        ax2.set_ylim(0, 100)
        ax2.legend(loc='best', fontsize=8)
        ax2.grid(True, alpha=0.3)

        # Доходность
        returns = plot_data['price'].pct_change() * 100
        colors = ['red' if x < 0 else 'green' for x in returns]
        ax3.bar(plot_data['date'], returns, color=colors, alpha=0.7, width=0.8)
        ax3.axhline(y=0, color='black', linewidth=1)
        ax3.set_title('Дневная доходность', fontsize=10)
        ax3.set_xlabel('Дата')
        ax3.set_ylabel('Доходность (%)')
        ax3.grid(True, alpha=0.3)

        figure.tight_layout()
        canvas.draw()

        return tab

    def save_forecast(self):
        timeframe = f"{self.forecast_result['forecast_days']} дней"
        self.db.save_forecast(
            self.ticker,
            self.company_name,
            timeframe,
            self.forecast_result['predicted_price'],
            self.forecast_result['expected_return'],
            self.forecast_result['direction']
        )

        msg = QtWidgets.QMessageBox()
        msg.setWindowTitle("Прогноз сохранён")
        msg.setText(f"✅ Прогноз для {self.ticker} успешно сохранён!")
        msg.setInformativeText(
            f"Прогнозируемая цена: ${self.forecast_result['predicted_price']:.2f}\nОжидаемое изменение: {self.forecast_result['expected_return'] * 100:+.2f}%")
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setStyleSheet("""
            QMessageBox {
                background-color: #1a1a2e;
                color: white;
            }
            QMessageBox QLabel {
                color: white;
            }
            QPushButton {
                background-color: #e94560;
                color: white;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        msg.exec_()


class ForecastsListWin(QtWidgets.QWidget):
    def __init__(self, db):
        super().__init__()
        self.db = db
        self.setWindowTitle("Мои прогнозы")
        self.setGeometry(300, 200, 1100, 500)

        self.setStyleSheet("""
            QWidget {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1a1a2e, stop:1 #16213e);
            }
            QLabel {
                color: white;
            }
            QTableWidget {
                background-color: #0f3460;
                color: white;
                border: 2px solid #e94560;
                border-radius: 10px;
                gridline-color: #e94560;
                alternate-background-color: #16213e;
            }
            QTableWidget::item {
                padding: 10px;
            }
            QHeaderView::section {
                background-color: #e94560;
                color: white;
                padding: 10px;
                border: none;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton {
                background-color: #e94560;
                color: white;
                border-radius: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #ff6b6b;
            }
            QPushButton#deleteBtn {
                background-color: #dc3545;
            }
            QPushButton#deleteBtn:hover {
                background-color: #c82333;
            }
        """)

        layout = QtWidgets.QVBoxLayout()

        title = QtWidgets.QLabel("📊 МОИ ПРОГНОЗЫ")
        title.setFont(QtGui.QFont("Arial", 16, QtGui.QFont.Bold))
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet("color: #FF9800; padding: 20px;")
        layout.addWidget(title)

        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels(
            ["№", "Тикер", "Компания", "Горизонт", "Прогноз цена", "Изменение", "Направление", "Дата"])

        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QtWidgets.QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QTableWidget.SelectRows)
        self.table.setAlternatingRowColors(True)

        layout.addWidget(self.table)

        btn_layout = QtWidgets.QHBoxLayout()

        self.delete_btn = QtWidgets.QPushButton("🗑️ УДАЛИТЬ ВЫБРАННЫЙ")
        self.delete_btn.setObjectName("deleteBtn")
        self.delete_btn.setFixedSize(200, 40)
        self.delete_btn.clicked.connect(self.delete_selected)
        btn_layout.addWidget(self.delete_btn)

        self.close_btn = QtWidgets.QPushButton("❌ ЗАКРЫТЬ")
        self.close_btn.setFixedSize(120, 40)
        self.close_btn.clicked.connect(self.close)
        btn_layout.addWidget(self.close_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        self.setLayout(layout)
        self.load_forecasts()

    def load_forecasts(self):
        self.table.setRowCount(0)
        forecasts = self.db.get_all_forecasts()

        if not forecasts:
            self.table.setRowCount(1)
            self.table.setSpan(0, 0, 1, 8)
            no_data_item = QtWidgets.QTableWidgetItem("📭 НЕТ СОХРАНЁННЫХ ПРОГНОЗОВ")
            no_data_item.setTextAlignment(QtCore.Qt.AlignCenter)
            no_data_item.setFlags(QtCore.Qt.NoItemFlags)
            no_data_item.setForeground(QtGui.QColor("#FF9800"))
            self.table.setItem(0, 0, no_data_item)
            return

        self.table.setRowCount(len(forecasts))

        for row, forecast in enumerate(forecasts):
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(row + 1)))

            ticker_item = QtWidgets.QTableWidgetItem(forecast[1])
            ticker_item.setForeground(QtGui.QColor("#00ff88"))
            self.table.setItem(row, 1, ticker_item)

            self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(forecast[2][:40]))
            self.table.setItem(row, 3, QtWidgets.QTableWidgetItem(forecast[3]))

            price_text = f"${forecast[4]:.2f}" if forecast[4] else "—"
            self.table.setItem(row, 4, QtWidgets.QTableWidgetItem(price_text))

            return_text = f"{forecast[5] * 100:+.2f}%" if forecast[5] else "—"
            return_item = QtWidgets.QTableWidgetItem(return_text)
            if forecast[5] and forecast[5] > 0:
                return_item.setForeground(QtGui.QColor("#00ff88"))
            elif forecast[5] and forecast[5] < 0:
                return_item.setForeground(QtGui.QColor("#ff6b6b"))
            self.table.setItem(row, 5, return_item)

            direction_item = QtWidgets.QTableWidgetItem(forecast[6] if forecast[6] else "—")
            if forecast[6] == "Рост":
                direction_item.setForeground(QtGui.QColor("#00ff88"))
            elif forecast[6] == "Падение":
                direction_item.setForeground(QtGui.QColor("#ff6b6b"))
            self.table.setItem(row, 6, direction_item)

            date_str = forecast[7]
            try:
                date_obj = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                formatted_date = date_obj.strftime("%d.%m.%Y %H:%M")
            except:
                formatted_date = date_str
            self.table.setItem(row, 7, QtWidgets.QTableWidgetItem(formatted_date))

        self.table.resizeColumnsToContents()
        self.table.setColumnWidth(0, 50)
        self.table.setColumnWidth(1, 80)
        self.table.setColumnWidth(2, 250)

    def delete_selected(self):
        selected = self.table.currentRow()
        if selected >= 0:
            forecasts = self.db.get_all_forecasts()
            if selected < len(forecasts):
                forecast_id = forecasts[selected][0]
                ticker = forecasts[selected][1]

                reply = QtWidgets.QMessageBox.question(
                    self, "Подтверждение удаления",
                    f"Удалить прогноз для {ticker}?",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
                )

                if reply == QtWidgets.QMessageBox.Yes:
                    self.db.delete_forecast(forecast_id)
                    self.load_forecasts()


class InfoWin(QtWidgets.QWidget):
    def __init__(self, ticker, data, db, prediction_engine):
        super().__init__()
        self.ticker = ticker
        self.data = data
        self.db = db
        self.prediction_engine = prediction_engine
        self.setWindowTitle(f"Компания: {ticker}")
        self.setGeometry(200, 100, 1200, 800)

        self.setStyleSheet("""
            QWidget {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1a1a2e, stop:1 #16213e);
            }
            QLabel {
                color: white;
            }
            QScrollArea {
                background-color: transparent;
                border: none;
            }
            QPushButton {
                background-color: #e94560;
                color: white;
                border-radius: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #ff6b6b;
            }
        """)

        main = QtWidgets.QHBoxLayout()

        left = QtWidgets.QVBoxLayout()

        title = QtWidgets.QLabel(f"📈 {self.data['name']}")
        title.setFont(QtGui.QFont("Arial", 18, QtGui.QFont.Bold))
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet("color: #e94560; padding: 15px;")
        left.addWidget(title)

        tick = QtWidgets.QLabel(f"🏷️ Тикер: {ticker}")
        tick.setFont(QtGui.QFont("Arial", 12, QtGui.QFont.Bold))
        tick.setAlignment(QtCore.Qt.AlignCenter)
        tick.setStyleSheet("color: #00ff88; padding: 5px;")
        left.addWidget(tick)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")

        info_widget = QtWidgets.QWidget()
        info_widget.setStyleSheet("background-color: transparent;")
        info_layout = QtWidgets.QGridLayout(info_widget)

        fields = [
            ("💰 Капитализация", "cap"),
            ("💵 Цена", "price"),
            ("📊 Изменение %", "change"),
            ("📦 Объём", "volume"),
            ("📈 Относит. объём", "rel_vol"),
            ("🏷️ Цена/прибыль", "pe"),
            ("💎 Разводн. приб./акцию", "eps"),
            ("📈 Разводн. приб./акцию, рост", "eps_growth"),
            ("💸 Див. доход %", "div_yield"),
            ("🏭 Сектор", "sector"),
        ]

        for i, (label_text, key) in enumerate(fields):
            lbl = QtWidgets.QLabel(f"{label_text}:")
            lbl.setStyleSheet(
                "color: #e94560; padding: 8px; background-color: rgba(15, 52, 96, 0.8); border-radius: 5px;")

            val = self.data.get(key, "Нет данных")

            if pd.isna(val) or val == "" or val is None:
                val = "Нет данных"

            val_lbl = QtWidgets.QLabel(str(val))

            if key == "change":
                val_str = str(val)
                if val_str != "Нет данных":
                    try:
                        if isinstance(val, str):
                            val_str = val_str.replace('%', '').strip()
                        num = float(val_str)
                        if num < 0:
                            val_lbl.setStyleSheet(
                                "color: #ff6b6b; padding: 8px; font-weight: bold; background-color: rgba(15, 52, 96, 0.8); border-radius: 5px;")
                        else:
                            val_lbl.setStyleSheet(
                                "color: #00ff88; padding: 8px; font-weight: bold; background-color: rgba(15, 52, 96, 0.8); border-radius: 5px;")
                    except:
                        val_lbl.setStyleSheet(
                            "color: white; padding: 8px; background-color: rgba(15, 52, 96, 0.8); border-radius: 5px;")
                else:
                    val_lbl.setStyleSheet(
                        "color: white; padding: 8px; background-color: rgba(15, 52, 96, 0.8); border-radius: 5px;")
            else:
                val_lbl.setStyleSheet(
                    "color: white; padding: 8px; background-color: rgba(15, 52, 96, 0.8); border-radius: 5px;")

            val_lbl.setWordWrap(True)

            info_layout.addWidget(lbl, i, 0)
            info_layout.addWidget(val_lbl, i, 1)

        info_layout.setColumnStretch(0, 1)
        info_layout.setColumnStretch(1, 2)
        info_layout.setSpacing(10)

        scroll.setWidget(info_widget)
        left.addWidget(scroll)

        right = QtWidgets.QVBoxLayout()

        img = QtWidgets.QLabel()
        pix = QtGui.QPixmap("sec_picture.jpg")
        if not pix.isNull():
            scaled = pix.scaled(550, 450, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            img.setPixmap(scaled)
        else:
            img.setText("📊 График NASDAQ")
            img.setAlignment(QtCore.Qt.AlignCenter)
            img.setStyleSheet(
                "color: #e94560; font-size: 20px; border: 2px solid #e94560; border-radius: 15px; padding: 50px;")
        img.setAlignment(QtCore.Qt.AlignCenter)
        right.addWidget(img)

        # Выбор количества дней
        days_layout = QtWidgets.QHBoxLayout()
        days_label = QtWidgets.QLabel("Количество дней для прогноза:")
        days_label.setFont(QtGui.QFont("Arial", 12))
        days_label.setStyleSheet("color: #e94560;")
        self.forecast_days_spin = QtWidgets.QSpinBox()
        self.forecast_days_spin.setRange(1, 30)
        self.forecast_days_spin.setValue(5)
        self.forecast_days_spin.setStyleSheet("""
            QSpinBox {
                background-color: #0f3460;
                color: white;
                border: 2px solid #e94560;
                border-radius: 10px;
                padding: 5px;
                font-size: 14px;
            }
        """)
        days_layout.addWidget(days_label)
        days_layout.addWidget(self.forecast_days_spin)
        days_layout.addStretch()
        right.addLayout(days_layout)

        right.addSpacing(20)

        forecast_btn = QtWidgets.QPushButton("🔮 Получить прогноз")
        forecast_btn.setFixedSize(220, 50)
        forecast_btn.setFont(QtGui.QFont("Arial", 14, QtGui.QFont.Bold))
        forecast_btn.setStyleSheet("background-color: #FF9800;")
        forecast_btn.clicked.connect(self.get_forecast)

        btn_box = QtWidgets.QHBoxLayout()
        btn_box.addStretch()
        btn_box.addWidget(forecast_btn)
        btn_box.addStretch()
        right.addLayout(btn_box)

        close_btn = QtWidgets.QPushButton("❌ Закрыть")
        close_btn.setFixedSize(120, 40)
        close_btn.setStyleSheet("background-color: #2E7D32;")
        close_btn.clicked.connect(self.close)

        btn_box2 = QtWidgets.QHBoxLayout()
        btn_box2.addStretch()
        btn_box2.addWidget(close_btn)
        btn_box2.addStretch()
        right.addLayout(btn_box2)

        main.addLayout(left, 1)
        main.addLayout(right, 1)

        self.setLayout(main)

    def get_forecast(self):
        if self.prediction_engine is None or not self.prediction_engine.is_trained:
            msg = QtWidgets.QMessageBox()
            msg.setWindowTitle("Модель не обучена")
            msg.setText(
                "⚠️ Модель прогнозирования ещё не обучена!\n\nПожалуйста, сначала обучите модель в главном окне (кнопка '🚀 Обучить модель прогноза').")
            msg.setIcon(QtWidgets.QMessageBox.Warning)
            msg.exec_()
            return

        forecast_days = self.forecast_days_spin.value()

        forecast_result = self.prediction_engine.predict(self.ticker, forecast_days)

        if forecast_result is None:
            msg = QtWidgets.QMessageBox()
            msg.setWindowTitle("Ошибка прогноза")
            msg.setText(
                "❌ Не удалось выполнить прогноз для этой компании.\n\nВозможные причины:\n• Недостаточно исторических данных\n• Ошибка при создании признаков\n\nПроверьте консоль для подробной информации.")
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.exec_()
            return

        self.forecast_result_win = ForecastResultWin(self.ticker, self.data['name'], forecast_result, self.db)
        self.forecast_result_win.show()


class MainWin(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Анализ компаний")
        self.setGeometry(300, 200, 700, 550)

        self.db = Database()
        self.prediction_engine = None
        self.df_prices = None
        self.df_fund = None

        self.setStyleSheet("""
            QMainWindow {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1a1a2e, stop:1 #16213e);
            }
            QLabel {
                color: white;
            }
            QLineEdit {
                background-color: #0f3460;
                color: white;
                border: 2px solid #e94560;
                border-radius: 10px;
                padding: 5px;
                font-size: 14px;
            }
            QPushButton {
                background-color: #e94560;
                color: white;
                border-radius: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #ff6b6b;
            }
            QPushButton:disabled {
                background-color: #533483;
            }
            QProgressBar {
                border: 2px solid #e94560;
                border-radius: 5px;
                text-align: center;
                color: white;
            }
            QProgressBar::chunk {
                background-color: #00ff88;
                border-radius: 3px;
            }
        """)

        self.cur_ticker = None
        self.companies = {}

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        layout = QtWidgets.QVBoxLayout(central)

        title_font = QtGui.QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)

        header = QtWidgets.QLabel("📊 Аналитическая платформа")
        header.setFont(title_font)
        header.setAlignment(QtCore.Qt.AlignCenter)
        header.setStyleSheet("color: #e94560; padding: 20px;")
        layout.addWidget(header)

        img = QtWidgets.QLabel()
        pix = QtGui.QPixmap("first_picture.webp")
        if not pix.isNull():
            scaled = pix.scaled(650, 250, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            img.setPixmap(scaled)
        else:
            img.setText("📈 NASDAQ Chart")
            img.setAlignment(QtCore.Qt.AlignCenter)
            img.setStyleSheet(
                "color: #e94560; font-size: 20px; border: 2px solid #e94560; border-radius: 15px; padding: 40px;")
        img.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(img)

        search_label = QtWidgets.QLabel("🔍 Поиск компании по тикеру:")
        search_label.setStyleSheet("color: #e94560; font-size: 14px; font-weight: bold; padding: 10px;")
        layout.addWidget(search_label)

        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Введите тикер: AAPL, MSFT, NVDA...")
        layout.addWidget(self.search)

        # Загружаем данные для автопоиска
        tickers, self.companies = self.load_data("Проект_10М.xlsx")

        # Настраиваем completer
        completer = QtWidgets.QCompleter(list(set(tickers)))
        completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
        completer.setFilterMode(QtCore.Qt.MatchStartsWith)
        completer.setCompletionMode(QtWidgets.QCompleter.PopupCompletion)
        completer.setMaxVisibleItems(10)

        self.search.setCompleter(completer)
        completer.activated.connect(self.on_select)

        btn_layout = QtWidgets.QHBoxLayout()

        self.select_btn = QtWidgets.QPushButton("✅ Выбрать компанию")
        self.select_btn.setFixedSize(200, 45)
        self.select_btn.setFont(QtGui.QFont("Arial", 12, QtGui.QFont.Bold))
        self.select_btn.clicked.connect(self.open_info)
        self.select_btn.setEnabled(False)
        btn_layout.addWidget(self.select_btn)

        self.train_btn = QtWidgets.QPushButton("🚀 Обучить модель прогноза")
        self.train_btn.setFixedSize(200, 45)
        self.train_btn.setFont(QtGui.QFont("Arial", 12, QtGui.QFont.Bold))
        self.train_btn.setStyleSheet("background-color: #17a2b8;")
        self.train_btn.clicked.connect(self.train_model)
        self.train_btn.setEnabled(False)
        btn_layout.addWidget(self.train_btn)

        self.forecasts_btn = QtWidgets.QPushButton("📋 Мои прогнозы")
        self.forecasts_btn.setFixedSize(200, 45)
        self.forecasts_btn.setFont(QtGui.QFont("Arial", 12, QtGui.QFont.Bold))
        self.forecasts_btn.setStyleSheet("background-color: #28a745;")
        self.forecasts_btn.clicked.connect(self.open_forecasts_list)
        btn_layout.addWidget(self.forecasts_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Прогресс бар
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        layout.addStretch()

        self.result = QtWidgets.QLabel("")
        self.result.setAlignment(QtCore.Qt.AlignCenter)
        self.result.setStyleSheet("color: #00ff88; font-size: 14px; font-weight: bold; padding: 10px;")
        layout.addWidget(self.result)

        # Загружаем ценовые данные
        self.load_price_data()

    def load_data(self, fname):
        try:
            df = pd.read_excel(fname)
            data = {}
            tickers = []

            for _, row in df.iterrows():
                name = row.get('Инструмент', '')
                if pd.isna(name):
                    continue

                name_str = str(name)

                # Извлекаем тикер (первое слово до пробела)
                parts = name_str.split(maxsplit=1)
                if len(parts) >= 1:
                    tick = parts[0].strip().upper()
                    tickers.append(tick)

                    data[tick] = {
                        'name': name_str,
                        'cap': clean_numeric_string(row.get('Капитализация', 'Нет данных')),
                        'price': clean_numeric_string(row.get('Цена', 'Нет данных')),
                        'change': row.get('Изменение %', 'Нет данных'),
                        'volume': row.get('Объём', 'Нет данных'),
                        'rel_vol': row.get('Относит. объём', 'Нет данных'),
                        'pe': clean_numeric_string(row.get('Цена/прибыль', 'Нет данных')),
                        'eps': clean_numeric_string(row.get('Разводн. приб./акцию', 'Нет данных')),
                        'eps_growth': row.get('Разводн. приб./акцию, рост', 'Нет данных'),
                        'div_yield': row.get('Див. доход %', 'Нет данных'),
                        'sector': row.get('Сектор', 'Нет данных'),
                    }

            # Убираем дубликаты тикеров
            unique_tickers = []
            seen = set()
            for t in tickers:
                if t not in seen:
                    seen.add(t)
                    unique_tickers.append(t)

            return unique_tickers, data

        except Exception as e:
            print(f"Ошибка загрузки: {e}")
            traceback.print_exc()
            default = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN"]
            return default, {t: {'name': t} for t in default}

    def load_price_data(self):
        """Загрузка ценовых данных из файла"""
        try:
            self.df_prices = pd.read_excel("stock_prices_real.xlsx")
            self.df_prices['date'] = pd.to_datetime(self.df_prices['date'])
            if self.df_prices['price'].dtype == 'object':
                self.df_prices['price'] = self.df_prices['price'].astype(str).str.replace(',', '.').astype(float)
            self.df_prices = self.df_prices.drop_duplicates(subset=['ticker', 'date'])
            self.df_prices = self.df_prices.sort_values(['ticker', 'date'])

            print(f"Загружено {len(self.df_prices)} записей о ценах")
            print(f"Доступные тикеры: {sorted(self.df_prices['ticker'].unique())[:10]}...")
            self.result.setText(f"✅ Данные о ценах загружены. Теперь можно обучить модель.")
            self.train_btn.setEnabled(True)
            return True

        except Exception as e:
            print(f"Ошибка загрузки цен: {e}")
            traceback.print_exc()
            self.result.setText(f"⚠️ Не удалось загрузить файл с ценами. Прогноз недоступен.")
            return False

    def train_model(self):
        if self.df_prices is None:
            self.show_error_message("Нет данных о ценах для обучения модели!")
            return

        self.train_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.result.setText("🔄 Обучение модели прогнозирования...")
        QtWidgets.QApplication.processEvents()

        try:
            # Загружаем фундаментальные данные для движка
            self.df_fund = pd.read_excel("Проект_10М.xlsx")
            self.df_fund['ticker'] = self.df_fund['Инструмент'].str.split().str[0].str.upper()

            self.prediction_engine = StockPredictionEngine(self.df_prices, self.df_fund)

            self.progress_bar.setValue(50)
            QtWidgets.QApplication.processEvents()

            forecast_days = 5
            metrics = self.prediction_engine.train(forecast_days)

            self.progress_bar.setValue(100)

            if metrics:
                self.result.setText(f"✅ Модель обучена! R² = {metrics['r2']:.4f}, Accuracy = {metrics['accuracy']:.4f}")
                msg = QtWidgets.QMessageBox()
                msg.setWindowTitle("Модель обучена")
                msg.setText(
                    f"✅ Модель прогнозирования успешно обучена!\n\n📊 R² (доходность): {metrics['r2']:.4f}\n🎯 Accuracy (направление): {metrics['accuracy']:.4f}\n📈 Размер выборки: {metrics['samples']} записей")
                msg.setIcon(QtWidgets.QMessageBox.Information)
                msg.exec_()
                print("\n" + "=" * 60)
                print("МОДЕЛЬ УСПЕШНО ОБУЧЕНА!")
                print("=" * 60 + "\n")
            else:
                self.result.setText("❌ Недостаточно данных для обучения модели")
                msg = QtWidgets.QMessageBox()
                msg.setWindowTitle("Ошибка")
                msg.setText(
                    "❌ Недостаточно данных для обучения модели!\n\nПроверьте, что файл stock_prices_real.xlsx содержит достаточно данных.")
                msg.setIcon(QtWidgets.QMessageBox.Critical)
                msg.exec_()

        except Exception as e:
            self.result.setText(f"❌ Ошибка обучения: {str(e)}")
            msg = QtWidgets.QMessageBox()
            msg.setWindowTitle("Ошибка")
            msg.setText(f"❌ Ошибка при обучении модели:\n{str(e)}")
            msg.exec_()
            traceback.print_exc()
        finally:
            self.train_btn.setEnabled(True)
            self.progress_bar.setVisible(False)

    def on_select(self, text):
        self.cur_ticker = text.upper().strip()
        company_name = self.companies.get(self.cur_ticker, {}).get('name', self.cur_ticker)
        self.result.setText(f"✅ Выбран: {self.cur_ticker} - {company_name[:50]}")
        self.search.setText(self.cur_ticker)
        self.select_btn.setEnabled(True)

    def open_info(self):
        current_text = self.search.text().strip().upper()

        if not current_text:
            self.show_error_message("Введите тикер компании в поле поиска")
            return

        if current_text not in self.companies:
            self.show_error_message(
                f"Компания '{current_text}' не найдена. Попробуйте AAPL, MSFT, NVDA или выберите из списка")
            return

        self.info_win = InfoWin(current_text, self.companies[current_text], self.db, self.prediction_engine)
        self.info_win.show()

    def show_error_message(self, message):
        error_box = QtWidgets.QMessageBox()
        error_box.setWindowTitle("Ошибка")
        error_box.setIcon(QtWidgets.QMessageBox.Warning)
        error_box.setText(message)
        error_box.setStyleSheet("""
            QMessageBox {
                background-color: #1a1a2e;
                color: white;
            }
            QMessageBox QLabel {
                color: white;
                min-width: 350px;
            }
            QPushButton {
                background-color: #e94560;
                color: white;
                border-radius: 5px;
                padding: 5px 15px;
            }
        """)
        error_box.exec_()

    def open_forecasts_list(self):
        self.forecasts_win = ForecastsListWin(self.db)
        self.forecasts_win.show()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = MainWin()
    win.show()
    sys.exit(app.exec_())