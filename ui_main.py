import sys
import pandas as pd
import numpy as np
import os
import traceback
from PyQt5 import QtCore, QtGui, QtWidgets

from database import Database
from prediction_engine import StockPredictionEngine
from gui_widgets import InfoWin, ForecastsListWin


class MainWin(QtWidgets.QMainWindow):
    # Главное окно программы

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Анализ компаний - 6+8 ML + ARIMA/SARIMA")
        self.setGeometry(200, 100, 1400, 900)

        # Инициализация компонентов
        self.db = Database()
        self.prediction_engine = None
        self.df_prices = None
        self.df_fund = None
        self.forecasts_win = None
        self.cur_ticker = None
        self.companies = {}
        self.all_tickers = []

        # Создаём центральный виджет
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(30, 30, 30, 30)

        # Заголовок
        header = QtWidgets.QLabel("📊 ПРОГНОЗИРОВАНИЕ АКЦИЙ")
        header.setFont(QtGui.QFont("Segoe UI", 24, QtGui.QFont.Bold))
        header.setAlignment(QtCore.Qt.AlignCenter)
        header.setStyleSheet(
            "color: #fab387; padding: 20px; background: #313244; border-radius: 20px; margin-bottom: 20px;")
        layout.addWidget(header)

        # Подзаголовок
        subheader = QtWidgets.QLabel(
            "Классификация (6 моделей) | Регрессия (8 моделей) | ARIMA | SARIMA")
        subheader.setFont(QtGui.QFont("Segoe UI", 12))
        subheader.setAlignment(QtCore.Qt.AlignCenter)
        subheader.setStyleSheet(
            "color: #89b4fa; padding: 12px; background: #313244; border-radius: 15px; margin-bottom: 20px;")
        layout.addWidget(subheader)

        # Картинка
        img = QtWidgets.QLabel()
        pix = QtGui.QPixmap("first_picture.webp")
        if not pix.isNull():
            img.setPixmap(pix.scaled(800, 200, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        else:
            img.setText("📈 6 ML ДЛЯ НАПРАВЛЕНИЯ + 8 ML ДЛЯ ЦЕНЫ + ARIMA/SARIMA")
            img.setAlignment(QtCore.Qt.AlignCenter)
            img.setStyleSheet(
                "color: #a6e3a1; font-size: 14px; border: 2px solid #fab387; border-radius: 20px; padding: 30px; background: #313244;")
        layout.addWidget(img)

        # Поиск
        search_label = QtWidgets.QLabel("🔍 ПОИСК КОМПАНИИ ПО ТИКЕРУ:")
        search_label.setFont(QtGui.QFont("Segoe UI", 14, QtGui.QFont.Bold))
        search_label.setStyleSheet("color: #fab387; padding: 10px 0;")
        layout.addWidget(search_label)

        # Поле ввода
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Введите тикер: AAPL, MSFT, NVDA, GOOGL, AMZN...")
        self.search.setMinimumHeight(50)
        self.search.setStyleSheet(
            "background-color: #f0f0f0; color: #1e1e2e; border: 2px solid #89b4fa; border-radius: 12px; padding: 12px; font-size: 14px;")
        layout.addWidget(self.search)

        # Статус
        self.status_label = QtWidgets.QLabel("")
        self.status_label.setAlignment(QtCore.Qt.AlignCenter)
        self.status_label.setStyleSheet(
            "color: #a6e3a1; font-size: 13px; font-weight: bold; padding: 15px; background: #313244; border-radius: 12px; margin-top: 20px;")
        layout.addWidget(self.status_label)

        # Кнопки
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.setSpacing(25)

        self.select_btn = QtWidgets.QPushButton("✅ ВЫБРАТЬ КОМПАНИЮ")
        self.select_btn.setFixedSize(350, 65)
        self.select_btn.setFont(QtGui.QFont("Segoe UI", 14, QtGui.QFont.Bold))
        self.select_btn.setStyleSheet("background-color: #89b4fa; color: #1e1e2e; border-radius: 12px;")
        self.select_btn.clicked.connect(self.open_info)
        self.select_btn.setEnabled(False)
        btn_layout.addWidget(self.select_btn)

        self.forecasts_btn = QtWidgets.QPushButton("📋 МОИ ПРОГНОЗЫ")
        self.forecasts_btn.setFixedSize(350, 65)
        self.forecasts_btn.setFont(QtGui.QFont("Segoe UI", 14, QtGui.QFont.Bold))
        self.forecasts_btn.setStyleSheet("background-color: #a6e3a1; color: #1e1e2e; border-radius: 12px;")
        self.forecasts_btn.clicked.connect(self.open_forecasts_list)
        btn_layout.addWidget(self.forecasts_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        layout.addStretch()

        # Загружаем данные
        self.load_all_data()

    def load_all_data(self):
        # Загружает данные о компаниях и ценах

        try:
            # Проверяем наличие файла с компаниями
            if os.path.exists("Проект_10М.xlsx"):
                # Читаем Excel без заголовков
                df = pd.read_excel("Проект_10М.xlsx", header=None)

                tickers = []
                data = {}

                # Перебираем все строки
                for idx, row in df.iterrows():
                    ticker = str(row[0]).strip().upper() if pd.notna(row[0]) else ""

                    # Пропускаем пустые строки
                    if ticker == "" or ticker == 'NAN' or ticker == 'NONE':
                        continue

                    tickers.append(ticker)

                    # Сохраняем данные компании
                    data[ticker] = {
                        'name': ticker,
                        'cap': row[1] if len(row) > 1 and pd.notna(row[1]) else "Нет данных",
                        'price': row[2] if len(row) > 2 and pd.notna(row[2]) else "Нет данных",
                        'change': row[3] if len(row) > 3 and pd.notna(row[3]) else "Нет данных",
                        'volume': row[4] if len(row) > 4 and pd.notna(row[4]) else "Нет данных",
                        'rel_vol': row[5] if len(row) > 5 and pd.notna(row[5]) else "Нет данных",
                        'pe': row[6] if len(row) > 6 and pd.notna(row[6]) else "Нет данных",
                        'eps': row[7] if len(row) > 7 and pd.notna(row[7]) else "Нет данных",
                        'eps_growth': row[8] if len(row) > 8 and pd.notna(row[8]) else "Нет данных",
                        'div_yield': row[9] if len(row) > 9 and pd.notna(row[9]) else "Нет данных",
                        'sector': row[10] if len(row) > 10 and pd.notna(row[10]) else "Нет данных",
                    }

                self.companies = data
                self.all_tickers = list(set(tickers))
                self.all_tickers.sort()
            else:
                # Если файл не найден, используем тестовые данные
                self.all_tickers = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN", "META", "TSLA", "ORLY"]
                self.companies = {}
                for t in self.all_tickers:
                    self.companies[t] = {'name': t, 'sector': 'Тест'}
                self.status_label.setText("⚠️ Файл 'Проект_10М.xlsx' не найден. Использую тестовые данные.")

            # Настраиваем автопоиск
            if self.all_tickers:
                self.completer = QtWidgets.QCompleter(self.all_tickers)
                self.completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
                self.completer.setFilterMode(QtCore.Qt.MatchStartsWith)
                self.completer.setCompletionMode(QtWidgets.QCompleter.PopupCompletion)
                self.search.setCompleter(self.completer)
                self.completer.activated.connect(self.on_select)
                self.search.textChanged.connect(self.on_text_changed)

                self.status_label.setText(f"✅ ДАННЫЕ ЗАГРУЖЕНЫ. ДОСТУПНО {len(self.all_tickers)} ТИКЕРОВ")

            # Загружаем данные о ценах и фундаментальные данные
            self.load_price_data()
            self.load_fundamental_data()

        except Exception as e:
            self.status_label.setText(f"⚠️ ОШИБКА ЗАГРУЗКИ: {str(e)[:50]}")

            # Используем тестовые данные при ошибке
            self.all_tickers = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN", "ORLY"]
            self.companies = {}
            for t in self.all_tickers:
                self.companies[t] = {'name': t, 'sector': 'Тест'}

            self.completer = QtWidgets.QCompleter(self.all_tickers)
            self.completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
            self.completer.setFilterMode(QtCore.Qt.MatchStartsWith)
            self.completer.setCompletionMode(QtWidgets.QCompleter.PopupCompletion)
            self.search.setCompleter(self.completer)
            self.completer.activated.connect(self.on_select)
            self.search.textChanged.connect(self.on_text_changed)

            self.status_label.setText(f"⚠️ ИСПОЛЬЗУЮТСЯ ТЕСТОВЫЕ ДАННЫЕ. ДОСТУПНО {len(self.all_tickers)} ТИКЕРОВ")
            self.load_price_data()
            self.load_fundamental_data()

    def on_text_changed(self, text):
        # Обновляет состояние кнопки при вводе текста

        if text.strip():
            ticker_upper = text.strip().upper()
            if ticker_upper in self.companies:
                self.select_btn.setEnabled(True)
            else:
                self.select_btn.setEnabled(False)
        else:
            self.select_btn.setEnabled(False)

    def on_select(self, text):
        # Обрабатывает выбор из выпадающего списка

        self.cur_ticker = text.upper().strip()
        self.status_label.setText(f"✅ ВЫБРАН: {self.cur_ticker}")
        self.search.setText(self.cur_ticker)
        self.select_btn.setEnabled(True)

    def load_price_data(self):
        # Загружает данные о ценах акций

        try:
            if os.path.exists("stock_prices_real.xlsx"):
                self.df_prices = pd.read_excel("stock_prices_real.xlsx")
                self.df_prices['date'] = pd.to_datetime(self.df_prices['date'])

                # Преобразуем цены в числа, если они в виде строк
                if self.df_prices['price'].dtype == 'object':
                    self.df_prices['price'] = self.df_prices['price'].astype(str).str.replace(',', '.').astype(float)

                # Удаляем дубликаты и сортируем
                self.df_prices = self.df_prices.drop_duplicates(['ticker', 'date']).sort_values(['ticker', 'date'])
        except:
            pass

    def load_fundamental_data(self):
        # Загружает фундаментальные данные

        try:
            if os.path.exists("Проект_10М.xlsx"):
                self.df_fund = pd.read_excel("Проект_10М.xlsx", header=None)
                self.df_fund.columns = ['ticker', 'cap', 'price', 'change', 'volume', 'rel_vol', 'pe', 'eps',
                                        'eps_growth', 'div_yield', 'sector']
                self.df_fund['ticker'] = self.df_fund['ticker'].astype(str).str.upper().str.strip()
        except:
            self.df_fund = None

    def open_info(self):
        # Открывает окно информации о компании

        ticker = self.search.text().strip().upper()

        # Проверяем, что поле не пустое
        if not ticker:
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Пожалуйста, введите тикер компании")
            return

        # Проверяем, что тикер существует
        if ticker not in self.companies:
            available = []
            for t in list(self.companies.keys())[:20]:
                available.append(t)
            available_str = ', '.join(available)
            QtWidgets.QMessageBox.warning(
                self, "Ошибка",
                f"Тикер '{ticker}' не найден.\nДоступные тикеры: {available_str}..."
            )
            return

        # Проверяем, что есть данные о ценах
        if self.df_prices is None:
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Нет данных о ценах")
            return

        # Инициализируем движок прогнозирования
        if self.prediction_engine is None:
            self.status_label.setText("🔄 ИНИЦИАЛИЗАЦИЯ ДВИЖКА...")
            QtWidgets.QApplication.processEvents()
            self.prediction_engine = StockPredictionEngine(self.df_prices, self.df_fund)

        # Открываем окно
        self.info_win = InfoWin(ticker, self.companies[ticker], self.db, self.prediction_engine)
        self.info_win.show()

    def open_forecasts_list(self):
        # Открывает окно со списком прогнозов

        if self.forecasts_win is None or not self.forecasts_win.isVisible():
            self.forecasts_win = ForecastsListWin(self.db)
            self.forecasts_win.show()
        else:
            self.forecasts_win.raise_()
            self.forecasts_win.activateWindow()