import sys
import pandas as pd
import numpy as np
from datetime import datetime
from PyQt5 import QtCore, QtGui, QtWidgets
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from database import Database
from statistical_validator import StatisticalValidator


class StatisticalSignificanceWidget(QtWidgets.QWidget):
    # Виджет для отображения статистической значимости

    def __init__(self, validator, best_model_name):
        super().__init__()
        self.validator = validator
        self.best_model_name = best_model_name
        self.statistics = validator.get_statistics()
        self.initUI()

    def initUI(self):
        # Настраиваем внешний вид
        self.setStyleSheet("background-color: #1e1e2e;")
        layout = QtWidgets.QVBoxLayout(self)

        # Заголовок
        self.title_label = QtWidgets.QLabel(
            f"СТАТИСТИЧЕСКАЯ ЗНАЧИМОСТЬ (Bootstrap, 1000 итераций, α=0.05)\nЛучшая модель: {self.best_model_name}")
        self.title_label.setFont(QtGui.QFont("Segoe UI", 14, QtGui.QFont.Bold))
        self.title_label.setAlignment(QtCore.Qt.AlignCenter)
        self.title_label.setStyleSheet("color: #ffffff; padding: 15px;")
        layout.addWidget(self.title_label)

        # Таблица с результатами
        self.table_widget = QtWidgets.QTableWidget()
        self.table_widget.setColumnCount(5)
        self.table_widget.setHorizontalHeaderLabels(["Модель", "ΔROC-AUC", "95% CI", "p-value", "Значимо"])
        self.table_widget.setFont(QtGui.QFont("Segoe UI", 11))
        self.table_widget.setColumnWidth(0, 140)
        self.table_widget.setColumnWidth(1, 100)
        self.table_widget.setColumnWidth(2, 180)
        self.table_widget.setColumnWidth(3, 100)
        self.table_widget.setColumnWidth(4, 90)
        self.table_widget.setMinimumHeight(250)

        self.table_widget.setStyleSheet("""
            QTableWidget {
                background-color: #ffffff;
                color: #1e1e2e;
                border: 2px solid #89b4fa;
                border-radius: 10px;
                gridline-color: #c0c0c0;
            }
            QTableWidget::item {
                padding: 10px;
                color: #1e1e2e;
            }
            QTableWidget::item:selected {
                background-color: #89b4fa;
                color: #ffffff;
            }
            QHeaderView::section {
                background-color: #89b4fa;
                color: #ffffff;
                padding: 10px;
                font-weight: bold;
                border: none;
            }
        """)
        self.table_widget.setAlternatingRowColors(True)

        # Сокращённые названия моделей
        short_names = {
            'Logistic Regression': 'Logistic',
            'Random Forest (Clf)': 'RF',
            'XGBoost (Clf)': 'XGB',
            'LightGBM (Clf)': 'LGBM',
            'CatBoost (Clf)': 'CatB',
            'MLP (Clf)': 'MLP'
        }

        # Заполняем таблицу
        self.table_widget.setRowCount(len(self.statistics))

        for row, stat in enumerate(self.statistics):
            # Название модели
            model_short = short_names.get(stat['model'], stat['model'])
            item_model = QtWidgets.QTableWidgetItem(model_short)
            item_model.setForeground(QtGui.QColor("#1e1e2e"))
            item_model.setFont(QtGui.QFont("Segoe UI", 11, QtGui.QFont.Bold))
            self.table_widget.setItem(row, 0, item_model)

            # Разница ROC-AUC
            if stat['roc_auc_diff'] > 0:
                diff_str = f"+{stat['roc_auc_diff']:.4f}"
            else:
                diff_str = f"{stat['roc_auc_diff']:.4f}"
            item_diff = QtWidgets.QTableWidgetItem(diff_str)
            if stat['roc_auc_diff'] > 0:
                item_diff.setForeground(QtGui.QColor("#00aa00"))
            else:
                item_diff.setForeground(QtGui.QColor("#cc0000"))
            self.table_widget.setItem(row, 1, item_diff)

            # Доверительный интервал
            ci_str = f"[{stat['ci_lower']:.4f}, {stat['ci_upper']:.4f}]"
            item_ci = QtWidgets.QTableWidgetItem(ci_str)
            item_ci.setForeground(QtGui.QColor("#1e1e2e"))
            self.table_widget.setItem(row, 2, item_ci)

            # p-value
            if stat['p_value'] < 0.001:
                pval_str = "< 0.001"
            else:
                pval_str = f"{stat['p_value']:.4f}"
            item_pval = QtWidgets.QTableWidgetItem(pval_str)
            if stat['p_value'] < 0.05:
                item_pval.setForeground(QtGui.QColor("#00aa00"))
            else:
                item_pval.setForeground(QtGui.QColor("#cc0000"))
            item_pval.setFont(QtGui.QFont("Segoe UI", 11, QtGui.QFont.Bold))
            self.table_widget.setItem(row, 3, item_pval)

            # Значимость
            sign_str = "✅ ДА" if stat['is_significant'] else "❌ НЕТ"
            item_sign = QtWidgets.QTableWidgetItem(sign_str)
            if stat['is_significant']:
                item_sign.setForeground(QtGui.QColor("#00aa00"))
            else:
                item_sign.setForeground(QtGui.QColor("#cc0000"))
            item_sign.setFont(QtGui.QFont("Segoe UI", 11, QtGui.QFont.Bold))
            self.table_widget.setItem(row, 4, item_sign)

        layout.addWidget(self.table_widget)

        # График
        fig = self.create_statistical_plot()
        canvas = FigureCanvas(fig)
        canvas.setMinimumHeight(400)
        toolbar = NavigationToolbar(canvas, self)
        toolbar.setStyleSheet("background-color: #313244; color: #cdd6f4; border-radius: 10px;")
        layout.addWidget(toolbar)
        layout.addWidget(canvas)

    def create_statistical_plot(self):
        # Создаёт график для визуализации статистической значимости

        fig = plt.figure(figsize=(15, 5), facecolor='#1e1e2e')

        short_names = {
            'Logistic Regression': 'Logistic',
            'Random Forest (Clf)': 'RF',
            'XGBoost (Clf)': 'XGB',
            'LightGBM (Clf)': 'LGBM',
            'CatBoost (Clf)': 'CatB',
            'MLP (Clf)': 'MLP'
        }

        # Список моделей для отображения
        models = []
        for stat in self.statistics:
            model_short = short_names.get(stat['model'], stat['model'])
            models.append(model_short)

        # График 1: p-value
        ax1 = fig.add_subplot(131)
        ax1.set_facecolor('#313244')
        pvalues = [stat['p_value'] for stat in self.statistics]

        # Цвета: зелёный если p < 0.05, иначе красный
        colors_p = []
        for p in pvalues:
            if p < 0.05:
                colors_p.append('#a6e3a1')
            else:
                colors_p.append('#f38ba8')

        ax1.barh(models, pvalues, color=colors_p, edgecolor='white', linewidth=0.5, height=0.6)
        ax1.axvline(x=0.05, color='#fab387', linestyle='--', alpha=0.8, linewidth=2, label='α = 0.05')
        ax1.set_xlabel('p-value', fontsize=11, color='#cdd6f4')
        ax1.set_title('p-value', fontsize=12, fontweight='bold', color='#ffffff')
        ax1.legend(loc='lower right', facecolor='#313244', edgecolor='#fab387')
        ax1.tick_params(colors='#cdd6f4', labelsize=10)
        ax1.grid(True, alpha=0.15, axis='x', color='#cdd6f4', linestyle='--')

        # График 2: доверительные интервалы
        ax2 = fig.add_subplot(132)
        ax2.set_facecolor('#313244')
        for i, stat in enumerate(self.statistics):
            ci_lower = stat['ci_lower']
            ci_upper = stat['ci_upper']
            if stat['is_significant']:
                color = '#a6e3a1'
            else:
                color = '#f38ba8'
            ax2.plot([ci_lower, ci_upper], [i, i], color=color, linewidth=3, alpha=0.8)
            ax2.plot(ci_lower, i, 'v', color=color, markersize=8, alpha=0.8)
            ax2.plot(ci_upper, i, '^', color=color, markersize=8, alpha=0.8)
            ax2.axvline(x=0, color='#fab387', linestyle='--', alpha=0.5, linewidth=1)
        ax2.set_yticks(range(len(self.statistics)))
        ax2.set_yticklabels(models, color='#cdd6f4', fontsize=10)
        ax2.set_xlabel('95% CI разницы ROC-AUC', fontsize=11, color='#cdd6f4')
        ax2.set_title('95% доверительный интервал', fontsize=12, fontweight='bold', color='#ffffff')
        ax2.tick_params(colors='#cdd6f4', labelsize=10)
        ax2.grid(True, alpha=0.15, axis='x', color='#cdd6f4', linestyle='--')

        # График 3: разница ROC-AUC
        ax3 = fig.add_subplot(133)
        ax3.set_facecolor('#313244')
        roc_diffs = [stat['roc_auc_diff'] for stat in self.statistics]
        colors_d = []
        for d in roc_diffs:
            if d > 0:
                colors_d.append('#a6e3a1')
            else:
                colors_d.append('#f38ba8')
        ax3.barh(models, roc_diffs, color=colors_d, edgecolor='white', linewidth=0.5, height=0.6)
        ax3.axvline(x=0, color='#fab387', linestyle='-', alpha=0.8, linewidth=1)
        ax3.set_xlabel('ΔROC-AUC', fontsize=11, color='#cdd6f4')
        ax3.set_title('Разница ROC-AUC', fontsize=12, fontweight='bold', color='#ffffff')
        ax3.tick_params(colors='#cdd6f4', labelsize=10)
        ax3.grid(True, alpha=0.15, axis='x', color='#cdd6f4', linestyle='--')

        plt.tight_layout()
        return fig


class ForecastResultWin(QtWidgets.QWidget):
    # Окно с результатами прогноза

    def __init__(self, ticker, company_name, forecast_result, db):
        super().__init__()
        self.ticker = ticker
        self.company_name = company_name
        self.forecast_result = forecast_result
        self.db = db
        self.setWindowTitle(f"Результат прогноза - {ticker}")
        self.setGeometry(100, 50, 1600, 1000)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)
        self.initUI()

    def initUI(self):
        # Основной layout
        self.setStyleSheet("background-color: #1e1e2e;")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)

        # Заголовок
        self.title = QtWidgets.QLabel(f"📊 ПРОГНОЗ ДЛЯ {self.ticker}")
        self.title.setFont(QtGui.QFont("Segoe UI", 20, QtGui.QFont.Bold))
        self.title.setAlignment(QtCore.Qt.AlignCenter)
        self.title.setStyleSheet("color: #fab387; padding: 15px;")
        layout.addWidget(self.title)

        # Название компании
        self.company = QtWidgets.QLabel(self.company_name[:70])
        self.company.setAlignment(QtCore.Qt.AlignCenter)
        self.company.setStyleSheet("color: #a6e3a1; font-size: 13px; padding: 5px;")
        layout.addWidget(self.company)

        # Вкладки
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setMinimumHeight(800)
        self.tabs.setStyleSheet(
            "QTabWidget::pane { background-color: #2a2a3e; border-radius: 15px; border: 1px solid #fab387; }"
            "QTabBar::tab { background-color: #313244; color: #cdd6f4; padding: 12px 24px; margin: 5px 3px; border-radius: 12px; font-weight: bold; }"
            "QTabBar::tab:selected { background-color: #fab387; color: #1e1e2e; }")

        # Добавляем вкладки
        self.tabs.addTab(self.create_result_tab(), "📊 РЕЗУЛЬТАТЫ")
        self.tabs.addTab(self.create_price_chart_tab(), "📈 ГРАФИК ЦЕНЫ")
        self.tabs.addTab(self.create_indicators_tab(), "📉 ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ")

        # Добавляем вкладки с моделями, если есть данные
        if 'classification_results' in self.forecast_result:
            if self.forecast_result['classification_results'] is not None:
                self.tabs.addTab(self.create_classification_tab(), "🔬 СРАВНЕНИЕ 6 ML КЛАССИФИКАТОРОВ")

                if 'regression_results' in self.forecast_result:
                    if self.forecast_result['regression_results'] is not None:
                        self.tabs.addTab(self.create_regression_tab(), "📊 СРАВНЕНИЕ 8 ML РЕГРЕССОРОВ")

                if 'time_series_results' in self.forecast_result:
                    if self.forecast_result['time_series_results'] is not None:
                        self.tabs.addTab(self.create_time_series_tab(), "📈 ARIMA/SARIMA")

                # Статистическая значимость
                if 'y_test' in self.forecast_result:
                    if self.forecast_result['y_test'] is not None:
                        if 'y_proba_dict' in self.forecast_result:
                            if self.forecast_result['y_proba_dict'] is not None:
                                validator = StatisticalValidator(
                                    self.forecast_result['classification_results'],
                                    self.forecast_result['y_test'],
                                    self.forecast_result['y_proba_dict']
                                )
                                validator.calculate_all_tests(self.forecast_result['best_classifier'])
                                self.stat_widget = StatisticalSignificanceWidget(validator, self.forecast_result[
                                    'best_classifier'])
                                self.tabs.addTab(self.stat_widget, "📊 СТАТИСТИЧЕСКАЯ ЗНАЧИМОСТЬ")

        layout.addWidget(self.tabs)

        # Кнопки
        btn_layout = QtWidgets.QHBoxLayout()

        self.save_btn = QtWidgets.QPushButton("💾 СОХРАНИТЬ ПРОГНОЗ")
        self.save_btn.setFixedSize(200, 45)
        self.save_btn.setFont(QtGui.QFont("Segoe UI", 12, QtGui.QFont.Bold))
        self.save_btn.setStyleSheet("background-color: #89b4fa; color: #1e1e2e; border-radius: 12px;")
        self.save_btn.clicked.connect(self.save_forecast)
        btn_layout.addWidget(self.save_btn)

        self.close_btn = QtWidgets.QPushButton("❌ ЗАКРЫТЬ")
        self.close_btn.setFixedSize(140, 45)
        self.close_btn.setFont(QtGui.QFont("Segoe UI", 12, QtGui.QFont.Bold))
        self.close_btn.setStyleSheet("background-color: #f38ba8; color: #1e1e2e; border-radius: 12px;")
        self.close_btn.clicked.connect(self.close)
        btn_layout.addWidget(self.close_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    def create_result_tab(self):
        # Вкладка с основными результатами прогноза

        tab = QtWidgets.QWidget()
        tab.setStyleSheet("background-color: #2a2a3e;")
        layout = QtWidgets.QVBoxLayout(tab)

        # Карточка с результатами
        card = QtWidgets.QFrame()
        card.setStyleSheet("background-color: #313244; border-radius: 20px; border: 1px solid #fab387;")
        card_layout = QtWidgets.QVBoxLayout(card)
        card_layout.setSpacing(15)

        # Текущая цена
        self.current = QtWidgets.QLabel(f"💰 ТЕКУЩАЯ ЦЕНА: ${self.forecast_result['current_price']:.2f}")
        self.current.setFont(QtGui.QFont("Segoe UI", 14))
        self.current.setAlignment(QtCore.Qt.AlignCenter)
        self.current.setStyleSheet("color: #89b4fa;")
        card_layout.addWidget(self.current)

        # Прогнозируемая цена
        self.forecast = QtWidgets.QLabel(f"🔮 ПРОГНОЗИРУЕМАЯ ЦЕНА: ${self.forecast_result['predicted_price']:.2f}")
        self.forecast.setFont(QtGui.QFont("Segoe UI", 18, QtGui.QFont.Bold))
        self.forecast.setAlignment(QtCore.Qt.AlignCenter)
        self.forecast.setStyleSheet(
            "color: #fab387; padding: 10px; background: rgba(250, 179, 135, 0.1); border-radius: 15px;")
        card_layout.addWidget(self.forecast)

        # Ожидаемое изменение
        change = self.forecast_result['expected_return'] * 100
        change_text = f"📈 ОЖИДАЕМОЕ ИЗМЕНЕНИЕ: {change:+.4f}%"
        if change > 0:
            change_color = "#a6e3a1"
        else:
            change_color = "#f38ba8"
        self.change_label = QtWidgets.QLabel(change_text)
        self.change_label.setFont(QtGui.QFont("Segoe UI", 13))
        self.change_label.setAlignment(QtCore.Qt.AlignCenter)
        self.change_label.setStyleSheet(f"color: {change_color}; padding: 8px; font-weight: bold;")
        card_layout.addWidget(self.change_label)

        # Направление
        direction_text = f"🎯 НАПРАВЛЕНИЕ: {self.forecast_result['direction']}"
        self.direction_label = QtWidgets.QLabel(direction_text)
        self.direction_label.setFont(QtGui.QFont("Segoe UI", 13))
        self.direction_label.setAlignment(QtCore.Qt.AlignCenter)
        if self.forecast_result['direction'] == 'Рост':
            self.direction_label.setStyleSheet("color: #a6e3a1;")
        else:
            self.direction_label.setStyleSheet("color: #f38ba8;")
        card_layout.addWidget(self.direction_label)

        # Горизонт прогноза
        days = self.forecast_result['forecast_days']
        self.horizon_label = QtWidgets.QLabel(f"📆 ГОРИЗОНТ ПРОГНОЗА: {days} ДНЕЙ")
        self.horizon_label.setFont(QtGui.QFont("Segoe UI", 11))
        self.horizon_label.setAlignment(QtCore.Qt.AlignCenter)
        self.horizon_label.setStyleSheet("color: #a6adc8; padding: 10px;")
        card_layout.addWidget(self.horizon_label)

        # Лучший классификатор
        if 'best_classifier' in self.forecast_result:
            if self.forecast_result['best_classifier'] is not None:
                self.best_classifier_label = QtWidgets.QLabel(
                    f"🏆 ЛУЧШИЙ КЛАССИФИКАТОР: {self.forecast_result['best_classifier']}")
                self.best_classifier_label.setFont(QtGui.QFont("Segoe UI", 12, QtGui.QFont.Bold))
                self.best_classifier_label.setAlignment(QtCore.Qt.AlignCenter)
                self.best_classifier_label.setStyleSheet(
                    "color: #a6e3a1; padding: 10px; background: rgba(166, 227, 161, 0.1); border-radius: 10px;")
                card_layout.addWidget(self.best_classifier_label)

        # Лучший регрессор
        if 'best_regressor' in self.forecast_result:
            if self.forecast_result['best_regressor'] is not None:
                self.best_regressor_label = QtWidgets.QLabel(
                    f"🏆 ЛУЧШИЙ РЕГРЕССОР: {self.forecast_result['best_regressor']}")
                self.best_regressor_label.setFont(QtGui.QFont("Segoe UI", 12, QtGui.QFont.Bold))
                self.best_regressor_label.setAlignment(QtCore.Qt.AlignCenter)
                self.best_regressor_label.setStyleSheet(
                    "color: #cba6f7; padding: 10px; background: rgba(203, 166, 247, 0.1); border-radius: 10px;")
                card_layout.addWidget(self.best_regressor_label)

        layout.addWidget(card)
        layout.addStretch()
        return tab

    def create_price_chart_tab(self):
        # Вкладка с графиком цены и прогнозов

        tab = QtWidgets.QWidget()
        tab.setStyleSheet("background-color: #2a2a3e;")
        layout = QtWidgets.QVBoxLayout(tab)

        # Создаём график
        figure = Figure(figsize=(12, 8), dpi=100, facecolor='#1e1e2e')
        canvas = FigureCanvas(figure)
        toolbar = NavigationToolbar(canvas, self)
        toolbar.setStyleSheet("background-color: #313244; color: #cdd6f4; border-radius: 10px;")
        layout.addWidget(toolbar)
        layout.addWidget(canvas)

        # Данные для графика
        hist_data = self.forecast_result['hist_data']
        forecast_days = self.forecast_result['forecast_days']
        current_price = self.forecast_result['current_price']

        # Берём последние 150 дней для истории
        plot_days = 150
        if len(hist_data) < plot_days:
            plot_days = len(hist_data)
        plot_data = hist_data.tail(plot_days).copy()

        # Настройка осей
        ax = figure.add_subplot(111)
        ax.set_facecolor('#313244')
        figure.patch.set_facecolor('#1e1e2e')

        # Рисуем исторические данные
        ax.plot(plot_data['date'], plot_data['price'], '#89b4fa', linewidth=2.5, label='Исторические данные', alpha=0.9)

        # Скользящие средние
        ma20 = plot_data['price'].rolling(20).mean()
        ma50 = plot_data['price'].rolling(50).mean()
        ax.plot(plot_data['date'], ma20, '#fab387', linewidth=1.5, label='MA 20', alpha=0.8)
        ax.plot(plot_data['date'], ma50, '#cba6f7', linewidth=1.5, label='MA 50', alpha=0.8)

        # Последняя дата в истории
        last_date = plot_data['date'].iloc[-1]

        # Генерируем даты для прогноза (только рабочие дни)
        forecast_dates = []
        current_date = last_date
        while len(forecast_dates) < forecast_days:
            current_date = current_date + pd.Timedelta(days=1)
            if current_date.weekday() < 5:
                forecast_dates.append(current_date)

        # Прогнозы ARIMA и SARIMA
        arima_prices = self.forecast_result.get('arima_forecast_prices', [])
        sarima_prices = self.forecast_result.get('sarima_forecast_prices', [])

        # === ЛУЧШАЯ ML МОДЕЛЬ (регрессор) ===
        best_regressor = self.forecast_result.get('best_regressor', None)
        predicted_return_ml = self.forecast_result.get('predicted_return_ml', None)

        # Строим прогноз лучшего ML регрессора (линейная интерполяция по дням)
        if best_regressor is not None and predicted_return_ml is not None:
            if not np.isnan(predicted_return_ml):
                # Преобразуем доходность в проценты
                ml_return_pct = predicted_return_ml / 100.0  # так как predicted_return_ml уже в процентах
                # Строим линейную траекторию от текущей цены до прогнозируемой
                ml_final_price = current_price * (1 + ml_return_pct / 100.0)
                ml_prices = np.linspace(current_price, ml_final_price, len(forecast_dates))
                ax.plot(forecast_dates, ml_prices, '#a6e3a1', linewidth=2.5, marker='s', markersize=5,
                        label=f'Лучший ML регрессор: {best_regressor}', linestyle='-', alpha=0.9)

        # === ЛУЧШАЯ МОДЕЛЬ ВРЕМЕННЫХ РЯДОВ ===
        # Определяем, у какой модели меньше RMSE
        time_series_results = self.forecast_result.get('time_series_results', {})
        best_ts_model = None
        best_ts_rmse = float('inf')

        for model_name, metrics in time_series_results.items():
            rmse = metrics.get('RMSE', np.inf)
            if rmse is not None and not np.isnan(rmse):
                if rmse < best_ts_rmse:
                    best_ts_rmse = rmse
                    best_ts_model = model_name

        # Рисуем лучшую модель временных рядов
        if best_ts_model == 'ARIMA':
            if len(arima_prices) == len(forecast_dates):
                ax.plot(forecast_dates, arima_prices, '#fab387', linewidth=2.5, marker='o', markersize=5,
                        label=f'Лучший временной ряд: ARIMA (RMSE={best_ts_rmse:.2f}%)', linestyle='-', alpha=0.9)
        elif best_ts_model == 'SARIMA':
            if len(sarima_prices) == len(forecast_dates):
                ax.plot(forecast_dates, sarima_prices, '#fab387', linewidth=2.5, marker='o', markersize=5,
                        label=f'Лучший временной ряд: SARIMA (RMSE={best_ts_rmse:.2f}%)', linestyle='-', alpha=0.9)

        # === ВТОРАЯ МОДЕЛЬ ДЛЯ СРАВНЕНИЯ (пунктиром) ===
        if best_ts_model == 'ARIMA':
            if len(sarima_prices) == len(forecast_dates):
                ax.plot(forecast_dates, sarima_prices, '#f38ba8', linewidth=1.5, marker='^', markersize=4,
                        label='SARIMA (для сравнения)', linestyle='--', alpha=0.6)
        elif best_ts_model == 'SARIMA':
            if len(arima_prices) == len(forecast_dates):
                ax.plot(forecast_dates, arima_prices, '#f38ba8', linewidth=1.5, marker='s', markersize=4,
                        label='ARIMA (для сравнения)', linestyle='--', alpha=0.6)
        else:
            # Если нет лучшей модели, показываем обе
            if len(arima_prices) == len(forecast_dates):
                ax.plot(forecast_dates, arima_prices, '#fab387', linewidth=2, marker='o', markersize=4,
                        label='ARIMA прогноз', linestyle='-', alpha=0.8)
            if len(sarima_prices) == len(forecast_dates):
                ax.plot(forecast_dates, sarima_prices, '#f38ba8', linewidth=2, marker='^', markersize=4,
                        label='SARIMA прогноз', linestyle='-', alpha=0.8)

        # Вертикальная линия в текущий момент
        ax.axvline(x=last_date, color='#f38ba8', linestyle='--', alpha=0.8, linewidth=2, label='Текущий момент')

        # Зона неопределённости (доверительный интервал)
        if len(plot_data['price']) > 20:
            # Рассчитываем волатильность
            volatility = plot_data['price'].pct_change().std() * np.sqrt(252)
            daily_vol = volatility / np.sqrt(252) if volatility > 0 else 0.02

            # Используем прогноз лучшей модели для интервала
            base_forecast = None
            if best_ts_model == 'ARIMA' and len(arima_prices) == len(forecast_dates):
                base_forecast = arima_prices
            elif best_ts_model == 'SARIMA' and len(sarima_prices) == len(forecast_dates):
                base_forecast = sarima_prices
            elif best_regressor is not None and predicted_return_ml is not None:
                if not np.isnan(predicted_return_ml):
                    ml_final_price = current_price * (1 + predicted_return_ml / 100.0)
                    base_forecast = np.linspace(current_price, ml_final_price, len(forecast_dates))

            if base_forecast is not None:
                sqrt_time = np.sqrt(np.arange(1, len(forecast_dates) + 1))
                upper = []
                lower = []
                for p, t in zip(base_forecast, sqrt_time):
                    upper.append(p * (1 + 1.96 * daily_vol * t))
                    lower.append(p * (1 - 1.96 * daily_vol * t))
                ax.fill_between(forecast_dates, lower, upper, alpha=0.2, color='#fab387',
                                label='95% доверительный интервал')

        # Настройки графика
        ax.set_title(f'{self.ticker} - Сравнение прогнозов на {forecast_days} дней\n'
                     f'Лучший классификатор: {self.forecast_result["best_classifier"]} | '
                     f'Лучший регрессор: {self.forecast_result["best_regressor"]} | '
                     f'Лучший временной ряд: {best_ts_model if best_ts_model else "нет данных"}',
                     fontsize=12, fontweight='bold', color='#ffffff', pad=15)
        ax.set_xlabel('Дата', fontsize=11, color='#cdd6f4', labelpad=10)
        ax.set_ylabel('Цена ($)', fontsize=11, color='#cdd6f4', labelpad=10)
        ax.legend(loc='best', facecolor='#313244', edgecolor='#fab387', fontsize=8)
        ax.grid(True, alpha=0.15, color='#cdd6f4', linestyle='--')
        ax.tick_params(colors='#cdd6f4', labelsize=9)

        figure.tight_layout()
        canvas.draw()
        return tab

    def create_indicators_tab(self):
        # Вкладка с техническими индикаторами

        tab = QtWidgets.QWidget()
        tab.setStyleSheet("background-color: #2a2a3e;")
        layout = QtWidgets.QVBoxLayout(tab)

        # Создаём график с тремя подграфиками
        figure = Figure(figsize=(12, 10), dpi=100, facecolor='#1e1e2e')
        canvas = FigureCanvas(figure)
        toolbar = NavigationToolbar(canvas, self)
        toolbar.setStyleSheet("background-color: #313244; color: #cdd6f4;")
        layout.addWidget(toolbar)
        layout.addWidget(canvas)

        # Берём последние 150 дней
        plot_data = self.forecast_result['hist_data'].tail(150).copy()

        # График 1: Цена и скользящие средние
        ax1 = figure.add_subplot(311)
        ax1.set_facecolor('#313244')
        ax1.plot(plot_data['date'], plot_data['price'], '#89b4fa', linewidth=2, label='Цена', alpha=0.9)

        ma20 = plot_data['price'].rolling(20).mean()
        ma50 = plot_data['price'].rolling(50).mean()
        ax1.plot(plot_data['date'], ma20, '#fab387', linewidth=1.5, label='MA 20', alpha=0.8)
        ax1.plot(plot_data['date'], ma50, '#cba6f7', linewidth=1.5, label='MA 50', alpha=0.8)

        ax1.set_title('Цена и скользящие средние', fontsize=11, fontweight='bold', color='#ffffff', pad=10)
        ax1.set_ylabel('Цена ($)', fontsize=10, color='#cdd6f4')
        ax1.legend(loc='best', facecolor='#313244', edgecolor='#fab387', fontsize=9)
        ax1.grid(True, alpha=0.15, color='#cdd6f4', linestyle='--')
        ax1.tick_params(colors='#cdd6f4', labelsize=8)

        # График 2: RSI
        ax2 = figure.add_subplot(312)
        ax2.set_facecolor('#313244')

        # Расчёт RSI
        delta = plot_data['price'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-6)
        rsi = 100 - (100 / (1 + rs))

        ax2.plot(plot_data['date'], rsi, '#f38ba8', linewidth=2, label='RSI', alpha=0.9)
        ax2.axhline(70, color='#fab387', linestyle='--', alpha=0.8, linewidth=1.5, label='Перекупленность (70)')
        ax2.axhline(50, color='#cdd6f4', linestyle='--', alpha=0.3, linewidth=1)
        ax2.axhline(30, color='#a6e3a1', linestyle='--', alpha=0.8, linewidth=1.5, label='Перепроданность (30)')
        ax2.fill_between(plot_data['date'], 30, 70, alpha=0.1, color='#a6e3a1')

        ax2.set_title('RSI (Индекс относительной силы)', fontsize=11, fontweight='bold', color='#ffffff', pad=10)
        ax2.set_ylabel('RSI', fontsize=10, color='#cdd6f4')
        ax2.set_ylim(0, 100)
        ax2.legend(loc='best', facecolor='#313244', edgecolor='#fab387', fontsize=9)
        ax2.grid(True, alpha=0.15, color='#cdd6f4', linestyle='--')
        ax2.tick_params(colors='#cdd6f4', labelsize=8)

        # График 3: Дневная доходность
        ax3 = figure.add_subplot(313)
        ax3.set_facecolor('#313244')
        returns = plot_data['price'].pct_change() * 100

        # Цвета: красный для падения, зелёный для роста
        colors = []
        for x in returns:
            if x < 0:
                colors.append('#f38ba8')
            else:
                colors.append('#a6e3a1')

        ax3.bar(plot_data['date'], returns, color=colors, alpha=0.7, width=0.8)
        ax3.axhline(0, color='#cdd6f4', linewidth=1, alpha=0.8)

        ax3.set_title('Дневная доходность', fontsize=11, fontweight='bold', color='#ffffff', pad=10)
        ax3.set_xlabel('Дата', fontsize=10, color='#cdd6f4')
        ax3.set_ylabel('Доходность (%)', fontsize=10, color='#cdd6f4')
        ax3.grid(True, alpha=0.15, color='#cdd6f4', linestyle='--', axis='y')
        ax3.tick_params(colors='#cdd6f4', labelsize=8)

        figure.subplots_adjust(hspace=0.35)
        canvas.draw()
        return tab

    def create_classification_tab(self):
        # Вкладка со сравнением классификаторов

        tab = QtWidgets.QWidget()
        tab.setStyleSheet("background-color: #2a2a3e;")
        layout = QtWidgets.QVBoxLayout(tab)

        # Создаём график
        figure = Figure(figsize=(13, 10), dpi=100, facecolor='#1e1e2e')
        canvas = FigureCanvas(figure)
        toolbar = NavigationToolbar(canvas, self)
        toolbar.setStyleSheet("background-color: #313244; color: #cdd6f4;")
        layout.addWidget(toolbar)
        layout.addWidget(canvas)

        # Данные
        results = self.forecast_result['classification_results']
        best_model = self.forecast_result['best_classifier']
        models = list(results.keys())

        # Цвета для моделей
        model_colors = {
            'Logistic Regression': '#89b4fa',
            'Random Forest (Clf)': '#fab387',
            'XGBoost (Clf)': '#a6e3a1',
            'LightGBM (Clf)': '#f38ba8',
            'CatBoost (Clf)': '#cba6f7',
            'MLP (Clf)': '#f9e2af'
        }

        # Сокращённые названия
        short_names = {
            'Logistic Regression': 'Logistic',
            'Random Forest (Clf)': 'RF',
            'XGBoost (Clf)': 'XGB',
            'LightGBM (Clf)': 'LGBM',
            'CatBoost (Clf)': 'CatB',
            'MLP (Clf)': 'MLP'
        }

        models_short = []
        for m in models:
            models_short.append(short_names.get(m, m))

        # Данные для графиков
        roc_auc = []
        f1 = []
        for m in models:
            roc_auc.append(results[m]['roc_auc'])
            f1.append(results[m]['f1'])

        # График 1: ROC-AUC
        ax1 = figure.add_subplot(221)
        ax1.set_facecolor('#313244')
        bars1 = []
        for i, model in enumerate(models):
            color = model_colors.get(model, '#89b4fa')
            bar = ax1.bar(models_short[i], roc_auc[i], color=color, edgecolor='white', linewidth=0.5, alpha=0.8)
            bars1.append(bar)
        ax1.axhline(y=0.5, color='#f38ba8', linestyle='--', alpha=0.7, linewidth=1.5, label='Случайное угадывание')
        ax1.set_ylabel('ROC-AUC', fontsize=10, color='#cdd6f4')
        ax1.set_title('Сравнение по ROC-AUC', fontsize=11, fontweight='bold', color='#ffffff')
        ax1.legend(loc='lower right', facecolor='#313244', edgecolor='#fab387')
        ax1.tick_params(axis='x', rotation=45, colors='#cdd6f4', labelsize=8)
        ax1.tick_params(axis='y', colors='#cdd6f4', labelsize=9)
        ax1.set_ylim(0.4, 0.65)

        for bar, val in zip(bars1, roc_auc):
            ax1.text(bar[0].get_x() + bar[0].get_width() / 2, bar[0].get_height() + 0.003, f'{val:.4f}',
                     ha='center', fontsize=8, color='#ffffff', fontweight='bold')

        # График 2: F1-score
        ax2 = figure.add_subplot(222)
        ax2.set_facecolor('#313244')
        bars2 = []
        for i, model in enumerate(models):
            color = model_colors.get(model, '#89b4fa')
            bar = ax2.bar(models_short[i], f1[i], color=color, edgecolor='white', linewidth=0.5, alpha=0.8)
            bars2.append(bar)
        ax2.set_ylabel('F1-score', fontsize=10, color='#cdd6f4')
        ax2.set_title('Сравнение по F1-score', fontsize=11, fontweight='bold', color='#ffffff')
        ax2.tick_params(axis='x', rotation=45, colors='#cdd6f4', labelsize=8)
        ax2.tick_params(axis='y', colors='#cdd6f4', labelsize=9)
        ax2.set_ylim(0, 1)

        for bar, val in zip(bars2, f1):
            ax2.text(bar[0].get_x() + bar[0].get_width() / 2, bar[0].get_height() + 0.01, f'{val:.4f}',
                     ha='center', fontsize=8, color='#ffffff', fontweight='bold')

        # График 3: Детальное сравнение метрик
        ax3 = figure.add_subplot(223)
        ax3.set_facecolor('#313244')
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1']
        x = np.arange(len(metric_names))
        width = 0.13

        for i, model in enumerate(models):
            values = [
                results[model]['accuracy'],
                results[model]['precision'],
                results[model]['recall'],
                results[model]['f1']
            ]
            color = model_colors.get(model, '#89b4fa')
            offset = (i - len(models) / 2 + 0.5) * width
            bars = ax3.bar(x + offset, values, width, label=models_short[i],
                           color=color, alpha=0.85, edgecolor='white', linewidth=0.5)

            for bar, val in zip(bars, values):
                if val > 0.05:
                    ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{val:.3f}',
                             ha='center', fontsize=7, color='#ffffff')

        ax3.set_xticks(x)
        ax3.set_xticklabels(metric_names, color='#cdd6f4', fontsize=8)
        ax3.set_ylabel('Значение метрики', fontsize=10, color='#cdd6f4')
        ax3.set_title('Детальное сравнение метрик', fontsize=11, fontweight='bold', color='#ffffff')
        ax3.legend(loc='lower right', facecolor='#313244', edgecolor='#fab387', fontsize=7, ncol=2)
        ax3.set_ylim(0, 1.05)
        ax3.tick_params(colors='#cdd6f4', labelsize=8)
        ax3.grid(True, alpha=0.15, color='#cdd6f4', axis='y', linestyle='--')

        # График 4: Сводная таблица
        ax4 = figure.add_subplot(224)
        ax4.axis('off')
        ax4.set_facecolor('#313244')

        table_data = []
        headers = ['Модель', 'ROC-AUC', 'F1', 'Acc', 'Prec', 'Rec']
        table_data.append(headers)

        for model in models:
            short_name = short_names.get(model, model)
            table_data.append([
                short_name,
                f"{results[model]['roc_auc']:.4f}",
                f"{results[model]['f1']:.4f}",
                f"{results[model]['accuracy']:.4f}",
                f"{results[model]['precision']:.4f}",
                f"{results[model]['recall']:.4f}"
            ])

        table = ax4.table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)

        for i in range(len(table_data)):
            for j in range(len(headers)):
                cell = table[(i, j)]
                if i == 0:
                    cell.set_facecolor('#fab387')
                    cell.set_text_props(color='#1e1e2e', fontweight='bold')
                else:
                    model_name = table_data[i][0]
                    full_model_name = None
                    for short, full in short_names.items():
                        if short == model_name:
                            full_model_name = full
                            break
                    if full_model_name is not None:
                        color = model_colors.get(full_model_name, '#89b4fa')
                        cell.set_facecolor(color)
                        cell.set_text_props(color='#1e1e2e', fontweight='bold')
                        if full_model_name == best_model:
                            current_text = cell.get_text().get_text()
                            cell.get_text().set_text(f"★ {current_text}")
                    else:
                        cell.set_facecolor('#313244')
                        cell.set_text_props(color='#cdd6f4')

        ax4.set_title('СВОДНАЯ ТАБЛИЦА\n★ - лучшая модель', fontsize=10, fontweight='bold', color='#ffffff', pad=20)

        figure.tight_layout()
        canvas.draw()
        return tab

    def create_regression_tab(self):
        # Вкладка со сравнением регрессоров

        tab = QtWidgets.QWidget()
        tab.setStyleSheet("background-color: #2a2a3e;")
        layout = QtWidgets.QVBoxLayout(tab)

        figure = Figure(figsize=(13, 8), dpi=100, facecolor='#1e1e2e')
        canvas = FigureCanvas(figure)
        toolbar = NavigationToolbar(canvas, self)
        toolbar.setStyleSheet("background-color: #313244; color: #cdd6f4;")
        layout.addWidget(toolbar)
        layout.addWidget(canvas)

        results = self.forecast_result['regression_results']
        best_model = self.forecast_result['best_regressor']
        models = list(results.keys())

        # Цвета для моделей
        model_colors = {
            'Linear Regression': '#89b4fa',
            'Ridge Regression': '#89b4fa',
            'Lasso Regression': '#89b4fa',
            'Random Forest (Reg)': '#fab387',
            'XGBoost (Reg)': '#a6e3a1',
            'LightGBM (Reg)': '#f38ba8',
            'CatBoost (Reg)': '#cba6f7',
            'MLP (Reg)': '#f9e2af'
        }

        # Сокращённые названия
        short_names = {
            'Linear Regression': 'Linear',
            'Ridge Regression': 'Ridge',
            'Lasso Regression': 'Lasso',
            'Random Forest (Reg)': 'RF',
            'XGBoost (Reg)': 'XGB',
            'LightGBM (Reg)': 'LGBM',
            'CatBoost (Reg)': 'CatB',
            'MLP (Reg)': 'MLP'
        }

        models_short = []
        for m in models:
            models_short.append(short_names.get(m, m))

        # Данные для графиков
        rmse = []
        mae = []
        for m in models:
            rmse.append(results[m]['RMSE'])
            mae.append(results[m]['MAE'])

        # График 1: RMSE
        ax1 = figure.add_subplot(121)
        ax1.set_facecolor('#313244')
        for i, model in enumerate(models):
            color = model_colors.get(model, '#89b4fa')
            ax1.bar(models_short[i], rmse[i], color=color, edgecolor='white', linewidth=0.5, alpha=0.8)
        ax1.set_ylabel('RMSE (%)', fontsize=10, color='#cdd6f4')
        ax1.set_title('RMSE (меньше - лучше)', fontsize=11, fontweight='bold', color='#ffffff')
        ax1.tick_params(axis='x', rotation=45, colors='#cdd6f4', labelsize=8)
        ax1.tick_params(axis='y', colors='#cdd6f4', labelsize=9)

        # График 2: MAE
        ax2 = figure.add_subplot(122)
        ax2.set_facecolor('#313244')
        for i, model in enumerate(models):
            color = model_colors.get(model, '#89b4fa')
            ax2.bar(models_short[i], mae[i], color=color, edgecolor='white', linewidth=0.5, alpha=0.8)
        ax2.set_ylabel('MAE (%)', fontsize=10, color='#cdd6f4')
        ax2.set_title('MAE (меньше - лучше)', fontsize=11, fontweight='bold', color='#ffffff')
        ax2.tick_params(axis='x', rotation=45, colors='#cdd6f4', labelsize=8)
        ax2.tick_params(axis='y', colors='#cdd6f4', labelsize=9)

        figure.suptitle(f'СРАВНЕНИЕ 8 ML РЕГРЕССОРОВ\n★ Лучшая модель: {best_model}',
                        fontsize=12, fontweight='bold', color='#ffffff', y=0.98)
        figure.tight_layout()
        canvas.draw()
        return tab

    def create_time_series_tab(self):
        # Вкладка с результатами ARIMA и SARIMA

        tab = QtWidgets.QWidget()
        tab.setStyleSheet("background-color: #2a2a3e;")
        layout = QtWidgets.QVBoxLayout(tab)

        figure = Figure(figsize=(12, 6), dpi=100, facecolor='#1e1e2e')
        canvas = FigureCanvas(figure)
        toolbar = NavigationToolbar(canvas, self)
        toolbar.setStyleSheet("background-color: #313244; color: #cdd6f4;")
        layout.addWidget(toolbar)
        layout.addWidget(canvas)

        results = self.forecast_result['time_series_results']
        models = list(results.keys())

        mae = []
        rmse = []
        for m in models:
            mae.append(results[m]['MAE'])
            rmse.append(results[m]['RMSE'])

        # График 1: MAE
        ax1 = figure.add_subplot(121)
        ax1.set_facecolor('#313244')
        bars1 = ax1.bar(models, mae, color=['#89b4fa', '#f38ba8'], edgecolor='white', linewidth=0.5)
        ax1.set_ylabel('MAE (%)', fontsize=10, color='#cdd6f4')
        ax1.set_title('Средняя абсолютная ошибка', fontsize=11, fontweight='bold', color='#ffffff')
        ax1.tick_params(colors='#cdd6f4', labelsize=9)
        for bar, val in zip(bars1, mae):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f'{val:.2f}%',
                     ha='center', fontsize=9, color='#ffffff', fontweight='bold')

        # График 2: RMSE
        ax2 = figure.add_subplot(122)
        ax2.set_facecolor('#313244')
        bars2 = ax2.bar(models, rmse, color=['#89b4fa', '#f38ba8'], edgecolor='white', linewidth=0.5)
        ax2.set_ylabel('RMSE (%)', fontsize=10, color='#cdd6f4')
        ax2.set_title('Корень из среднеквадратичной ошибки', fontsize=11, fontweight='bold', color='#ffffff')
        ax2.tick_params(colors='#cdd6f4', labelsize=9)
        for bar, val in zip(bars2, rmse):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f'{val:.2f}%',
                     ha='center', fontsize=9, color='#ffffff', fontweight='bold')

        figure.suptitle('РЕЗУЛЬТАТЫ ARIMA И SARIMA', fontsize=12, fontweight='bold', color='#ffffff', y=0.98)
        figure.tight_layout()
        canvas.draw()
        return tab

    def save_forecast(self):
        # Сохраняет прогноз в базу данных

        timeframe = f"{self.forecast_result['forecast_days']} дней"

        best_classifier = None
        if 'best_classifier' in self.forecast_result:
            best_classifier = self.forecast_result['best_classifier']

        best_regressor = None
        if 'best_regressor' in self.forecast_result:
            best_regressor = self.forecast_result['best_regressor']

        self.db.save_forecast(
            self.ticker, self.company_name, timeframe,
            self.forecast_result['predicted_price'],
            self.forecast_result['expected_return'],
            self.forecast_result['direction'],
            best_classifier,
            best_regressor
        )

        msg = QtWidgets.QMessageBox()
        msg.setWindowTitle("Прогноз сохранён")
        msg.setText(f"✅ Прогноз для {self.ticker} сохранён!\n\n"
                    f"Цена: ${self.forecast_result['predicted_price']:.2f}\n"
                    f"Изменение: {self.forecast_result['expected_return'] * 100:+.4f}%\n"
                    f"Классификатор: {best_classifier}\n"
                    f"Регрессор: {best_regressor}")
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setStyleSheet(
            "QLabel{color: #cdd6f4; background-color: #1e1e2e;} QPushButton{background-color: #89b4fa; color: #1e1e2e; border-radius: 8px; padding: 5px;}")
        msg.exec_()


class ForecastsListWin(QtWidgets.QWidget):
    # Окно со списком сохранённых прогнозов

    def __init__(self, db):
        super().__init__()
        self.db = db
        self.setWindowTitle("Мои прогнозы")
        self.setGeometry(300, 200, 1200, 500)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)
        self.initUI()

    def initUI(self):
        self.setStyleSheet("background-color: #1e1e2e;")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)

        # Заголовок
        self.title = QtWidgets.QLabel("📊 МОИ ПРОГНОЗЫ")
        self.title.setFont(QtGui.QFont("Segoe UI", 18, QtGui.QFont.Bold))
        self.title.setAlignment(QtCore.Qt.AlignCenter)
        self.title.setStyleSheet("color: #fab387; padding: 15px;")
        layout.addWidget(self.title)

        # Таблица
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(10)
        self.table.setHorizontalHeaderLabels(
            ["№", "Тикер", "Компания", "Горизонт", "Прогноз цена", "Изменение", "Направление", "Классификатор",
             "Регрессор", "Дата"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QtWidgets.QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QTableWidget.SelectRows)
        self.table.setAlternatingRowColors(True)

        self.table.setStyleSheet("""
            QTableWidget {
                background-color: #ffffff;
                color: #1e1e2e;
                border: 2px solid #89b4fa;
                border-radius: 15px;
                gridline-color: #c0c0c0;
            }
            QTableWidget::item {
                padding: 10px;
                color: #1e1e2e;
            }
            QTableWidget::item:selected {
                background-color: #89b4fa;
                color: #ffffff;
            }
            QHeaderView::section {
                background-color: #89b4fa;
                color: #ffffff;
                padding: 10px;
                font-weight: bold;
                border: none;
            }
        """)
        layout.addWidget(self.table)

        # Кнопки
        btn_layout = QtWidgets.QHBoxLayout()

        self.delete_btn = QtWidgets.QPushButton("🗑️ УДАЛИТЬ ВЫБРАННЫЙ")
        self.delete_btn.setFixedSize(200, 40)
        self.delete_btn.setFont(QtGui.QFont("Segoe UI", 11, QtGui.QFont.Bold))
        self.delete_btn.setStyleSheet("background-color: #f38ba8; color: #1e1e2e; border-radius: 12px;")
        self.delete_btn.clicked.connect(self.delete_selected)
        btn_layout.addWidget(self.delete_btn)

        self.refresh_btn = QtWidgets.QPushButton("🔄 ОБНОВИТЬ")
        self.refresh_btn.setFixedSize(140, 40)
        self.refresh_btn.setFont(QtGui.QFont("Segoe UI", 11, QtGui.QFont.Bold))
        self.refresh_btn.setStyleSheet("background-color: #89b4fa; color: #1e1e2e; border-radius: 12px;")
        self.refresh_btn.clicked.connect(self.load_forecasts)
        btn_layout.addWidget(self.refresh_btn)

        self.close_btn = QtWidgets.QPushButton("❌ ЗАКРЫТЬ")
        self.close_btn.setFixedSize(140, 40)
        self.close_btn.setFont(QtGui.QFont("Segoe UI", 11, QtGui.QFont.Bold))
        self.close_btn.setStyleSheet("background-color: #a6e3a1; color: #1e1e2e; border-radius: 12px;")
        self.close_btn.clicked.connect(self.close)
        btn_layout.addWidget(self.close_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        self.load_forecasts()

    def load_forecasts(self):
        # Загружает и отображает все прогнозы из базы

        self.table.setRowCount(0)
        forecasts = self.db.get_all_forecasts()

        # Если нет прогнозов, показываем сообщение
        if not forecasts:
            self.table.setRowCount(1)
            self.table.setSpan(0, 0, 1, 10)
            no_data_item = QtWidgets.QTableWidgetItem("📭 НЕТ СОХРАНЁННЫХ ПРОГНОЗОВ")
            no_data_item.setTextAlignment(QtCore.Qt.AlignCenter)
            no_data_item.setFlags(QtCore.Qt.NoItemFlags)
            no_data_item.setForeground(QtGui.QColor("#fab387"))
            no_data_item.setFont(QtGui.QFont("Segoe UI", 14))
            self.table.setItem(0, 0, no_data_item)
            return

        # Настраиваем ширину колонок
        self.table.setRowCount(len(forecasts))
        self.table.setColumnWidth(0, 45)
        self.table.setColumnWidth(1, 70)
        self.table.setColumnWidth(2, 200)
        self.table.setColumnWidth(3, 80)
        self.table.setColumnWidth(4, 100)
        self.table.setColumnWidth(5, 90)
        self.table.setColumnWidth(6, 80)
        self.table.setColumnWidth(7, 110)
        self.table.setColumnWidth(8, 110)

        # Заполняем таблицу
        for row, forecast in enumerate(forecasts):
            # Номер
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(row + 1)))

            # Тикер
            ticker_item = QtWidgets.QTableWidgetItem(forecast[1])
            ticker_item.setForeground(QtGui.QColor("#1e1e2e"))
            ticker_item.setFont(QtGui.QFont("Segoe UI", 11, QtGui.QFont.Bold))
            self.table.setItem(row, 1, ticker_item)

            # Компания
            self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(forecast[2][:50]))

            # Горизонт
            self.table.setItem(row, 3, QtWidgets.QTableWidgetItem(forecast[3]))

            # Прогноз цены
            if forecast[4] is not None:
                price_text = f"${forecast[4]:.2f}"
            else:
                price_text = "—"
            self.table.setItem(row, 4, QtWidgets.QTableWidgetItem(price_text))

            # Изменение
            if forecast[5] is not None:
                return_text = f"{forecast[5] * 100:+.4f}%"
                return_item = QtWidgets.QTableWidgetItem(return_text)
                if forecast[5] > 0:
                    return_item.setForeground(QtGui.QColor("#00aa00"))
                elif forecast[5] < 0:
                    return_item.setForeground(QtGui.QColor("#cc0000"))
            else:
                return_item = QtWidgets.QTableWidgetItem("—")
            self.table.setItem(row, 5, return_item)

            # Направление
            direction_item = QtWidgets.QTableWidgetItem(forecast[6] if forecast[6] else "—")
            if forecast[6] == "Рост":
                direction_item.setForeground(QtGui.QColor("#00aa00"))
            elif forecast[6] == "Падение":
                direction_item.setForeground(QtGui.QColor("#cc0000"))
            self.table.setItem(row, 6, direction_item)

            # Классификатор
            self.table.setItem(row, 7, QtWidgets.QTableWidgetItem(forecast[7] if forecast[7] else "—"))

            # Регрессор
            self.table.setItem(row, 8, QtWidgets.QTableWidgetItem(forecast[8] if forecast[8] else "—"))

            # Дата
            date_str = forecast[9]
            try:
                formatted_date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S").strftime("%d.%m.%Y %H:%M")
            except:
                formatted_date = date_str
            self.table.setItem(row, 9, QtWidgets.QTableWidgetItem(formatted_date))

        self.table.resizeColumnsToContents()

    def delete_selected(self):
        # Удаляет выбранный прогноз

        selected = self.table.currentRow()
        if selected >= 0:
            forecasts = self.db.get_all_forecasts()
            if selected < len(forecasts):
                reply = QtWidgets.QMessageBox.question(
                    self, "Подтверждение",
                    f"Удалить прогноз для {forecasts[selected][1]}?",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
                )
                if reply == QtWidgets.QMessageBox.Yes:
                    self.db.delete_forecast(forecasts[selected][0])
                    self.load_forecasts()


class InfoWin(QtWidgets.QWidget):
    # Окно с информацией о компании и кнопкой прогноза

    def __init__(self, ticker, data, db, prediction_engine):
        super().__init__()
        self.ticker = ticker
        self.data = data
        self.db = db
        self.prediction_engine = prediction_engine
        self.setWindowTitle(f"Компания: {ticker}")
        self.setGeometry(200, 100, 1200, 800)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)
        self.initUI()

    def initUI(self):
        self.setStyleSheet("background-color: #1e1e2e;")

        # Основной layout
        main = QtWidgets.QHBoxLayout(self)
        main.setContentsMargins(20, 20, 20, 20)

        # Левая панель - информация о компании
        left = QtWidgets.QVBoxLayout()

        # Название компании
        self.title = QtWidgets.QLabel(f"📈 {self.data['name'][:80]}")
        self.title.setFont(QtGui.QFont("Segoe UI", 18, QtGui.QFont.Bold))
        self.title.setAlignment(QtCore.Qt.AlignCenter)
        self.title.setStyleSheet("color: #fab387; padding: 15px;")
        left.addWidget(self.title)

        # Тикер
        self.tick = QtWidgets.QLabel(f"🏷️ Тикер: {self.ticker}")
        self.tick.setFont(QtGui.QFont("Segoe UI", 14, QtGui.QFont.Bold))
        self.tick.setAlignment(QtCore.Qt.AlignCenter)
        self.tick.setStyleSheet("color: #a6e3a1; padding: 8px;")
        left.addWidget(self.tick)

        # Область прокрутки для фундаментальных данных
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("border: none;")

        # Виджет с данными
        info_widget = QtWidgets.QWidget()
        info_widget.setStyleSheet("background-color: #ffffff; border-radius: 10px;")
        info_layout = QtWidgets.QGridLayout(info_widget)
        info_layout.setSpacing(12)

        # Список полей для отображения
        fields = [
            ("📊 Изменение", "change", "%"),
            ("🏭 Сектор", "sector", ""),
            ("📦 Объём торгов", "volume", "шт"),
            ("📈 Относит. объём", "rel_vol", "x"),
            ("🏷️ P/E (Цена/Прибыль)", "pe", ""),
            ("💎 EPS (разводн.)", "eps", "USD/акцию"),
            ("📈 Рост EPS", "eps_growth", "%"),
            ("💸 Див. доходность", "div_yield", "%")
        ]

        # Заполняем данными
        for i, (label_text, key, unit) in enumerate(fields):
            # Лейбл
            lbl = QtWidgets.QLabel(f"{label_text}:")
            lbl.setFont(QtGui.QFont("Segoe UI", 11, QtGui.QFont.Bold))
            lbl.setStyleSheet("color: #1e1e2e; padding: 10px; background-color: #f0f0f0; border-radius: 8px;")

            # Значение
            val = self.data.get(key, "Нет данных")

            # Преобразуем значение для отображения
            if isinstance(val, float):
                if not pd.isna(val):
                    if key == 'change':
                        val = f"{val:+.2f}{unit}"
                    elif key == 'eps':
                        val = f"{val:.2f} {unit}"
                    elif key == 'eps_growth':
                        val = f"{val:+.2f}{unit}"
                    elif key == 'div_yield':
                        val = f"{val:.2f}{unit}"
                    elif key == 'volume':
                        if val >= 1_000_000:
                            val = f"{val / 1_000_000:.2f} млн {unit}"
                        else:
                            val = f"{val:,.0f} {unit}"
                    elif key == 'rel_vol':
                        val = f"{val:.2f}{unit}"
                    else:
                        val = f"{val:.2f}"
                else:
                    val = "Нет данных"
            else:
                if pd.isna(val) or val == "":
                    val = "Нет данных"

            val_lbl = QtWidgets.QLabel(str(val))
            val_lbl.setStyleSheet("color: #1e1e2e; padding: 10px; background-color: #f0f0f0; border-radius: 8px;")

            info_layout.addWidget(lbl, i, 0)
            info_layout.addWidget(val_lbl, i, 1)

        self.scroll.setWidget(info_widget)
        left.addWidget(self.scroll)

        # Правая панель - прогноз
        right = QtWidgets.QVBoxLayout()
        right.setSpacing(20)

        # Картинка
        self.img = QtWidgets.QLabel()
        pix = QtGui.QPixmap("sec_picture.jpg")
        if not pix.isNull():
            self.img.setPixmap(pix.scaled(550, 400, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        else:
            self.img.setText("📊 График NASDAQ")
            self.img.setAlignment(QtCore.Qt.AlignCenter)
            self.img.setStyleSheet(
                "color: #fab387; font-size: 18px; border: 2px solid #fab387; border-radius: 15px; padding: 40px;")
        right.addWidget(self.img)

        # Выбор количества дней
        days_layout = QtWidgets.QHBoxLayout()
        days_label = QtWidgets.QLabel("📅 Количество дней для прогноза:")
        days_label.setFont(QtGui.QFont("Segoe UI", 12, QtGui.QFont.Bold))
        days_label.setStyleSheet("color: #fab387;")

        self.forecast_days_spin = QtWidgets.QSpinBox()
        self.forecast_days_spin.setRange(1, 30)
        self.forecast_days_spin.setValue(5)
        self.forecast_days_spin.setStyleSheet(
            "background-color: #f0f0f0; color: #1e1e2e; border: 2px solid #89b4fa; border-radius: 8px; padding: 5px;")

        days_layout.addWidget(days_label)
        days_layout.addWidget(self.forecast_days_spin)
        days_layout.addStretch()
        right.addLayout(days_layout)

        # Прогресс-бар
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet(
            "border: 2px solid #89b4fa; border-radius: 10px; text-align: center; color: #1e1e2e; background-color: #f0f0f0;")
        self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #a6e3a1; border-radius: 8px; }")
        right.addWidget(self.progress_bar)

        # Кнопка прогноза
        self.forecast_btn = QtWidgets.QPushButton("🔮 ПОЛУЧИТЬ ПРОГНОЗ")
        self.forecast_btn.setFixedSize(330, 50)
        self.forecast_btn.setFont(QtGui.QFont("Segoe UI", 13, QtGui.QFont.Bold))
        self.forecast_btn.setStyleSheet("background-color: #89b4fa; color: #1e1e2e; border-radius: 12px;")
        self.forecast_btn.clicked.connect(self.get_forecast)

        btn_box = QtWidgets.QHBoxLayout()
        btn_box.addStretch()
        btn_box.addWidget(self.forecast_btn)
        btn_box.addStretch()
        right.addLayout(btn_box)

        # Кнопка закрытия
        self.close_btn = QtWidgets.QPushButton("❌ ЗАКРЫТЬ")
        self.close_btn.setFixedSize(140, 40)
        self.close_btn.setStyleSheet("background-color: #f38ba8; color: #1e1e2e; border-radius: 10px;")
        self.close_btn.clicked.connect(self.close)

        btn_box2 = QtWidgets.QHBoxLayout()
        btn_box2.addStretch()
        btn_box2.addWidget(self.close_btn)
        btn_box2.addStretch()
        right.addLayout(btn_box2)

        main.addLayout(left, 1)
        main.addLayout(right, 1)

    def get_forecast(self):
        # Запускает прогнозирование

        forecast_days = self.forecast_days_spin.value()
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # Функция обновления прогресса
        def update_progress(message):
            if "Подготовка данных" in message:
                self.progress_bar.setValue(10)
            elif "Масштабирование" in message:
                self.progress_bar.setValue(20)
            elif "Обучение классификаторов" in message:
                self.progress_bar.setValue(40)
            elif "Обучение регрессоров" in message:
                self.progress_bar.setValue(60)
            elif "ARIMA/SARIMA" in message:
                self.progress_bar.setValue(80)
            elif "завершено" in message:
                self.progress_bar.setValue(100)

        self.prediction_engine.set_status_callback(update_progress)

        # Делаем прогноз
        forecast_result = self.prediction_engine.predict(self.ticker, forecast_days)

        self.progress_bar.setVisible(False)

        # Проверяем результат
        if forecast_result is None:
            msg = QtWidgets.QMessageBox()
            msg.setWindowTitle("Ошибка прогноза")
            msg.setText("❌ Не удалось выполнить прогноз.")
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setStyleSheet(
                "QLabel{color: #cdd6f4; background-color: #1e1e2e;} QPushButton{background-color: #89b4fa; color: #1e1e2e; border-radius: 8px; padding: 5px;}")
            msg.exec_()
            return

        # Показываем результаты
        self.forecast_result_win = ForecastResultWin(self.ticker, self.data['name'], forecast_result, self.db)
        self.forecast_result_win.show()