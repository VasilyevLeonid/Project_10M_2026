import sqlite3
from datetime import datetime


class Database:
    # Класс для работы с базой данных прогнозов

    def __init__(self, db_name="forecasts.db"):
        self.db_name = db_name
        self.create_table()

    def create_table(self):
        # Создаёт таблицу для хранения прогнозов, если её нет
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
                best_classifier TEXT,
                best_regressor TEXT,
                created_date TEXT NOT NULL,
                status TEXT DEFAULT 'active'
            )
        ''')
        conn.commit()
        conn.close()

    def save_forecast(self, ticker, company_name, timeframe, forecast_price=None,
                      expected_return=None, direction=None, best_classifier=None, best_regressor=None):
        # Сохраняет прогноз в базу данных
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        created_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute('''
            INSERT INTO forecasts (ticker, company_name, timeframe, forecast_price, expected_return, direction, best_classifier, best_regressor, created_date, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
        ticker, company_name, timeframe, forecast_price, expected_return, direction, best_classifier, best_regressor,
        created_date, 'active'))
        conn.commit()
        forecast_id = cursor.lastrowid
        conn.close()
        return forecast_id

    def get_all_forecasts(self):
        # Получает все активные прогнозы из базы
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, ticker, company_name, timeframe, forecast_price, expected_return, direction, best_classifier, best_regressor, created_date 
            FROM forecasts WHERE status = "active" ORDER BY id DESC
        ''')
        forecasts = cursor.fetchall()
        conn.close()
        return forecasts

    def delete_forecast(self, forecast_id):
        # Удаляет прогноз из базы
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM forecasts WHERE id = ?', (forecast_id,))
        conn.commit()
        conn.close()