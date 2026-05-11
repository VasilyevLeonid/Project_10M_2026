import pandas as pd
import numpy as np
import re


class DataCleaner:
    # Класс для очистки и преобразования данных

    @staticmethod
    def clean_numeric_string(value):
        # Преобразует строку с числом в число float
        # Проверяем, что значение не пустое
        if pd.isna(value) or value == "" or value is None:
            return np.nan

        # Если уже число, просто возвращаем
        if isinstance(value, (int, float)):
            return float(value)

        # Преобразуем строку: убираем лишние символы, заменяем запятую на точку
        try:
            str_val = str(value).strip().replace(',', '.')
            # Оставляем только цифры, точку и минус
            str_val = re.sub(r'[^\d.\-]', '', str_val)
            return float(str_val)
        except:
            return np.nan

    @staticmethod
    def clean_percent_string(value):
        # Преобразует строку с процентом в число
        if pd.isna(value) or value == "" or value is None:
            return np.nan

        if isinstance(value, (int, float)):
            return float(value)

        try:
            str_val = str(value).strip().replace(',', '.')
            # Убираем знак процента
            str_val = re.sub(r'[%]', '', str_val)
            return float(str_val)
        except:
            return np.nan