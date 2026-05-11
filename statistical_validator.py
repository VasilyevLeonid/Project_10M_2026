import numpy as np
from sklearn.metrics import roc_auc_score


class StatisticalValidator:
    # Класс для статистической проверки значимости различий между моделями

    def __init__(self, classification_results, y_true, y_proba_dict):
        # Сохраняем результаты классификации
        self.results = classification_results
        self.y_true = y_true
        self.y_proba_dict = y_proba_dict
        self.statistics = None

    def bootstrap_roc_auc_diff(self, y_true, proba1, proba2, n_bootstrap=1000):
        # Вычисляет разницу ROC-AUC между двумя моделями с помощью бутстрапа

        # Наблюдаемая разница
        observed_diff = roc_auc_score(y_true, proba1) - roc_auc_score(y_true, proba2)

        # Массив для хранения разниц на бутстрап-выборках
        diffs = []
        n = len(y_true)

        # Генерируем бутстрап-выборки
        for _ in range(n_bootstrap):
            # Выбираем случайные индексы с возвращением
            idx = np.random.choice(n, n, replace=True)
            try:
                # Считаем ROC-AUC на бутстрап-выборке
                roc1 = roc_auc_score(y_true[idx], proba1[idx])
                roc2 = roc_auc_score(y_true[idx], proba2[idx])
                diffs.append(roc1 - roc2)
            except:
                diffs.append(0)

        # Доверительный интервал (2.5% и 97.5% процентили)
        ci_lower = np.percentile(diffs, 2.5)
        ci_upper = np.percentile(diffs, 97.5)

        # Вычисляем p-value (вероятность, что разница не значима)
        if observed_diff > 0:
            p_value = np.mean(np.array(diffs) <= 0)
        elif observed_diff < 0:
            p_value = np.mean(np.array(diffs) >= 0)
        else:
            p_value = 1.0

        # Ограничиваем p-value снизу
        if p_value < 1.0 / n_bootstrap:
            p_value = 1.0 / n_bootstrap

        return {
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_value': p_value,
            'observed_diff': observed_diff
        }

    def calculate_all_tests(self, best_model_name):
        # Сравнивает лучшую модель со всеми остальными

        results = []

        # Перебираем все модели
        for model_name in self.y_proba_dict.keys():
            # Пропускаем лучшую модель
            if model_name == best_model_name:
                continue

            # Сравниваем лучшую модель с текущей
            stats = self.bootstrap_roc_auc_diff(
                self.y_true,
                self.y_proba_dict[best_model_name],
                self.y_proba_dict[model_name]
            )

            # Разница ROC-AUC
            roc_diff = self.results[best_model_name]['roc_auc'] - self.results[model_name]['roc_auc']

            # Значимо ли отличие (p-value < 0.05)
            is_significant = stats['p_value'] < 0.05

            results.append({
                'model': model_name,
                'roc_auc_diff': roc_diff,
                'ci_lower': stats['ci_lower'],
                'ci_upper': stats['ci_upper'],
                'p_value': stats['p_value'],
                'is_significant': is_significant
            })

        self.statistics = results
        return results

    def get_statistics(self):
        # Возвращает результаты статистических тестов
        return self.statistics