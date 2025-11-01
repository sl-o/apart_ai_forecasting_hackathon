import math
from pathlib import Path
import pandas as pd
import numpy as np

class BayesForecaster:
    """
    Строит байесовские коэффициенты k_i = P(S_i|H) / P(S_i|~H)
    и сохраняет их в indicator_likelihood_ratios.csv
    """

    def __init__(
        self,
        indicators_path="../data/list_of_indicators.csv",
        historical_indicators_path="../data/historical_indicators.csv",
        historical_events_path="../data/historical_events.csv",
        ratios_path="../data/indicator_likelihood_ratios.csv",  # <- опечатка специально не нужна, ниже исправим
        prior_prob=0.2,
        success_threshold=0.7,
        laplace_alpha=1.0,
    ):
        """
        prior_prob: базовая априорная вероятность целевого сценария.
        success_threshold: события с target_score >= этого считаем 'успешными' (H=1).
        laplace_alpha: сглаживание для вероятностей (add-one smoothing).
        """

        # исправим имя файла:
        ratios_path = "../data/indicator_likelihood_ratios.csv"

        self.indicators_path = Path(indicators_path)
        self.historical_indicators_path = Path(historical_indicators_path)
        self.historical_events_path = Path(historical_events_path)
        self.ratios_path = Path(ratios_path)

        self.prior_prob = float(prior_prob)
        self.success_threshold = float(success_threshold)
        self.laplace_alpha = float(laplace_alpha)

        # загружаем данные
        self.indicators_df = pd.read_csv(self.indicators_path)
        # columns: indicator_id, indicator_name, category, ...

        self.hist_ind_df = pd.read_csv(self.historical_indicators_path)
        # columns: event_id, event_name, indicator_id, severity

        self.hist_events_df = pd.read_csv(self.historical_events_path)
        # columns: event_id, event_name, target_score

        self.ratios_df = None

    def _label_events_success(self):
        """
        Добавляем бинарную метку 'is_success':
        is_success = 1, если target_score >= success_threshold
        иначе 0.
        """
        df = self.hist_events_df.copy()
        df["is_success"] = (df["target_score"] >= self.success_threshold).astype(int)
        return df[["event_id", "is_success"]]

    def fit_likelihoods(self):
        """
        Для каждого индикатора оцениваем:
          P(S_i | H)     = вероятность, что индикатор активен в 'успешных' событиях
          P(S_i | ~H)    = вероятность, что индикатор активен в 'неуспешных' событиях
        и считаем
          k_ratio = P(S_i|H) / P(S_i|~H)

        Результат складываем в self.ratios_df и сохраняем в CSV.
        """

        # 1. бинаризуем успех событий
        labels_df = self._label_events_success()
        # labels_df: event_id, is_success (0/1)

        # 2. активность индикаторов по событиям
        # active = 1 если severity > 0, иначе 0
        hist_active = self.hist_ind_df.copy()
        hist_active["active"] = (hist_active["severity"] > 0).astype(int)

        # 3. сведём по (event_id, indicator_id): был ли индикатор активен
        pivot_df = (
            hist_active
            .groupby(["event_id", "indicator_id"])["active"]
            .max()  # если индикатор прописан несколько раз, берём максимум
            .reset_index()
        )

        # 4. присоединим флаг успеха события
        pivot_df = pivot_df.merge(labels_df, on="event_id", how="left")

        # 5. Нам надо учесть даже те пары (event, indicator),
        #    где индикатор не упоминался вообще → active=0.
        #    Для этого построим полную матрицу событий × индикаторов.

        indicators = self.indicators_df["indicator_id"].unique().tolist()
        events = self.hist_events_df["event_id"].unique().tolist()

        # словари для быстрого доступа
        act_lookup = {
            (row["event_id"], row["indicator_id"]): row["active"]
            for _, row in pivot_df.iterrows()
        }
        success_lookup = {
            row["event_id"]: row["is_success"]
            for _, row in labels_df.iterrows()
        }

        full_rows = []
        for ev in events:
            for ind in indicators:
                active_val = act_lookup.get((ev, ind), 0)  # не было индикатора => считаем неактивен
                is_succ = success_lookup[ev]
                full_rows.append({
                    "event_id": ev,
                    "indicator_id": ind,
                    "active": active_val,
                    "is_success": is_succ
                })
        full_df = pd.DataFrame(full_rows)
        # теперь у нас есть для каждого события и каждого индикатора:
        # - active (0/1)
        # - is_success (0/1)

        succ = full_df[full_df["is_success"] == 1]
        fail = full_df[full_df["is_success"] == 0]

        result_rows = []
        alpha = self.laplace_alpha
        eps = 1e-9

        for ind in indicators:
            # данные по этому индикатору в успешных событиях
            succ_slice = succ[succ["indicator_id"] == ind]
            count_success_total  = len(succ_slice)
            count_success_active = succ_slice["active"].sum()

            # данные по неуспешным событиям
            fail_slice = fail[fail["indicator_id"] == ind]
            count_fail_total  = len(fail_slice)
            count_fail_active = fail_slice["active"].sum()

            # сглаживание Лапласа:
            # если нет успешных событий - fallback: считаем p_given_H как 0.5 (нейтрально)
            if count_success_total == 0:
                p_given_H = 0.5
            else:
                p_given_H = (count_success_active + alpha) / (count_success_total + 2 * alpha)

            # если нет неуспешных событий - fallback: считаем p_given_notH = 0.5
            if count_fail_total == 0:
                p_given_notH = 0.5
            else:
                p_given_notH = (count_fail_active + alpha) / (count_fail_total + 2 * alpha)

            k_ratio = p_given_H / max(p_given_notH, eps)

            result_rows.append({
                "indicator_id": ind,
                "p_given_H": p_given_H,
                "p_given_notH": p_given_notH,
                "k_ratio": k_ratio
            })

        ratios_df = pd.DataFrame(result_rows)

        # добавим метаданные индикаторов (имя, категория)
        ratios_df = ratios_df.merge(
            self.indicators_df[["indicator_id", "description"]],
            on="indicator_id",
            how="left"
        )

        # сохраним
        self.ratios_df = ratios_df
        ratios_df.to_csv(self.ratios_path, index=False)

        return ratios_df


# Если этот файл запустить напрямую как скрипт:
if __name__ == "__main__":
    forecaster = BayesForecaster(
        indicators_path="../data/list_of_indicators.csv",
        historical_indicators_path="../data/historical_indicators.csv",
        historical_events_path="../data/historical_events.csv",
        # ratios_path по умолчанию пойдёт в data/indicator_likelihood_ratios.csv
        prior_prob=0.2,
        success_threshold=0.7,  # событие считаем "успехом", если target_score >= 0.7
        laplace_alpha=1.0       # сглаживание (1.0 = add-one smoothing)
    )

    df = forecaster.fit_likelihoods()
    print("indicator_likelihood_ratios.csv создан.")
    print(df)
