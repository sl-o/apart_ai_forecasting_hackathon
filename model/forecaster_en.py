import math
from pathlib import Path
import pandas as pd
import numpy as np

# Get the directory where this script is located
_SCRIPT_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _SCRIPT_DIR.parent

class BayesForecaster:
    """
    Builds Bayesian coefficients k_i = P(S_i|H) / P(S_i|~H)
    and saves them to indicator_likelihood_ratios.csv
    """

    def __init__(
        self,
        indicators_path="../data/list_of_indicators.csv",
        historical_indicators_path="../data/historical_indicators.csv",
        historical_events_path="../data/historical_events.csv",
        ratios_path="../data/indicator_likelihood_ratios.csv",  # <- typo not needed intentionally, we'll fix it below
        prior_prob=0.2,
        success_threshold=0.7,
        laplace_alpha=1.0,
    ):
        """
        prior_prob: base prior probability of the target scenario.
        success_threshold: events with target_score >= this are considered 'successful' (H=1).
        laplace_alpha: smoothing for probabilities (add-one smoothing).
        """

        # fix the filename:
        ratios_path = "../data/indicator_likelihood_ratios.csv"

        # Resolve paths relative to project root for robustness
        # Remove '../' prefix since _PROJECT_ROOT is already at project root
        def normalize_path(path_str):
            path_str = str(path_str)
            if path_str.startswith("../"):
                return path_str[3:]  # Remove '../' prefix
            return path_str
        
        self.indicators_path = (_PROJECT_ROOT / normalize_path(indicators_path)).resolve()
        self.historical_indicators_path = (_PROJECT_ROOT / normalize_path(historical_indicators_path)).resolve()
        self.historical_events_path = (_PROJECT_ROOT / normalize_path(historical_events_path)).resolve()
        self.ratios_path = (_PROJECT_ROOT / normalize_path(ratios_path)).resolve()

        self.prior_prob = float(prior_prob)
        self.success_threshold = float(success_threshold)
        self.laplace_alpha = float(laplace_alpha)

        # load data
        self.indicators_df = pd.read_csv(self.indicators_path)
        # columns: indicator_id, indicator_name, category, ...

        self.hist_ind_df = pd.read_csv(self.historical_indicators_path)
        # columns: event_id, event_name, indicator_id, severity

        self.hist_events_df = pd.read_csv(self.historical_events_path)
        # columns: event_id, event_name, target_score

        self.ratios_df = None

    def _label_events_success(self):
        """
        Add binary label 'is_success':
        is_success = 1, if target_score >= success_threshold
        otherwise 0.
        """
        df = self.hist_events_df.copy()
        df["is_success"] = (df["target_score"] >= self.success_threshold).astype(int)
        return df[["event_id", "is_success"]]

    def fit_likelihoods(self):
        """
        For each indicator, estimate:
          P(S_i | H)     = probability that indicator is active in 'successful' events
          P(S_i | ~H)    = probability that indicator is active in 'unsuccessful' events
        and calculate
          k_ratio = P(S_i|H) / P(S_i|~H)

        Result is stored in self.ratios_df and saved to CSV.
        """

        # 1. binarize event success
        labels_df = self._label_events_success()
        # labels_df: event_id, is_success (0/1)

        # 2. indicator activity by events
        # active = 1 if severity > 0, otherwise 0
        hist_active = self.hist_ind_df.copy()
        hist_active["active"] = (hist_active["severity"] > 0).astype(int)

        # 3. aggregate by (event_id, indicator_id): whether indicator was active
        pivot_df = (
            hist_active
            .groupby(["event_id", "indicator_id"])["active"]
            .max()  # if indicator appears multiple times, take maximum
            .reset_index()
        )

        # 4. join event success flag
        pivot_df = pivot_df.merge(labels_df, on="event_id", how="left")

        # 5. We need to account for even those pairs (event, indicator),
        #    where the indicator wasn't mentioned at all → active=0.
        #    To do this, build a full matrix of events × indicators.

        indicators = self.indicators_df["indicator_id"].unique().tolist()
        events = self.hist_events_df["event_id"].unique().tolist()

        # dictionaries for fast access
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
                active_val = act_lookup.get((ev, ind), 0)  # if indicator wasn't present => consider inactive
                is_succ = success_lookup[ev]
                full_rows.append({
                    "event_id": ev,
                    "indicator_id": ind,
                    "active": active_val,
                    "is_success": is_succ
                })
        full_df = pd.DataFrame(full_rows)
        # now we have for each event and each indicator:
        # - active (0/1)
        # - is_success (0/1)

        succ = full_df[full_df["is_success"] == 1]
        fail = full_df[full_df["is_success"] == 0]

        result_rows = []
        alpha = self.laplace_alpha
        eps = 1e-9

        for ind in indicators:
            # data for this indicator in successful events
            succ_slice = succ[succ["indicator_id"] == ind]
            count_success_total  = len(succ_slice)
            count_success_active = succ_slice["active"].sum()

            # data for unsuccessful events
            fail_slice = fail[fail["indicator_id"] == ind]
            count_fail_total  = len(fail_slice)
            count_fail_active = fail_slice["active"].sum()

            # Laplace smoothing:
            # if there are no successful events - fallback: set p_given_H as 0.5 (neutral)
            if count_success_total == 0:
                p_given_H = 0.5
            else:
                p_given_H = (count_success_active + alpha) / (count_success_total + 2 * alpha)

            # if there are no unsuccessful events - fallback: set p_given_notH = 0.5
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

        # add indicator metadata (name, category)
        ratios_df = ratios_df.merge(
            self.indicators_df[["indicator_id", "description"]],
            on="indicator_id",
            how="left"
        )

        # save
        self.ratios_df = ratios_df
        ratios_df.to_csv(self.ratios_path, index=False)

        return ratios_df


# If this file is run directly as a script:
if __name__ == "__main__":
    forecaster = BayesForecaster(
        indicators_path="../data/list_of_indicators.csv",
        historical_indicators_path="../data/historical_indicators.csv",
        historical_events_path="../data/historical_events.csv",
        # ratios_path by default goes to data/indicator_likelihood_ratios.csv
        prior_prob=0.2,
        success_threshold=0.7,  # event is considered "success" if target_score >= 0.7
        laplace_alpha=1.0       # smoothing (1.0 = add-one smoothing)
    )

    df = forecaster.fit_likelihoods()
    print("indicator_likelihood_ratios.csv created.")
    print(df)

