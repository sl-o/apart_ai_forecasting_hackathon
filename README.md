# Bayesian Scenario Forecaster

This repo is a prototype for forecasting the likelihood of a target scenario (e.g. formation of a coordinated policy regime) using observable signals from the real world.

It has two main parts:

1. Learn how strong each indicator actually is, using historical events.
2. Given current signals, estimate the probability that the scenario is happening now / about to happen.

This is meant to be simple, auditable, and explainable to non-ML stakeholders.

---

## 1. Core idea

We assume there is some "target scenario" we care about (for example: coordinated international control of advanced compute, enforceable compliance regime, etc.).

We watch the world for **indicators**:

* export controls,
* fab/capacity disruption,
* political salience in top-tier media,
* existence of a funding/compensation mechanism,
* technical/operational feasibility path,
* etc.

We treat these indicators as "evidence." We ask:

* Historically, which indicators actually showed up before real coordination breakthroughs?
* Which indicators are just noise / media panic?

We then do a Bayesian update:

* Start from a prior probability of the scenario.
* Look at which indicators are active now and how strong they are.
* Update to a posterior probability.

So the output is:

> "Given what we're seeing right now, the chance that we're entering the target scenario is X%."

And we can explain exactly which signals pushed it up.

---

## 2. Data inputs

All input CSVs live in `data/`.

### `list_of_indicators.csv`

Reference list of all indicators we track.

Required columns:

* `indicator_id` (string, unique key, no spaces)
* `indicator_name` (human-readable name)
* `category` (`trigger` or `condition`)

Example:

```csv
indicator_id,indicator_name,category
gpu_controls,GPU/export-control changes,trigger
fab_disruption,Major fab / capex disruption,trigger
public_salience,High-credibility media pressure,trigger
financing_ready,Financing vehicle ready,condition
feasibility_path,Industry feasibility path exists,condition
```

**Trigger** = crisis / catalyst (what forces action).
**Condition** = system readiness / infrastructure (what lets action stick).

### `historical_events.csv`

Historical “important events” we use as training examples.

Required columns:

* `event_id`
* `event_name`
* `target_score`

`target_score` is in `[0..1]` and means:

* How close this historical event is to the *kind* of scenario we're trying to forecast.
* 1.0 = "this is basically the thing we're afraid / hoping to detect"
* 0.2 = "lots of noise / panic but no real durable regime"

Example:

```csv
event_id,event_name,target_score
montreal1987,Montreal Protocol,0.95
covax2020,COVAX vaccine allocation,0.6
gpu_coord2023,US-EU-JP GPU export controls,0.8
media_spike_2024,Media panic no policy,0.2
```

During training we convert this into a binary label:

```text
is_success = 1 if target_score >= success_threshold (default 0.7)
is_success = 0 otherwise
```

Think of `is_success=1` as "the real pattern we're trying to detect."

### `historical_indicators.csv`

Which indicators were actually present before each historical event.

Required columns:

* `event_id`
* `event_name`
* `indicator_id`
* `severity`

`severity` is typically {0,1,2}:

* 0 = basically not present
* 1 = present / moderate
* 2 = very present / coordinated / multi-actor

Example:

```csv
event_id,event_name,indicator_id,severity
montreal1987,Montreal Protocol,public_salience,2
montreal1987,Montreal Protocol,financing_ready,1
montreal1987,Montreal Protocol,feasibility_path,2

covax2020,COVAX vaccine allocation,financing_ready,2
covax2020,COVAX vaccine allocation,feasibility_path,2
covax2020,COVAX vaccine allocation,public_salience,1

gpu_coord2023,US-EU-JP GPU export controls,gpu_controls,2
gpu_coord2023,US-EU-JP GPU export controls,fab_disruption,1
gpu_coord2023,US-EU-JP GPU export controls,public_salience,1

media_spike_2024,Media panic no policy,public_salience,2
media_spike_2024,Media panic no policy,financing_ready,0
```

If an (event_id, indicator_id) pair is missing completely, we assume severity = 0.

---

## 3. Step 1 — Train indicator strength

Script: `bayes_forecaster.py`

What it does:

1. Loads the 3 CSVs above.
2. Labels each historical event as "successful" (is_success=1) or "not" (is_success=0), based on `target_score >= success_threshold`.
3. For each indicator:

   * Compute how often it appears in successful events
     ( P(S_i | H) )
   * Compute how often it appears in non-successful events
     ( P(S_i | ¬H) )
   * Use Laplace smoothing so we don't explode on zeros.
4. Compute a likelihood ratio:
   [
   k_i = \frac{P(S_i \mid H)}{P(S_i \mid \neg H)}
   ]

Intuition:

* If `k_i` >> 1, this indicator is strongly associated with real coordination / real regime formation.
* If `k_i` ~ 1, it's just background noise (it happens in both serious and unserious cases).
* If `k_i` < 1, it's actually more common in false alarms than in real regimes.

The script writes these results to:
`data/indicator_likelihood_ratios.csv`

That file will look like:

```csv
indicator_id,indicator_name,category,p_given_H,p_given_notH,k_ratio
gpu_controls,GPU/export-control changes,trigger,0.78,0.12,6.50
fab_disruption,Major fab / capex disruption,trigger,0.55,0.18,3.06
public_salience,High-credibility media pressure,trigger,0.82,0.60,1.37
financing_ready,Financing vehicle ready,condition,0.67,0.40,1.68
feasibility_path,Industry feasibility path exists,condition,0.74,0.44,1.68
```

This file is the learned "signal strength" of each indicator.

### How to run training

1. Make sure `data/` contains:

   * `list_of_indicators.csv`
   * `historical_events.csv`
   * `historical_indicators.csv`

2. Run:

```bash
python bayes_forecaster.py
```

3. You should now have:

   * `data/indicator_likelihood_ratios.csv`
   * plus a printed table in console

---

## 4. Step 2 — Forecast for a new situation

Script: `estimate_probability.py` -- NOT HERE YET

Goal: take what’s happening **right now** (which indicators are active and how strong), and produce a probability that we are in / entering the target scenario.

How it works:

1. Load `indicator_likelihood_ratios.csv` to get each indicator’s `k_ratio`.
2. Choose a baseline `prior_prob`.
   Example: "Before looking at current signals, I think the scenario had a 20% chance in the next 12-18 months."
3. Describe the situation now as a dict of `{indicator_id: severity_now}`.
4. Do a Bayesian update:

We treat each active indicator as evidence and update odds:

* Convert prior probability ( P(H) ) to odds:
  [
  \text{odds}_\text{prior} = \frac{P(H)}{1 - P(H)}
  ]
* For each indicator i with severity > 0:

  * multiply odds by ( k_i^{\text{severity}} )
* Convert odds back to probability:
  [
  P(H \mid \text{signals}) = \frac{\text{odds}*\text{post}}{1 + \text{odds}*\text{post}}
  ]

So strong signals (high `k_i`) at high severity push probability up fast.

### Example usage

```python
from estimate_probability import load_indicator_ratios, posterior_probability

# load learned ratios
k_map, ratios_df = load_indicator_ratios("data/indicator_likelihood_ratios.csv")

# prior belief before looking at current signals
prior_prob = 0.2  # 20%

# what we are observing right now
current_signals = {
    "gpu_controls": 2,        # coordinated export controls in multiple jurisdictions
    "financing_ready": 1,     # funding / compensation mechanism is already in place
    "public_salience": 0,     # not a huge media panic at the moment
    "fab_disruption": 0,      # fabs are not offline
    "feasibility_path": 1     # industry has a viable path to comply without collapse
}

posterior, details = posterior_probability(
    prior_prob=prior_prob,
    active_indicators=current_signals,
    k_map=k_map
)

print("Posterior probability:", posterior)
print("Breakdown:")
print(details)
```

Typical output:

* Posterior probability (e.g. `0.94` → 94%)
* A breakdown table that shows, per indicator:

  * severity_now
  * k_ratio
  * how much it multiplied the odds

You can paste that breakdown directly into a briefing slide.

---

## 5. Mental model for stakeholders

How to explain this to non-technical people:

1. We looked at real historical coordination events.
2. For each “signal” (indicator), we measured: does this signal usually show up before serious, enforced, durable coordination, or does it mostly appear in empty talk?
3. From that, we learned how strong each signal is as early warning.
4. Now, when we see a set of signals happening at once, we update our estimate of how likely we are to be entering that regime.
5. The result is not "a vibe"; it's "given what historically preceded real shifts, this pattern is X% likely to be the real thing now."

---

## 6. TL;DR

* Put CSVs in `data/`.
* Run `bayes_forecaster.py` → this learns indicator strengths and generates `indicator_likelihood_ratios.csv`.
* Use `estimate_probability.py` to plug in "what’s happening now" and get:

  * posterior probability that we're entering the scenario,
  * contribution of each signal.

This is already usable for decision support / briefings.
