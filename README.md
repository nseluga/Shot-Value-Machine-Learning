# NBA Shot Value

Michael O'Brien and I built a player-agnostic model to estimate the expected point value of an NBA shot attempt, make probability times point value, from shot context: distance, defender proximity, shot clock, game state.

## Headline result

Predictability plateaus hard: **AUC ~0.63–0.64** regardless of model class or feature engineering (tuned XGBoost: accuracy 0.620, AUC 0.6365). What's left over is shooter skill, which we excluded on purpose. The probabilities are still well-calibrated, so multiplying them by point value gives a trustworthy expected-shot-value estimate even with that ceiling.

## Data & method

- 128,069 shots from the 2014–15 NBA season shot-logs dataset (122,203 after cleaning): shot distance, closest-defender distance, shot clock, dribbles, touch time, period, home/away. No x/y court coordinates; location comes from shot zone × distance bin instead.
- Logistic regression and tree models (decision tree, random forest, XGBoost), each run on raw features and on 15 engineered features (`defender_distance_ratio`, `time_pressure_index`, shot-type flags).
- `RandomizedSearchCV` with stratified 3-fold CV on the training split only; random 80/20 train/test split (97,762 / 24,441).
- Full SHAP suite (beeswarm, dependence grid, interaction heatmap) plus a calibration/uncertainty analysis on 6,000 held-out shots.

## Concrete findings

- Expected value is a clean U-shape by distance and defensive pressure: wide-open shots at the rim run ~1.85 EV, contested 20–22 ft shots ~0.62 EV, with a distinct bump at the corner-three distance bin (22–24 ft) and a trough through mid-range at every pressure level.
- SHAP ranks `defender_distance_ratio` as the top feature, ahead of raw shot distance and touch time.
- Engineered features barely move the needle: +0.003 AUC for logistic regression, next to nothing for tree models. We're reporting that as the honest result, not hiding it.

## Repository Structure

```text
Shot-Value-Machine-Learning/
├── data/
│   ├── raw/shot_logs.csv
│   └── processed/shot_logs_cleaned*.csv
├── models/                       # trained model + SHAP/calibration figures
├── notebooks/                    # 01-09, sequential pipeline
├── README.md
└── .gitignore
```

## Future Work

- Player-specific shooting skill (excluded here on purpose)
- Player/ball tracking data for real spatial features
- Temporal sequence modeling of possessions

## Authors

Nathaniel Seluga, Michael O'Brien
Harvey Mudd College, Computer Science Department
