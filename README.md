# Basketball Shot Value Prediction

A machine learning project focused on estimating the expected value of basketball shots using play-by-play and shot-level tracking data. The goal is to move beyond simple made/missed shot prediction and instead quantify the true value of a shot attempt based on its context, location, and surrounding game conditions.

## Overview

Traditional basketball statistics often evaluate shots only by whether they are made or missed. However, shot quality depends on many factors including shot distance, defender proximity, court location, game situation, and player positioning.

This project develops and evaluates machine learning models that predict the expected value of a shot attempt. By estimating the probability of different scoring outcomes, the model provides a more nuanced measure of offensive decision-making and shot selection.

## Dataset

The project uses shot-level basketball data containing:

* Shot location coordinates
* Shot distance
* Shot type
* Game context variables
* Offensive and defensive positioning information
* Possession-level metadata
* Shot outcome labels

Data preprocessing included:

* Removal of invalid and mislabeled observations
* Feature standardization and cleaning
* Missing value handling
* Construction of engineered contextual features

## Feature Engineering

In addition to the raw dataset features, several higher-level basketball features were engineered to better capture shot context.

Examples include:

* Distance-based shot characteristics
* Relative offensive and defensive positioning
* Spatial court relationships
* Contextual possession features
* Interaction features between location and game state variables

These engineered features were designed to represent basketball concepts that are not directly observable from the raw data alone.

## Models

The following models were explored and compared:

### Baseline Logistic Regression

A simple interpretable baseline using the original dataset features.

### Enhanced Logistic Regression

An improved version incorporating engineered basketball-context features.

### Tree-Based Models

Ensemble methods capable of capturing nonlinear relationships between shot context and expected value.

### Neural Network Models

Multi-layer perceptrons used to learn complex feature interactions and spatial relationships.

## Evaluation

Model performance was evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Calibration analysis
* Cross-validation

Comparisons were made between baseline and enhanced models to measure the contribution of feature engineering.

## Results

Key findings include:

* Engineered contextual features significantly improved predictive performance.
* Spatial and relational features provided meaningful information beyond raw shot coordinates.
* More complex models captured nonlinear shot-value relationships but required careful tuning to avoid overfitting.
* Expected shot value provides a richer evaluation metric than simple field goal percentage.

## Visualizations

The project includes visual analyses such as:

* Feature importance rankings
* Shot value heatmaps
* Model calibration plots
* Performance comparison charts
* Spatial shot distributions

These visualizations help interpret model behavior and identify patterns in offensive decision-making.

## Repository Structure

```text
Shot-Value-Machine-Learning/
├── data/
│   ├── raw/
│   │   └── shot_logs.csv
│   └── processed/
│       ├── shot_logs_cleaned.csv
│       └── shot_logs_cleaned_engineered.csv
│
├── models/
│   ├── xgboost_engineered_shot_model.joblib
│   ├── shap_beeswarm.png
│   ├── shap_dependence_grid.png
│   ├── shap_interaction_heatmap.png
│   └── shot_uncertainty_calibration.png
│
├── notebooks/
│   ├── 01_data_import_cleaning.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_logistic_regression_baseline.ipynb
│   ├── 05_tree_models.ipynb
│   ├── 06_logistic_regression_engineered.ipynb
│   ├── 07_tree_models_engineered.ipynb
│   ├── 08_shot_data_visualizations.ipynb
│   └── 09_shot_uncertainty_and_shap.ipynb
│
├── README.md
└── .gitignore
```

## Future Work

Potential extensions include:

* Incorporating player-specific information
* Using player and ball tracking data
* Temporal sequence modeling of possessions
* Deep learning approaches for spatial shot representations
* Real-time expected shot value estimation

## Authors

Nathaniel Seluga
Michael O'Brien

Harvey Mudd College
Computer Science Department
