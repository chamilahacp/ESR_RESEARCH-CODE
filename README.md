# An Integrated Remote Sensing-Machine Learning Framework for Assessing Plant Air Pollution 1 Tolerance Using Satellite-Drived Air Quality, Reanalysis Meteorological Data

This repository contains the scripts, notebooks, and software information used for the study on predicting Air Pollution Tolerance Index (APTI) of *Pterocarpus indicus* using satellite-derived air pollutant exposure, meteorological variables, and machine-learning models.

## Study overview

The workflow integrates:

- laboratory-derived APTI as the response variable
- satellite-derived air pollutant exposure variables
- meteorological predictors
- a species-specific deciduous-state indicator
- machine-learning models including RF, GBR, XGB, and SVR
- model interpretation using PDP and SHAP
- validation using train–test split and related analyses

## Repository contents

- `scripts/` – Python scripts for model fitting, prediction, PDP, SHAP, and figure generation
- `notebooks/` – Colab/Jupyter notebook version of the workflow
- `data/` – dataset description and access information
- `results/` – exported predictions, metrics, and figures
- `requirements.txt` – Python package requirements

## Input variables

The final predictor set used in the models is:

- N6
- S4
- T4
- P5
- O5
- R4
- H5
- S

The response variable is:

- APTI (`A`)

## Models included

- Random Forest (RF)
- Gradient Boosting Regressor (GBR)
- Extreme Gradient Boosting (XGB)
- Support Vector Regression (SVR)

## Reproducibility

To run the script locally:

1. Clone the repository
2. Install the required packages from `requirements.txt`
3. Place the dataset in the `data/` directory
4. Run:

```bash
python scripts/final_models_pdp_shap.py
