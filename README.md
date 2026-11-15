# NDHS 2024 Maternal Care Machine Learning

This repository contains the Python scripts, Jupyter notebook, and aggregate supporting results for the study of dropout from antenatal care, facility delivery, and maternal postnatal care in Nigeria using the 2024 Nigeria Demographic and Health Survey (NDHS).

## Repository contents

The following files are available in the root directory:

- `main_analysis.ipynb`: Corrected Jupyter notebook for the analysis.
- `model_comparison.py`: Comparison of eight machine-learning algorithms.
- `maternal_pnc_reconstruction.py`: Construction and audit of the maternal postnatal-care outcome.
- `feature_importance_shap.py`: Feature-importance and SHAP analysis.
- `model_comparison_summary.csv`: Aggregate model-comparison results.
- `shap_importance_summary.csv`: Aggregate SHAP importance results.
- `weighted_permutation_importance.csv`: Weighted permutation-importance results.
- `requirements.txt`: Python package dependencies.
- `.gitignore`: Files excluded from version control.

## Study outcomes

The analysis examines three outcomes:

1. Antenatal care (ANC) dropout.
2. Facility-delivery dropout.
3. Maternal postnatal-care (PNC) dropout.

The maternal PNC outcome is based on maternal postnatal-check indicators M62 and M66. The child postnatal-check indicator M70 is not used to define maternal PNC.

## Machine-learning models

Eight algorithms are compared:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting
5. XGBoost
6. LightGBM
7. AdaBoost
8. Naive Bayes

Model evaluation uses repeated primary-sampling-unit (PSU)-grouped holdout partitions and DHS-weighted performance metrics.

## Data access

The study uses the 2024 Nigeria Demographic and Health Survey (NDHS) Children’s Recode.

Individual-level DHS microdata are not included in this repository. Researchers must request access directly from The DHS Program and comply with its registration and data-use requirements:

https://dhsprogram.com/

## Running the analysis

1. Obtain authorized access to the relevant 2024 NDHS recode data.
2. Install the Python dependencies listed in `requirements.txt`.
3. Review the notebook and scripts for the expected input filename and local data path.
4. Run the analysis using Jupyter Notebook or Google Colab.

The scripts may require adjustment of local file paths before execution.

## Code availability

The author-generated analysis code is publicly available through this GitHub repository. The archived release is available on Zenodo.

## License

The code in this repository is licensed under the Apache License 2.0. Third-party materials and restricted DHS microdata are not covered by this license.

## Citation and DOI

Please cite the archived software release when using this code.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24247128.svg)](https://doi.org/10.5281/zenodo.24247128)

https://doi.org/10.5281/zenodo.24247128
