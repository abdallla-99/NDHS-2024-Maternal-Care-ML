# NDHS 2024 Maternal Care Machine Learning

This repository contains code and non-disclosive supporting outputs for the analysis of dropout from antenatal care, facility delivery, and maternal postnatal care using the 2024 Nigeria Demographic and Health Survey (NDHS).

## Repository contents

- `code/`: scripts for outcome reconstruction, model comparison, and feature-importance/SHAP analysis.
- `notebooks/`: corrected analysis notebook.
- `results/`: aggregate model-comparison and feature-importance summaries.
- `requirements.txt`: Python dependencies.

## Data access

The analysis uses the 2024 Nigeria Demographic and Health Survey (NDHS) Children’s Recode. Individual-level DHS microdata are not included in this repository. Researchers must request and obtain access directly from The DHS Program and comply with its terms of use.

## Outcomes

The analysis distinguishes:
- Antenatal care dropout (ANC)
- Facility-delivery dropout
- Maternal postnatal-care (PNC) dropout

Maternal PNC is based on the mother's postnatal check indicators (M62 and M66). The child postnatal check variable (M70) is not used as the maternal PNC outcome.

## Reproducibility

1. Obtain authorized access to the relevant 2024 NDHS recode file from The DHS Program.
2. Install the packages listed in `requirements.txt`.
3. Review the scripts and notebook for the expected input filename and data path before execution.
4. Run the notebook and scripts in the documented order.

The analysis scripts and notebook may require adaptation of local file paths. The results folder contains aggregate summaries; it does not contain the restricted microdata.

## License

The license for the authors' code will be stated after all relevant authors have approved the chosen open-source license. No license in this repository grants rights to DHS microdata or third-party materials.

## Citation

A formal citation and Zenodo DOI will be added after the archived release is created.
