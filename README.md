# Hybrid Causal Discovery and Inference — ATTICA Cohort

**Research-code portfolio** · MSc Information and Computer Sciences (Information Systems Engineering), University of Luxembourg · Faten Miqdad, 2026

This repository contains selected **Python notebooks and utility modules** from the Master's thesis *A Hybrid Approach for Mining Causal Dependencies within the ATTICA Cohort Study*. It explores domain-informed causal structure learning and effect estimation for longitudinal cardiovascular research data.

> **Public code, not public health data.** The underlying ATTICA research datasets are **not included**. The code is provided for methodological inspection and requires authorised access to the study data for original analyses. Notebook outputs have been cleared to avoid disclosing participant-level information. No claim of a fully executable public replication package is made.

## Research questions and workflow

- Prepare longitudinal follow-up data and define an outcome and analysis cohort.
- Construct interpretable domain-level representations of demographic, behavioural, metabolic, and clinical variables.
- Examine statistical dependencies and compare unconstrained, time-informed, and expert-informed causal discovery using PC and FCI.
- Study within-domain biomarker dependencies.
- Estimate causal contrasts under stated assumptions, using model-based effect estimation and sensitivity checks.

**Important:** A learned graph does not, by itself, establish causal truth. Findings depend on data quality, selection decisions, temporal assumptions, modelling choices and the availability of relevant confounders.

## Contents

```text
notebooks/
  00_Project_Setup.ipynb
  04_Biomarker_Level_Causal_Discovery.ipynb
  FULL_cohort_exp/
    01_Data_Preparation-full_cohort.ipynb
    02_Semantic_Domain_Engineering-full_cohort.ipynb
    03_Block-Level_Causal_Discovery-full_cohort.ipynb
    05_Causal_Effect_Estimation_full_cohort.ipynb
utils/
  *.py                        # Feature engineering, graphs, discovery, plotting
requirements.txt              # Indicative Python dependencies
DATA_ACCESS.md                # Data and privacy restrictions
```

The primary sequence uses the **full-cohort experiment** notebooks from the supplied project archive, alongside the separate biomarker analysis notebook. The folder structure preserves the original notebooks’ relative-path expectations. Exploratory balanced-subsample experiments were deliberately omitted from this public portfolio; they should not be read as formal population-level sensitivity validation.

## Running the notebooks (authorised researchers only)

1. Set up a Python 3 environment and install the packages in `requirements.txt` (versions may need adjustment for compatibility).
2. Place the original ATTICA workbook, **if you are authorised to use it**, under `data/raw/` with the filename expected by the preparation notebook.
3. For the full-cohort sequence, launch Jupyter **from `notebooks/FULL_cohort_exp/`** and run notebooks `01 → 02 → 03 → 05`. These notebooks use their working directory to resolve relative paths. For the biomarker notebook `04`, launch Jupyter from the **`notebooks/`** directory.
4. The `00_Project_Setup.ipynb` notebook also assumes the Jupyter working directory is `notebooks/`. Generated files under `data/processed/`, `results/`, and `notebooks/FULL_cohort_exp/results/` are excluded by `.gitignore` and must not be committed.

The notebooks are preserved as research artifacts and **have not been retested end-to-end in this cleaned public release** because the original dataset is not shared.

## Methods and tools

Python, NumPy, pandas, scikit-learn, causal-learn (PC/FCI), networkx, statsmodels, DoWhy, and matplotlib. Domain hypotheses are encoded using temporal ordering and clinical background knowledge; they are assumptions, not discoveries guaranteed by the data.

## Research status and attribution

The Master's thesis was completed at the University of Luxembourg in 2026. A related research manuscript is **in preparation**, not claimed here as a published journal paper. Research was conducted with academic supervision and collaboration. This repository does not itself assert that collaborators or the ATTICA data custodians approved public distribution of other project materials.

**Code release note:** No open-source license is included pending confirmation of the appropriate rights and permission for academic project code sharing. Public visibility alone does not grant reuse rights. Please contact the author for licensing requests.
