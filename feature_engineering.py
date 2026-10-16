"""
Semantic Feature Engineering Module

This module prepares engineered biomedical features from raw ATTICA variables.
Aggregation is not done here. Each function returns a configuration dictionary
used by the generic aggregation engine.

This version is compatible with the pre-encoding completeness workflow: raw
categorical labels are mapped explicitly inside the relevant domain functions.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from utils.preprocessing import minmax_normalize


# =============================================================================
# SHARED HELPERS
# =============================================================================

def _normalise_text(series):
    """Return lower-cased, stripped text while preserving missing values."""
    return (
        series.astype("string")
        .str.strip()
        .str.casefold()
        .str.replace(r"\s+", " ", regex=True)
    )


def _to_numeric(series, missing_codes=None):
    """
    Convert a series to numeric.

    Missing-code replacement is intentionally column-specific. Values such as
    9 and 99 are valid for several ATTICA measurements and must not be removed
    globally.
    """
    numeric = pd.to_numeric(series, errors="coerce")

    if missing_codes:
        numeric = numeric.replace(list(missing_codes), np.nan)

    return numeric.astype("float64")


def _binary_from_labels(
    series,
    positive_labels,
    negative_labels,
):
    """
    Convert existing 0/1 values or recognised labels into a binary indicator.

    Unrecognised values remain NaN so that the domain-specific imputation rule
    remains explicit rather than silently treating unknown values as absence.
    """
    result = pd.Series(np.nan, index=series.index, dtype="float64")

    numeric = pd.to_numeric(series, errors="coerce")
    numeric_mask = numeric.isin([0, 1])
    result.loc[numeric_mask] = numeric.loc[numeric_mask]

    text = _normalise_text(series)
    positive = {str(value).strip().casefold() for value in positive_labels}
    negative = {str(value).strip().casefold() for value in negative_labels}

    result.loc[text.isin(positive)] = 1.0
    result.loc[text.isin(negative)] = 0.0

    return result


def _fill_median(series):
    """Median-impute a numeric series and fail if it has no observed values."""
    median = series.median()

    if pd.isna(median):
        raise ValueError(f"Cannot median-impute all-missing series: {series.name}")

    return series.fillna(median)


def _fill_mode(series, fallback=0.0):
    """Mode-impute a series with an explicit fallback."""
    mode = series.mode(dropna=True)
    fill_value = mode.iloc[0] if len(mode) else fallback
    return series.fillna(fill_value)


def _validate_aggregation_inputs(out, variables, block_name):
    """Fail fast when an aggregation component is missing, empty, or constant."""
    for variable in variables:
        if variable not in out.columns:
            raise ValueError(f"Missing {block_name} component: {variable}")

        if out[variable].notna().sum() == 0:
            raise ValueError(f"{block_name}.{variable} is entirely missing.")

        if out[variable].nunique(dropna=True) <= 1:
            distribution = out[variable].value_counts(dropna=False).to_dict()
            raise ValueError(
                f"{block_name}.{variable} is constant. Distribution: {distribution}"
            )


# =============================================================================
# LEVEL 1: NON-MODIFIABLE RISK FACTORS
# =============================================================================

def engineer_level1_features(df):
    out = pd.DataFrame(index=df.index)

    if "age" in df.columns:
        out["Age"] = _fill_median(_to_numeric(df["age"]))

        out["AgeGroup"] = pd.cut(
            out["Age"],
            bins=[0, 45, 55, 65, 120],
            labels=[0, 1, 2, 3],
            right=False,
        ).astype("Int64")

        out["Age_Norm"] = minmax_normalize(out["Age"])

    if "sex" in df.columns:
        sex = _binary_from_labels(
            df["sex"],
            positive_labels={"female"},
            negative_labels={"male"},
        )
        out["Sex_Female"] = _fill_mode(sex).astype(int)
        out["sex_trace"] = df["sex"]

    family_specs = {
        "FamHistoryCVD": (
            {"yes"},
            {"no"},
        ),
        "FamHistoryHTN": (
            {"ναι", "yes"},
            {"όχι", "no"},
        ),
        "FamHistoryDM": (
            {"ναι", "yes"},
            {"όχι", "no"},
        ),
        "FamHistoryHCHOL": (
            {"yes"},
            {"no", "no, or don't know"},
        ),
    }

    family_columns = []

    for column, (positive, negative) in family_specs.items():
        if column not in df.columns:
            continue

        out[column] = _binary_from_labels(
            df[column],
            positive_labels=positive,
            negative_labels=negative,
        )
        family_columns.append(column)

    if family_columns:
        family_matrix = out[family_columns]

        # Use the available family-history responses for each participant.
        # Participants with no observed family-history fields remain missing and
        # are median-imputed later by the generic aggregation engine.
        out["FamilyHistory_Count"] = family_matrix.sum(axis=1, min_count=1)
        out["AnyFamilyHistory"] = family_matrix.max(axis=1, skipna=True)
        out["FamilyHistoryRisk_Norm"] = family_matrix.mean(axis=1, skipna=True)

    variables = [
        variable
        for variable in ["Age_Norm", "FamilyHistoryRisk_Norm"]
        if variable in out.columns
    ]

    _validate_aggregation_inputs(out, variables, "Demographics")

    return {
        "level_name": "Level 1: Non-modifiable factors",
        "block_name": "Demographics",
        "features": out,
        "variables": variables,
        "risk_direction": {variable: "positive" for variable in variables},
        "imputation": "median",
        "score_prefix": "NonModifiableRisk",
        "score_column": "NonModifiableRiskScore_Norm",
    }


# =============================================================================
# LEVEL 2: SOCIOECONOMIC STATUS
# =============================================================================

def engineer_level2_features(df):
    out = pd.DataFrame(index=df.index)

    if "Education" in df.columns:
        # Nine years is a legitimate education value, so no global special-code
        # replacement is used here.
        education = _to_numeric(df["Education"])
        education = education.mask(education < 0)
        out["Education"] = _fill_median(education)

    if "financia" in df.columns:
        out["financia_trace"] = df["financia"]
        out["financia"] = _normalise_text(df["financia"]).map(
            {
                "bad": 0.0,
                "poor": 1.0,
                "good": 2.0,
                "very good": 3.0,
            }
        )
        out["financia"] = _fill_median(out["financia"])

    if "SES" in df.columns:
        out["SES_trace"] = df["SES"]
        out["SES"] = _normalise_text(df["SES"]).map(
            {
                "low": 0.0,
                "middle": 1.0,
                "high": 2.0,
            }
        )
        out["SES"] = _fill_median(out["SES"])

    if "Education" in out.columns:
        out["Education_binary"] = (out["Education"] >= 12).astype(int)

    if "MarritalStatus" in df.columns:
        out["MarritalStatus_trace"] = df["MarritalStatus"]

    if {"financia", "SES"}.issubset(out.columns):
        out["SES_financia_binary"] = (
            (out["financia"] >= out["financia"].median())
            & (out["SES"] >= out["SES"].median())
        ).astype(int)

    if {"Education_binary", "SES_financia_binary"}.issubset(out.columns):
        out["Strong_SES_indicator"] = (
            out["Education_binary"].eq(1)
            & out["SES_financia_binary"].eq(1)
        ).astype(int)

    variables = [
        variable
        for variable in ["Education", "financia", "SES"]
        if variable in out.columns
    ]

    _validate_aggregation_inputs(out, variables, "SES")

    # Higher education, finances, and SES are protective. The generic
    # aggregation engine reverses these variables so that a higher final domain
    # score consistently represents greater socioeconomic risk.
    return {
        "level_name": "Level 2: Socioeconomic status",
        "block_name": "SES",
        "features": out,
        "variables": variables,
        "risk_direction": {variable: "negative" for variable in variables},
        "imputation": "median",
        "score_prefix": "SES_Composite",
        "score_column": "SES_CompositeScore_Norm",
    }


# =============================================================================
# LEVEL 3: BEHAVIORAL RISK
# =============================================================================

def engineer_level3_features(df):
    """
    Engineer the Level 3 Behavioral domain.

    Final aggregation components:
    - PhysicalActivityRisk
    - SmokingRisk_Norm

    Alcohol_daily is excluded because its completeness was 16.04%.
    """
    out = pd.DataFrame(index=df.index)

    if "PhysAct" not in df.columns:
        raise ValueError("PhysAct is required to construct PhysicalActivityRisk.")

    out["PhysAct_trace"] = df["PhysAct"]
    out["PhysicalActivityRisk"] = _normalise_text(df["PhysAct"]).map(
        {
            "sedentary": 1.0,
            "physically active": 0.0,
        }
    )

    unexpected_activity = (
        df.loc[
            out["PhysicalActivityRisk"].isna() & df["PhysAct"].notna(),
            "PhysAct",
        ]
        .drop_duplicates()
        .tolist()
    )

    if unexpected_activity:
        raise ValueError(f"Unrecognised PhysAct values: {unexpected_activity}")

    out["PhysicalActivityRisk"] = _fill_median(out["PhysicalActivityRisk"])

    if "PhysAct_level" in df.columns:
        out["PhysAct_level_trace"] = df["PhysAct_level"]

    smoking_status = pd.Series(np.nan, index=df.index, dtype="float64")

    if "smoking_ever" in df.columns:
        out["smoking_ever"] = _binary_from_labels(
            df["smoking_ever"],
            positive_labels={"yes"},
            negative_labels={"no"},
        )
        smoking_status = out["smoking_ever"].copy()

    if "Smoking" in df.columns:
        out["Smoking"] = _binary_from_labels(
            df["Smoking"],
            positive_labels={"yes"},
            negative_labels={"no"},
        )

    smoking_exposure_variables = [
        column
        for column in ["cig_day", "pck_years", "year_smo", "typcigar"]
        if column in df.columns
    ]

    for column in smoking_exposure_variables:
        out[column] = _to_numeric(df[column])

    if "pck_years" in out.columns:
        smoking_status = smoking_status.mask(
            smoking_status.isna() & out["pck_years"].gt(0),
            1.0,
        )

    if "Smoking" in out.columns:
        smoking_status = smoking_status.combine_first(out["Smoking"])

    if smoking_status.notna().sum() == 0:
        raise ValueError("Smoking status could not be derived.")

    smoking_status = _fill_mode(smoking_status).clip(0, 1)
    out["SmokingEver"] = smoking_status.astype(int)

    never_smoker = out["SmokingEver"].eq(0)
    ever_smoker = out["SmokingEver"].eq(1)

    for column in smoking_exposure_variables:
        out.loc[never_smoker, column] = 0.0

        median_ever = out.loc[ever_smoker, column].median()
        if pd.notna(median_ever):
            out.loc[ever_smoker, column] = (
                out.loc[ever_smoker, column].fillna(median_ever)
            )

        out[column] = out[column].fillna(0.0)

    if "pck_years" in out.columns:
        out["SmokingRisk"] = pd.cut(
            out["pck_years"],
            bins=[-np.inf, 0, 100, 500, np.inf],
            labels=[0, 1, 2, 3],
            include_lowest=True,
        ).astype(int)

        out["SmokingRisk_Norm"] = out["SmokingRisk"].astype(float) / 3.0
    else:
        out["SmokingRisk_Norm"] = out["SmokingEver"].astype(float)

    variables = ["PhysicalActivityRisk", "SmokingRisk_Norm"]
    _validate_aggregation_inputs(out, variables, "Behavioral")

    return {
        "level_name": "Level 3: Behavioral risk",
        "block_name": "Behavioral",
        "features": out,
        "variables": variables,
        "risk_direction": {variable: "positive" for variable in variables},
        "imputation": "median",
        "score_prefix": "BehavioralRisk",
        "score_column": "BehavioralRiskScore_Norm",
    }


# =============================================================================
# LEVEL 4: OBESITY / ANTHROPOMETRY
# =============================================================================

def engineer_level4_features(df):
    out = pd.DataFrame(index=df.index)

    continuous_columns = [
        "weight",
        "height",
        "BMI",
        "waist",
        "hip",
        "Waist_Height",
        "w_t_hr",
    ]

    for column in continuous_columns:
        if column in df.columns:
            out[column] = _fill_median(_to_numeric(df[column]))

    binary_specs = {
        "WeightLoss20": ({"yes"}, {"no"}),
        "bmi_25": ({"1"}, {"0"}),
        "obese": ({"1"}, {"0"}),
        "Central_obesity": ({"central obesity"}, {"normal"}),
        "wthr_01": ({"1"}, {"0"}),
    }

    for column, (positive, negative) in binary_specs.items():
        if column in df.columns:
            indicator = _binary_from_labels(
                df[column],
                positive_labels=positive,
                negative_labels=negative,
            )
            out[column] = _fill_mode(indicator)

    if "BMI20" in df.columns:
        out["BMI20_followup_raw"] = _to_numeric(df["BMI20"])

    if "BMI" in out.columns:
        out["Overweight_BMI25"] = (out["BMI"] >= 25).astype(int)
        out["Obesity_BMI30"] = (out["BMI"] >= 30).astype(int)

    if "Waist_Height" in out.columns:
        out["High_WHtR_05"] = (out["Waist_Height"] >= 0.5).astype(int)

    if "w_t_hr" in out.columns:
        out["High_WHR_median"] = (
            out["w_t_hr"] >= out["w_t_hr"].median()
        ).astype(int)

    severity_columns = [
        column
        for column in ["BMI", "waist", "Waist_Height", "w_t_hr"]
        if column in out.columns
    ]

    if len(severity_columns) >= 2:
        severity_data = out[severity_columns].copy()
        severity_z = pd.DataFrame(
            StandardScaler().fit_transform(severity_data),
            index=out.index,
            columns=[f"{column}_z" for column in severity_columns],
        )

        out = pd.concat([out, severity_z], axis=1)
        out["ObesitySeverityScore"] = severity_z.mean(axis=1)
        out["ObesitySeverityScore_Norm"] = minmax_normalize(
            out["ObesitySeverityScore"]
        )

    variables = [
        variable
        for variable in [
            "Overweight_BMI25",
            "Obesity_BMI30",
            "High_WHtR_05",
            "High_WHR_median",
            "ObesitySeverityScore_Norm",
        ]
        if variable in out.columns
    ]

    _validate_aggregation_inputs(out, variables, "Obesity")

    return {
        "level_name": "Level 4: Obesity / anthropometry",
        "block_name": "Obesity",
        "features": out,
        "variables": variables,
        "risk_direction": {variable: "positive" for variable in variables},
        "imputation": "median",
        "score_prefix": "ObesityRisk",
        "score_column": "ObesityRiskScore_Norm",
    }


# =============================================================================
# LEVEL 5: METABOLIC DYSFUNCTION
# =============================================================================

def engineer_level5_features(df):
    out = pd.DataFrame(index=df.index)

    if "MetS_IDF" in df.columns:
        out["MetS_IDF"] = _binary_from_labels(
            df["MetS_IDF"],
            positive_labels={"yes"},
            negative_labels={"no"},
        )
        out["MetS_IDF"] = _fill_mode(out["MetS_IDF"])

    if "mets" in df.columns:
        out["mets"] = _fill_median(_to_numeric(df["mets"]))

    for column in ["glucose", "insulin", "homa_r", "TyG_index", "LAP"]:
        if column in df.columns:
            out[column] = _fill_median(_to_numeric(df[column]))

    for column in ["quicki", "homa_b", "TyG"]:
        if column in df.columns:
            out[f"{column}_trace"] = _fill_median(_to_numeric(df[column]))

    variables = [
        variable
        for variable in [
            "MetS_IDF",
            "mets",
            "glucose",
            "insulin",
            "homa_r",
            "TyG_index",
            "LAP",
        ]
        if variable in out.columns
    ]

    _validate_aggregation_inputs(out, variables, "Metabolic")

    return {
        "level_name": "Level 5: Metabolic dysfunction",
        "block_name": "Metabolic",
        "features": out,
        "variables": variables,
        "risk_direction": {variable: "positive" for variable in variables},
        "imputation": "median",
        "score_prefix": "MetabolicRisk",
        "score_column": "MetabolicRiskScore_Norm",
    }


# =============================================================================
# LEVEL 6: BIOMARKERS
# =============================================================================

def engineer_level6_features(df):
    out = pd.DataFrame(index=df.index)

    inflammation_columns = [
        "Il_6",
        "CRP",
        "wbc",
        "amiloide",
        "Fibrinogen",
        "TNF_a",
        "Hcy",
    ]

    lipid_columns = [
        "Cholesterol",
        "LDL",
        "HDL",
        "TGL",
        "ApoA1",
        "ApoB",
        "ox_LDL",
    ]

    renal_columns = ["creatinine", "creat_clearence", "ourea", "uric"]

    trace_columns = [
        "non_HDL",
        "tc_hdl",
        "ldl_hdl",
        "apob_apoa1",
        "LDLapoB",
        "Lpa",
    ]

    main_columns = inflammation_columns + lipid_columns + renal_columns

    for column in main_columns:
        if column in df.columns:
            out[column] = _fill_median(_to_numeric(df[column]))

    for column in trace_columns:
        if column in df.columns:
            trace = _to_numeric(df[column])
            if trace.notna().any():
                trace = _fill_median(trace)
            out[f"{column}_trace"] = trace

    variables = [column for column in main_columns if column in out.columns]
    risk_direction = {variable: "positive" for variable in variables}

    for protective in ["HDL", "ApoA1", "creat_clearence"]:
        if protective in risk_direction:
            risk_direction[protective] = "negative"

    _validate_aggregation_inputs(out, variables, "Biomarkers")

    return {
        "level_name": "Level 6: Biomarkers",
        "block_name": "Biomarkers",
        "features": out,
        "variables": variables,
        "risk_direction": risk_direction,
        "imputation": "median",
        "score_prefix": "BiomarkerRisk",
        "score_column": "BiomarkerRiskScore_Norm",
    }


# =============================================================================
# LEVEL 7: CONDITIONS
# =============================================================================

def engineer_level7_features(df):
    out = pd.DataFrame(index=df.index)

    diabetes_components = []

    if "DM" in df.columns:
        out["DM"] = _binary_from_labels(
            df["DM"],
            positive_labels={"yes", "diabetes", "diabetic"},
            negative_labels={"no"},
        )
        diabetes_components.append(out["DM"])

    # NIDDM is normally removed by the 50% completeness filter, but this keeps
    # the function robust if another analytical sample contains it.
    if "NIDDM" in df.columns:
        out["NIDDM"] = _binary_from_labels(
            df["NIDDM"],
            positive_labels={
                "non-insulin dependent t2dm",
                "insulin dependent t2dm",
                "yes",
            },
            negative_labels={"0", "no"},
        )
        diabetes_components.append(out["NIDDM"])

    if diabetes_components:
        diabetes_matrix = pd.concat(diabetes_components, axis=1)
        out["DiabetesCondition"] = (
            diabetes_matrix.max(axis=1, skipna=True).fillna(0).astype(int)
        )

    hypertension_components = []

    if "HTN" in df.columns:
        out["HTN"] = _binary_from_labels(
            df["HTN"],
            positive_labels={"hypertensive", "hypertension", "yes"},
            negative_labels={"normal", "no"},
        )
        hypertension_components.append(out["HTN"])

    if "HTN_drugs" in df.columns:
        drug_text = _normalise_text(df["HTN_drugs"])
        drug_indicator = pd.Series(np.nan, index=df.index, dtype="float64")

        observed = drug_text.notna()
        untreated = drug_text.eq("untreated")

        drug_indicator.loc[observed & untreated] = 0.0
        drug_indicator.loc[observed & ~untreated] = 1.0

        out["HTN_drugs"] = drug_indicator
        hypertension_components.append(out["HTN_drugs"])

    if "DietHTN" in df.columns:
        out["DietHTN"] = _binary_from_labels(
            df["DietHTN"],
            positive_labels={"ναι", "yes"},
            negative_labels={"όχι", "no"},
        )
        hypertension_components.append(out["DietHTN"])

    for column in ["sbp", "dbp", "pulsepressure"]:
        if column in df.columns:
            out[column] = _fill_median(_to_numeric(df[column]))

    if "sbp" in out.columns:
        hypertension_components.append((out["sbp"] >= 140).astype(float))

    if "dbp" in out.columns:
        hypertension_components.append((out["dbp"] >= 90).astype(float))

    if hypertension_components:
        hypertension_matrix = pd.concat(hypertension_components, axis=1)
        out["HypertensionCondition"] = (
            hypertension_matrix.max(axis=1, skipna=True).fillna(0).astype(int)
        )

    if "HCHOL" in df.columns:
        out["HCHOL"] = _binary_from_labels(
            df["HCHOL"],
            positive_labels={"hypercholesterolemia", "hypercholesterolaemia"},
            negative_labels={"normal"},
        )
        out["HypercholesterolemiaCondition"] = out["HCHOL"].fillna(0).astype(int)

    # HTGL is a triglyceride condition and is therefore retained only as a
    # trace variable rather than merged into hypercholesterolaemia.
    if "HTGL" in df.columns:
        out["Hypertriglyceridemia_trace"] = _binary_from_labels(
            df["HTGL"],
            positive_labels={"1", "yes"},
            negative_labels={"0", "no"},
        )

    if "RenalDis20" in df.columns:
        out["RenalDis20_trace"] = _binary_from_labels(
            df["RenalDis20"],
            positive_labels={"yes"},
            negative_labels={"no"},
        )

    variables = [
        variable
        for variable in [
            "DiabetesCondition",
            "HypertensionCondition",
            "HypercholesterolemiaCondition",
        ]
        if variable in out.columns
    ]

    _validate_aggregation_inputs(out, variables, "Conditions")

    return {
        "level_name": "Level 7: Clinical conditions",
        "block_name": "Conditions",
        "features": out,
        "variables": variables,
        "risk_direction": {variable: "positive" for variable in variables},
        "imputation": "zero",
        "score_prefix": "ConditionLoad",
        "score_column": "ConditionLoadScore_Norm",
    }


# =============================================================================
# LEVEL 8: DIET TREATMENT / EXPOSURE
# =============================================================================

def engineer_level8_features(df):
    out = pd.DataFrame(index=df.index)

    if "MedDietScore" in df.columns:
        out["MedDietScore"] = _fill_median(
            _to_numeric(df["MedDietScore"])
        )

        out["DietAdherenceScore_Norm"] = minmax_normalize(
            out["MedDietScore"]
        )

    # Keep follow-up scores only as trace variables.
    for column in ["MedDietScore5", "MedDietScore10"]:
        if column in df.columns:
            out[f"{column}_trace"] = _to_numeric(
                df[column]
            )

    if "traject_MedDi" in df.columns:
        out["traject_MedDi_trace"] = df[
            "traject_MedDi"
        ]

    variables = [
        variable
        for variable in ["DietAdherenceScore_Norm"]
        if variable in out.columns
    ]

    _validate_aggregation_inputs(
        out,
        variables,
        "Treatment",
    )

    return {
        "level_name": "Level 8: Mediterranean diet adherence",
        "block_name": "Treatment",
        "features": out,
        "variables": variables,

        # "positive" tells the generic aggregation function
        # to preserve the current orientation.
        "risk_direction": {
            variable: "positive"
            for variable in variables
        },

        "imputation": "median",
        "score_prefix": "DietAdherence",
        "score_column": "DietAdherenceScore_Norm",
    }
# =============================================================================
# LEVEL 9: OUTCOMES
# =============================================================================

def engineer_level9_features(df):
    """
    Construct the Level 9 Outcomes block using only the binary
    10-year cardiovascular mortality variable.

    Expected source variable:
        CVD_mortality10

    Output:
        A binary Outcomes feature where:
        1 = cardiovascular mortality occurred
        0 = cardiovascular mortality did not occur

    Missing values follow the declared zero-imputation convention.
    """

    out = pd.DataFrame(index=df.index)

    required_column = "CVD_mortality10"

    if required_column not in df.columns:
        raise KeyError(
            f"Required outcome variable '{required_column}' "
            "was not found in the dataset."
        )

    mortality = _binary_from_labels(
        df[required_column],
        positive_labels={"1", "yes"},
        negative_labels={"0", "no"},
    )

    out[required_column] = mortality.fillna(0).astype(int)

    variables = [required_column]

    _validate_aggregation_inputs(
        out,
        variables,
        "Outcomes",
    )

    return {
        "level_name": "Level 9: Outcomes",
        "block_name": "Outcomes",
        "features": out,
        "variables": variables,
        "risk_direction": {
            required_column: "positive"
        },
        "imputation": "zero",
        "score_prefix": "OutcomeRisk",
        "score_column": "OutcomeRiskScore_Norm",
    }