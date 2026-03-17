import json
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "best_model.joblib"
META_PATH = ARTIFACT_DIR / "best_model_meta.json"
DATA_PATH = ROOT_DIR / "HRDataset_v14.csv"

COLS_TO_REMOVE = [
    "EmpID",
    "Employee_Name",
    "Zip",
    "ManagerID",
    "EmpStatusID",
    "DateofTermination",
    "TermReason",
    "EmploymentStatus",
]
SENSITIVE_FEATURES = ["Sex", "RaceDesc"]

FEATURE_MAP = {
    "department": "Department",
    "position": "Position",
    "performance_score": "PerformanceScore",
    "salary": "Salary",
    "absences": "Absences",
    "emp_satisfaction": "EmpSatisfaction",
    "engagement_survey": "EngagementSurvey",
    "days_late_last30": "DaysLateLast30",
    "special_projects_count": "SpecialProjectsCount",
    "recruitment_source": "RecruitmentSource",
    "state": "State",
    "marital_desc": "MaritalDesc",
    "citizen_desc": "CitizenDesc",
    "hispanic_latino": "HispanicLatino",
    "dob": "DOB",
    "date_of_hire": "DateofHire",
    "last_performance_review_date": "LastPerformanceReview_Date",
}

RECOMMENDATION_MAP = {
    "Absences": "Mettre en place un plan de suivi de presence et verifier la charge de travail.",
    "EmpSatisfaction": "Prevoir un point RH/manager pour ameliorer les irritants au poste.",
    "EngagementSurvey": "Definir un plan d'engagement avec feedback regulier et reconnaissance.",
    "Salary": "Revoir le positionnement salarial et la trajectoire d'evolution.",
    "DaysLateLast30": "Identifier la cause des retards et proposer un ajustement organisationnel.",
    "SpecialProjectsCount": "Donner davantage d'opportunites de projets transverses valorisants.",
    "PerformanceScore": "Mettre en place un accompagnement cible avec objectifs progressifs.",
    "Department": "Analyser les signaux d'alerte au niveau de l'equipe/manager.",
    "Position": "Revoir l'adequation poste-competences et les opportunites d'evolution.",
    "RecruitmentSource": "Renforcer l'onboarding et le mentorat pour ce profil d'entree.",
}


class ArtifactsMissingError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def load_bundle() -> Tuple[Any, Dict[str, Any]]:
    if not MODEL_PATH.exists() or not META_PATH.exists():
        raise ArtifactsMissingError(
            "Artifacts manquants. Lance d'abord la cellule SAVE_BEST_MODEL_ARTIFACTS dans code.ipynb."
        )
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    return model, meta


def _risk_level(score: float) -> str:
    if score >= 0.65:
        return "High"
    if score >= 0.4:
        return "Medium"
    return "Low"


def _safe_float(value: Any) -> Any:
    try:
        if value is None or value == "":
            return value
        return float(value)
    except Exception:
        return value


def _normalize_profile(base_features: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    defaults = dict(meta.get("defaults", {}))
    profile = defaults.copy()

    incoming = {str(k): v for k, v in base_features.items() if v is not None}
    for raw_key, raw_value in incoming.items():
        key_lower = raw_key.lower()
        if raw_key in profile:
            profile[raw_key] = raw_value
            continue
        mapped = FEATURE_MAP.get(key_lower)
        if mapped and mapped in profile:
            profile[mapped] = raw_value

    for feature in meta.get("numeric_features", []):
        if feature in profile:
            profile[feature] = _safe_float(profile[feature])

    ordered = {}
    for col in meta.get("feature_columns", []):
        ordered[col] = profile.get(col, defaults.get(col))
    return ordered


def _predict_probability(model: Any, profile: Dict[str, Any]) -> float:
    frame = pd.DataFrame([profile])
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(frame)[:, 1][0])
    pred = float(model.predict(frame)[0])
    return max(0.0, min(1.0, pred))


def _feature_effects(model: Any, meta: Dict[str, Any], profile: Dict[str, Any], top_n: int = 4):
    current_score = _predict_probability(model, profile)
    baseline = meta.get("baseline", {})
    explainable = [f for f in meta.get("explainable_features", []) if f in profile]

    effects: List[Dict[str, Any]] = []
    for feature in explainable:
        test_profile = profile.copy()
        if feature in baseline:
            test_profile[feature] = baseline[feature]
        changed_score = _predict_probability(model, test_profile)
        delta = current_score - changed_score
        effects.append(
            {
                "feature": feature,
                "value": profile.get(feature),
                "impact": round(float(delta), 4),
            }
        )

    risk_factors = sorted([e for e in effects if e["impact"] > 0], key=lambda x: x["impact"], reverse=True)[:top_n]
    protective_factors = sorted([e for e in effects if e["impact"] < 0], key=lambda x: x["impact"])[:top_n]

    recommendations = []
    for factor in risk_factors:
        text = RECOMMENDATION_MAP.get(factor["feature"])
        if text:
            recommendations.append(text)

    return risk_factors, protective_factors, recommendations


def score_profile(base_features: Dict[str, Any], first_name: str = "Personne", source: str = "manual") -> Dict[str, Any]:
    model, meta = load_bundle()
    profile = _normalize_profile(base_features, meta)
    score = _predict_probability(model, profile)
    risk_factors, protective_factors, recommendations = _feature_effects(model, meta, profile)

    return {
        "id": f"{source}-{uuid.uuid4().hex[:8]}",
        "source": source,
        "first_name": first_name,
        "risk_score": round(score, 4),
        "risk_level": _risk_level(score),
        "top_risk_factors": risk_factors,
        "protective_factors": protective_factors,
        "recommendations": recommendations,
        "base_features": base_features,
        "model_features": profile,
    }


def _prepare_dataset_for_model(df: pd.DataFrame, meta: Dict[str, Any]) -> pd.DataFrame:
    model_df = df.drop(columns=COLS_TO_REMOVE, errors="ignore")
    model_df = model_df.drop(columns=["Termd"] + SENSITIVE_FEATURES, errors="ignore")

    for col in meta.get("feature_columns", []):
        if col not in model_df.columns:
            model_df[col] = meta.get("defaults", {}).get(col)

    return model_df[meta.get("feature_columns", [])]


def score_dataset(top_n: int = 20) -> List[Dict[str, Any]]:
    model, meta = load_bundle()
    raw_df = pd.read_csv(DATA_PATH)
    model_df = _prepare_dataset_for_model(raw_df, meta)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(model_df)[:, 1]
    else:
        probs = model.predict(model_df)

    ranked = []
    for idx, score in enumerate(probs):
        emp_id = int(raw_df.iloc[idx].get("EmpID", idx))
        full_name = str(raw_df.iloc[idx].get("Employee_Name", "Unknown"))
        first_name = full_name.split(" ")[0] if full_name else f"Employee_{emp_id}"

        row_features = {
            "sex": raw_df.iloc[idx].get("Sex"),
            "department": raw_df.iloc[idx].get("Department"),
            "position": raw_df.iloc[idx].get("Position"),
            "performance_score": raw_df.iloc[idx].get("PerformanceScore"),
            "salary": raw_df.iloc[idx].get("Salary"),
            "absences": raw_df.iloc[idx].get("Absences"),
            "emp_satisfaction": raw_df.iloc[idx].get("EmpSatisfaction"),
            "engagement_survey": raw_df.iloc[idx].get("EngagementSurvey"),
            "days_late_last30": raw_df.iloc[idx].get("DaysLateLast30"),
            "special_projects_count": raw_df.iloc[idx].get("SpecialProjectsCount"),
            "recruitment_source": raw_df.iloc[idx].get("RecruitmentSource"),
        }

        ranked.append(
            {
                "id": f"dataset-{emp_id}",
                "source": "dataset",
                "employee_id": emp_id,
                "first_name": first_name,
                "full_name": full_name,
                "risk_score": round(float(score), 4),
                "risk_level": _risk_level(float(score)),
                "base_features": row_features,
            }
        )

    ranked.sort(key=lambda item: item["risk_score"], reverse=True)
    return ranked[:top_n]


def score_dataset_employee(employee_id: int) -> Dict[str, Any]:
    df = pd.read_csv(DATA_PATH)
    row = df[df["EmpID"] == employee_id]
    if row.empty:
        raise ValueError(f"Employe {employee_id} introuvable.")

    record = row.iloc[0]
    first_name = str(record.get("Employee_Name", "Unknown")).split(" ")[0]
    base_features = {
        "sex": record.get("Sex"),
        "department": record.get("Department"),
        "position": record.get("Position"),
        "performance_score": record.get("PerformanceScore"),
        "salary": record.get("Salary"),
        "absences": record.get("Absences"),
        "emp_satisfaction": record.get("EmpSatisfaction"),
        "engagement_survey": record.get("EngagementSurvey"),
        "days_late_last30": record.get("DaysLateLast30"),
        "special_projects_count": record.get("SpecialProjectsCount"),
        "recruitment_source": record.get("RecruitmentSource"),
        "state": record.get("State"),
        "marital_desc": record.get("MaritalDesc"),
        "citizen_desc": record.get("CitizenDesc"),
        "hispanic_latino": record.get("HispanicLatino"),
        "dob": record.get("DOB"),
        "date_of_hire": record.get("DateofHire"),
        "last_performance_review_date": record.get("LastPerformanceReview_Date"),
    }
    scored = score_profile(base_features=base_features, first_name=first_name, source="dataset")
    scored["id"] = f"dataset-{employee_id}"
    scored["employee_id"] = employee_id
    scored["full_name"] = str(record.get("Employee_Name", first_name))
    return scored


def generate_personas(count: int = 5) -> List[Dict[str, Any]]:
    df = pd.read_csv(DATA_PATH)
    sampled = df.sample(n=min(count, len(df)), random_state=42)
    personas = []
    for i, (_, row) in enumerate(sampled.iterrows(), start=1):
        base_features = {
            "department": row.get("Department"),
            "position": row.get("Position"),
            "performance_score": row.get("PerformanceScore"),
            "salary": row.get("Salary"),
            "absences": row.get("Absences"),
            "emp_satisfaction": row.get("EmpSatisfaction"),
            "engagement_survey": row.get("EngagementSurvey"),
            "days_late_last30": row.get("DaysLateLast30"),
            "special_projects_count": row.get("SpecialProjectsCount"),
            "recruitment_source": row.get("RecruitmentSource"),
            "state": row.get("State"),
            "marital_desc": row.get("MaritalDesc"),
            "citizen_desc": row.get("CitizenDesc"),
            "hispanic_latino": row.get("HispanicLatino"),
            "dob": row.get("DOB"),
            "date_of_hire": row.get("DateofHire"),
            "last_performance_review_date": row.get("LastPerformanceReview_Date"),
        }
        personas.append(score_profile(base_features, first_name=f"Persona_{i}", source="persona"))
    return personas


def llm_payload(person: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "employee_id": person.get("employee_id", person.get("id", "manual")),
        "risk_score": person.get("risk_score"),
        "risk_level": person.get("risk_level"),
        "top_risk_factors": person.get("top_risk_factors", []),
        "protective_factors": person.get("protective_factors", []),
    }


