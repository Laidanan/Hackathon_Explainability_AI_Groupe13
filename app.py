import json
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from predictive_model.model_service import llm_payload, score_dataset, score_dataset_employee, score_profile

ROOT_DIR = Path(__file__).resolve().parent
LLM_DIR = ROOT_DIR / "llm"
if str(LLM_DIR) not in sys.path:
    sys.path.append(str(LLM_DIR))

from llm_client import call_llm  # noqa: E402
from prompt_builder import build_prompt  # noqa: E402

app = FastAPI(title="HR Risk Dashboard", version="1.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PERSONA_DB_PATH = ROOT_DIR / "predictive_model" / "persona_db.json"

DEFAULT_PERSONAS: List[Dict[str, Any]] = [
    {
        "id": "persona-001",
        "name": "Lea Martin",
        "origin": "seed",
        "base_features": {
            "sex": "F",
            "department": "Production",
            "position": "Production Technician I",
            "performance_score": "Fully Meets",
            "salary": 48000,
            "absences": 14,
            "emp_satisfaction": 2,
            "engagement_survey": 3.1,
            "days_late_last30": 3,
            "special_projects_count": 0,
            "recruitment_source": "Indeed",
        },
    },
    {
        "id": "persona-002",
        "name": "Adam Klein",
        "origin": "seed",
        "base_features": {
            "sex": "M",
            "department": "IT/IS",
            "position": "Software Engineer",
            "performance_score": "Exceeds",
            "salary": 82000,
            "absences": 2,
            "emp_satisfaction": 4,
            "engagement_survey": 4.3,
            "days_late_last30": 0,
            "special_projects_count": 5,
            "recruitment_source": "LinkedIn",
        },
    },
    {
        "id": "persona-003",
        "name": "Sara Benali",
        "origin": "seed",
        "base_features": {
            "sex": "F",
            "department": "Sales",
            "position": "Sales Manager",
            "performance_score": "Fully Meets",
            "salary": 62000,
            "absences": 9,
            "emp_satisfaction": 2,
            "engagement_survey": 2.9,
            "days_late_last30": 4,
            "special_projects_count": 1,
            "recruitment_source": "Google Search",
        },
    },
    {
        "id": "persona-004",
        "name": "Noah Dupont",
        "origin": "seed",
        "base_features": {
            "sex": "M",
            "department": "Admin Offices",
            "position": "Administrative Assistant",
            "performance_score": "Fully Meets",
            "salary": 43000,
            "absences": 6,
            "emp_satisfaction": 3,
            "engagement_survey": 3.4,
            "days_late_last30": 1,
            "special_projects_count": 2,
            "recruitment_source": "Employee Referral",
        },
    },
    {
        "id": "persona-005",
        "name": "Mia Rossi",
        "origin": "seed",
        "base_features": {
            "sex": "F",
            "department": "Production",
            "position": "Production Technician II",
            "performance_score": "Needs Improvement",
            "salary": 41000,
            "absences": 18,
            "emp_satisfaction": 1,
            "engagement_survey": 2.1,
            "days_late_last30": 6,
            "special_projects_count": 0,
            "recruitment_source": "CareerBuilder",
        },
    },
]


class ManualEmployeeInput(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    sex: Optional[str] = None
    department: Optional[str] = None
    position: Optional[str] = None
    performance_score: Optional[str] = None
    salary: Optional[float] = None
    absences: Optional[float] = None
    emp_satisfaction: Optional[float] = None
    engagement_survey: Optional[float] = None
    days_late_last30: Optional[float] = None
    special_projects_count: Optional[float] = None
    recruitment_source: Optional[str] = None
    state: Optional[str] = None
    marital_desc: Optional[str] = None
    citizen_desc: Optional[str] = None
    hispanic_latino: Optional[str] = None
    dob: Optional[str] = None
    date_of_hire: Optional[str] = None
    last_performance_review_date: Optional[str] = None


class ChatRequest(BaseModel):
    person_id: str
    question: Optional[str] = None


def _name_of(entry: Dict[str, Any]) -> str:
    if entry.get("name"):
        return str(entry["name"])
    if entry.get("full_name"):
        return str(entry["full_name"])
    if entry.get("first_name"):
        return str(entry["first_name"])
    return "Unknown"


def _load_persona_db() -> List[Dict[str, Any]]:
    PERSONA_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not PERSONA_DB_PATH.exists():
        PERSONA_DB_PATH.write_text(json.dumps(DEFAULT_PERSONAS, ensure_ascii=False, indent=2), encoding="utf-8")
        return [dict(p) for p in DEFAULT_PERSONAS]

    raw = json.loads(PERSONA_DB_PATH.read_text(encoding="utf-8-sig"))
    entries = raw if isinstance(raw, list) else []
    for e in entries:
        if "name" not in e:
            e["name"] = _name_of(e)
    return entries


def _save_persona_db(entries: List[Dict[str, Any]]) -> None:
    PERSONA_DB_PATH.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")


def _risk_level(score: float) -> str:
    if score >= 0.65:
        return "High"
    if score >= 0.4:
        return "Medium"
    return "Low"


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _heuristic_person(entry: Dict[str, Any]) -> Dict[str, Any]:
    base = entry.get("base_features", {})
    absences = min(_num(base.get("absences"), 0.0) / 20.0, 1.0)
    satis = min(max(_num(base.get("emp_satisfaction"), 3.0), 0.0), 5.0) / 5.0
    engage = min(max(_num(base.get("engagement_survey"), 3.0), 0.0), 5.0) / 5.0
    late = min(_num(base.get("days_late_last30"), 0.0) / 10.0, 1.0)
    projects = min(_num(base.get("special_projects_count"), 1.0) / 5.0, 1.0)

    score = 0.2 + (0.25 * absences) + (0.2 * (1 - satis)) + (0.2 * (1 - engage)) + (0.2 * late) + (0.15 * (1 - projects))
    score = max(0.0, min(1.0, score))

    factors = [
        {"feature": "Absences", "value": base.get("absences"), "impact": round(0.25 * absences, 4)},
        {"feature": "EmpSatisfaction", "value": base.get("emp_satisfaction"), "impact": round(0.2 * (1 - satis), 4)},
        {"feature": "EngagementSurvey", "value": base.get("engagement_survey"), "impact": round(0.2 * (1 - engage), 4)},
        {"feature": "DaysLateLast30", "value": base.get("days_late_last30"), "impact": round(0.2 * late, 4)},
    ]
    factors = sorted(factors, key=lambda x: x["impact"], reverse=True)

    return {
        "id": entry.get("id", f"persona-{uuid.uuid4().hex[:8]}"),
        "name": _name_of(entry),
        "source": "persona_db",
        "origin": entry.get("origin", "seed"),
        "risk_score": round(score, 4),
        "risk_level": _risk_level(score),
        "top_risk_factors": factors[:4],
        "protective_factors": [],
        "recommendations": [
            "Plan de suivi RH mensuel.",
            "Ameliorer engagement et satisfaction via objectifs clairs + feedback.",
            "Reduire absences/retards via plan d'accompagnement manager.",
        ],
        "base_features": base,
    }


def _decorate_for_ranking(item: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(item)
    base = out.get("base_features") or {}
    out["name"] = _name_of(item)
    out["sex"] = base.get("sex") or base.get("Sex") or out.get("sex") or "-"
    out["position"] = base.get("position") or base.get("Position") or out.get("position") or "-"
    out["salary"] = base.get("salary") or base.get("Salary") or out.get("salary")
    if out["salary"] is None or out["salary"] == "":
        out["salary"] = "-"
    return out


def _score_persona_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    try:
        scored = score_profile(
            base_features=entry.get("base_features", {}),
            first_name=_name_of(entry).split(" ")[0],
            source="persona_db",
        )
        scored["id"] = entry.get("id", scored.get("id"))
        scored["name"] = _name_of(entry)
        scored["source"] = "persona_db"
        scored["origin"] = entry.get("origin", "seed")
        return _decorate_for_ranking(scored)
    except Exception:
        return _decorate_for_ranking(_heuristic_person(entry))


def _find_persona_entry(person_id: str) -> Optional[Dict[str, Any]]:
    for entry in PERSONA_PROFILES:
        if entry.get("id") == person_id:
            return entry
    return None


def _chat_fallback(person: Dict[str, Any], question: Optional[str] = None) -> str:
    recommendations = person.get("recommendations", [])
    factors = person.get("top_risk_factors", [])
    lines = [
        f"Synthese: risque {person.get('risk_level')} (score={person.get('risk_score')}).",
        "Facteurs de risque principaux:",
    ]
    if factors:
        for factor in factors:
            lines.append(f"- {factor.get('feature')}: valeur={factor.get('value')} impact={factor.get('impact')}")
    else:
        lines.append("- Aucun facteur majeur identifie.")

    lines.append("Actions recommandees:")
    if recommendations:
        for rec in recommendations[:4]:
            lines.append(f"- {rec}")
    else:
        lines.append("- Mettre en place un suivi manager/RH et feedback mensuel.")

    if question:
        lines.append(f"\nQuestion RH: {question}")

    return "\n".join(lines)


PERSONA_PROFILES: List[Dict[str, Any]] = _load_persona_db()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT_DIR / "static" / "index.html")


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/ranking")
def api_ranking(top_n: int = 20) -> Dict[str, Any]:
    warning = None
    try:
        dataset_people = score_dataset(top_n=max(top_n, 100))
        dataset_people = [_decorate_for_ranking(p) for p in dataset_people]
    except Exception as exc:
        dataset_people = []
        warning = f"Modele indisponible pour le dataset: {exc}"

    persona_scored = [_score_persona_entry(p) for p in PERSONA_PROFILES]
    combined = dataset_people + persona_scored
    combined.sort(key=lambda p: p.get("risk_score", 0.0), reverse=True)
    return {"items": combined[:top_n], "warning": warning}


@app.post("/api/manual")
def api_manual_employee(payload: ManualEmployeeInput) -> Dict[str, Any]:
    data = payload.model_dump(exclude_none=True)
    name = data.pop("name")

    new_entry = {
        "id": f"persona-{uuid.uuid4().hex[:8]}",
        "name": name,
        "origin": "manual",
        "base_features": data,
    }
    PERSONA_PROFILES.append(new_entry)
    _save_persona_db(PERSONA_PROFILES)

    return _score_persona_entry(new_entry)


@app.post("/api/chat")
def api_chat(payload: ChatRequest) -> Dict[str, Any]:
    person: Optional[Dict[str, Any]] = None
    person_id = payload.person_id

    if person_id.startswith("dataset-"):
        try:
            employee_id = int(person_id.split("-", 1)[1])
            person = _decorate_for_ranking(score_dataset_employee(employee_id))
        except Exception:
            person = None
    else:
        entry = _find_persona_entry(person_id)
        if entry:
            person = _score_persona_entry(entry)

    if not person:
        raise HTTPException(status_code=404, detail="Personne introuvable.")

    llm_input = llm_payload(person)
    prompt = build_prompt(llm_input)
    if payload.question:
        prompt += (
            "\n\nQuestion RH supplementaire:\n"
            + payload.question
            + "\n\nReponds en expliquant les causes probables et des actions concretes."
        )

    try:
        explanation = call_llm(prompt)
    except Exception:
        explanation = _chat_fallback(person, payload.question)

    return {
        "person_id": person_id,
        "name": person.get("name"),
        "risk_score": person.get("risk_score"),
        "risk_level": person.get("risk_level"),
        "recommendations": person.get("recommendations", []),
        "message": explanation,
    }
