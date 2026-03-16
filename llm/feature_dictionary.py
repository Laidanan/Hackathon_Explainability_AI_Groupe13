feature_labels = {
    "YearsSinceLastPromotion": "absence d’évolution récente",
    "MonthlyIncome": "niveau de rémunération",
    "JobSatisfaction": "niveau de satisfaction au travail",
    "DistanceFromHome": "distance domicile-travail",
    "OverTime": "heures supplémentaires",
    "WorkLifeBalance": "équilibre vie professionnelle / vie personnelle"
}

def translate_feature_name(name: str) -> str:
    return feature_labels.get(name, name)

def translate_features(data: dict) -> dict:
    for factor in data.get("top_risk_factors", []):
        factor["feature"] = translate_feature_name(factor["feature"])

    for factor in data.get("protective_factors", []):
        factor["feature"] = translate_feature_name(factor["feature"])

    return data