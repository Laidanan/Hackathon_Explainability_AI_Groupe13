def format_factors(factors):
    if not factors:
        return "- Aucun facteur fourni"

    lines = []
    for f in factors:
        feature = f.get("feature", "facteur inconnu")
        value = f.get("value", "non précisé")
        impact = f.get("impact")

        if impact is not None:
            lines.append(f"- {feature} : {value} (impact : {impact})")
        else:
            lines.append(f"- {feature} : {value}")

    return "\n".join(lines)

def build_prompt(data: dict) -> str:
    prompt = f"""
Tu es un assistant RH spécialisé dans l’explication des scores de risque de départ.

Règles :
- Ne jamais affirmer avec certitude qu’un employé va quitter l’entreprise.
- Parler en termes de risque ou de probabilité.
- Ne pas inventer de causes absentes des données.
- Ne pas utiliser d’attributs sensibles.
- Produire une réponse claire et professionnelle.

Données :
- ID employé : {data.get('employee_id')}
- Score de risque : {data.get('risk_score')}
- Niveau de risque : {data.get('risk_level')}

Facteurs qui augmentent le risque :
{format_factors(data.get('top_risk_factors', []))}

Facteurs qui réduisent le risque :
{format_factors(data.get('protective_factors', []))}

Tâche :
Rédige un rapport structuré avec :
1. Synthèse
2. Facteurs principaux
3. Actions recommandées
"""
    return prompt