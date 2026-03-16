from openai import OpenAI

client = OpenAI(
    api_key="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",
    base_url="https://api.groq.com/openai/v1"
)

prompt = """
Score de risque : 0.78
Facteurs :
- absence d'évolution depuis 4 ans
- satisfaction faible
- salaire inférieur à la moyenne

Explique les raisons possibles du risque de départ
et propose 3 actions RH.
"""

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "system", "content": "Tu es un assistant RH."},
        {"role": "user", "content": prompt}
    ]
)

print(response.choices[0].message.content)