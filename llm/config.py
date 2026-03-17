import os
from dotenv import load_dotenv

load_dotenv()

# La clé est récupérée depuis l'environnement, pas écrite en dur
OPENAI_API_KEY = os.getenv("GROQ_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("ERREUR CYBER : La clé GROQ_API_KEY est introuvable dans le fichier .env")