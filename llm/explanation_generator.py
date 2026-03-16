from prompt_builder import build_prompt
from llm_client import call_llm
from feature_dictionary import translate_features

def generate_explanation(employee_data):

    # transformer les variables techniques
    translated_data = translate_features(employee_data)

    # construire le prompt
    prompt = build_prompt(translated_data)

    # appeler le LLM
    response = call_llm(prompt)

    return response