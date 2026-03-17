import json
import os
from explanation_generator import generate_explanation
from predictor_client import get_prediction


employee_id = int(input("Entrez l'ID de l'employé : "))

prediction_data = get_prediction(employee_id)

explanation = generate_explanation(prediction_data)


print("\n===== EXPLICATION RH =====\n")
print(explanation)

try:
    user_input = input("Entrez l'ID de l'employé : ")
    
    # 1. Vérification du format (Type checking)
    employee_id = int(user_input) 
    
    # 2. Vérification de la valeur (Range checking)
    if employee_id < 0 or employee_id > 99999:
        raise ValueError("ID invalide")

except ValueError:
    print("Tentative d'injection ou format invalide détecté. Accès refusé.")
    exit()