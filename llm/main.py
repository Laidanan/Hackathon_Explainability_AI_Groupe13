import json
import os
from explanation_generator import generate_explanation
from predictor_client import get_prediction


employee_id = int(input("Entrez l'ID de l'employé : "))

prediction_data = get_prediction(employee_id)

explanation = generate_explanation(prediction_data)


print("\n===== EXPLICATION RH =====\n")
print(explanation)