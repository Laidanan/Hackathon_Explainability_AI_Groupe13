import os
import sys

# chemin vers le dossier du modèle de ton coéquipier
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
predictive_path = os.path.join(project_root, "predictive_model")

if predictive_path not in sys.path:
    sys.path.append(predictive_path)

from model_service import predict_employee


def get_prediction(employee_id: int) -> dict:
    return predict_employee(employee_id)