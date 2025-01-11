import json
from generate_llm_email_labels import Prediction

with open("prediction_schema.json", "w") as f:
    json.dump(Prediction.model_json_schema(), f, indent=2)
