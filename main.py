"""
FastAPI serving layer for the churn-prediction model.

The full-feature Decision Tree is exported once by `train.py` as
`decision_tree.onnx`. At server startup that file is loaded into an
`onnxruntime.InferenceSession` and reused across all requests — no per-request
model load.

The model expects 12 features in this fixed order (set by train.py):

   CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard,
   IsActiveMember, EstimatedSalary, BalancePerProduct,
   Geography_Germany, Geography_Spain, Gender_Male

`BalancePerProduct` and the one-hot dummies for Geography + Gender are
computed server-side so the API surface stays close to the raw customer
attributes the caller actually has.

main.py uses:
- decision_tree.onnx        → exported full-feature sklearn Decision Tree (from train.py)
- feature_columns.json      → canonical feature order, used to sanity-check at startup
- onnxruntime               → cross-platform inference engine
- pydantic.BaseModel        → validates POST /predict payload
- fastapi                   → HTTP layer (auto-OpenAPI docs at /docs)

Endpoint:
- POST /predict             → returns prediction label + churn probability
"""

import json
from typing import Literal

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel


ONNX_MODEL_PATH = "decision_tree.onnx"
FEATURE_COLUMNS_PATH = "feature_columns.json"

# Expected feature order — must match what train.py produced.
EXPECTED_FEATURES = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "BalancePerProduct",
    "Geography_Germany",
    "Geography_Spain",
    "Gender_Male",
]


def _load_session() -> ort.InferenceSession:
    """Load the ONNX model and verify its expected feature count matches ours."""
    with open(FEATURE_COLUMNS_PATH) as f:
        trained_columns = json.load(f)

    if trained_columns != EXPECTED_FEATURES:
        raise RuntimeError(
            "Feature schema drift between train.py and main.py:\n"
            f"  trained: {trained_columns}\n"
            f"  expected: {EXPECTED_FEATURES}\n"
            "Re-run `python train.py` or update EXPECTED_FEATURES."
        )

    return ort.InferenceSession(ONNX_MODEL_PATH)


ort_session = _load_session()

app = FastAPI(title="Churn Prediction", version="2.0.0")


class InputData(BaseModel):
    """Raw customer attributes. Field types double as validation."""
    CreditScore: int
    Geography: Literal["France", "Germany", "Spain"]
    Gender: Literal["Male", "Female"]
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int             # 0 or 1
    IsActiveMember: int        # 0 or 1
    EstimatedSalary: float


@app.post("/predict")
def predict(data: InputData):
    # Engineered + one-hot features computed server-side so the request payload
    # stays close to what the caller already has.
    balance_per_product = data.Balance / (data.NumOfProducts + 1)
    geography_germany = 1 if data.Geography == "Germany" else 0
    geography_spain = 1 if data.Geography == "Spain" else 0
    gender_male = 1 if data.Gender == "Male" else 0

    input_array = np.array(
        [[
            data.CreditScore,
            data.Age,
            data.Tenure,
            data.Balance,
            data.NumOfProducts,
            data.HasCrCard,
            data.IsActiveMember,
            data.EstimatedSalary,
            balance_per_product,
            geography_germany,
            geography_spain,
            gender_male,
        ]],
        dtype=np.float32,
    )

    # ONNX Runtime returns: [predicted_class_array, [{class_id: prob, ...}, ...]]
    predicted_class_arr, class_probabilities = ort_session.run(None, {"input": input_array})

    predicted_class = int(predicted_class_arr[0])
    churn_probability = class_probabilities[0][1]   # P(class = 1)

    return {
        "prediction": "Customer will churn" if predicted_class == 1 else "Customer will not churn",
        "churn_probability": f"{churn_probability * 100:.2f}%",
    }
