from pydantic import BaseModel
from typing import Dict, List


class ModelPrediction(BaseModel):
    model_name: str
    predicted_class: str
    confidence: float
    all_probabilities: Dict[str, float]


class AnalysisResponse(BaseModel):
    final_prediction: str
    final_confidence: float
    all_probabilities: Dict[str, float]
    individual_models: List[ModelPrediction]
    processing_time_ms: float