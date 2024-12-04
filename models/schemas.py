from pydantic import BaseModel
from typing import List, Optional

class PredictionResponse(BaseModel):
    filename: str
    predicted_label: str
    probability: float
    error: Optional[str] = None

class RecommendationRequest(BaseModel):
    minat: str
    kemampuan: str
    kondisi: str

class JobRecommendation(BaseModel):
    Nama_Pekerjaan: str
    Perusahaan: str

class RecommendationResponse(BaseModel):
    success: bool
    message: str
    data: List[JobRecommendation]