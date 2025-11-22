# app/schemas.py
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, Literal
from datetime import datetime

# ========== AUTH SCHEMAS ==========

class UserCreate(BaseModel):
    full_name: str
    email: EmailStr
    password: str
    mobile_number: str
    date_of_birth: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class ResetPasswordRequest(BaseModel):
    email: EmailStr

class ResetPasswordConfirm(BaseModel):
    otp: str

class ChangePasswordRequest(BaseModel):
    new_password: str
    reset_token: str

# ========== USER PROFILE SCHEMAS ==========

class UserProfile(BaseModel):
    id: int
    email: str
    full_name: Optional[str]
    mobile_number: Optional[str]
    date_of_birth: Optional[str]
    is_active: bool
    is_verified: bool
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class UpdateProfileRequest(BaseModel):
    full_name: Optional[str] = None
    mobile_number: Optional[str] = None
    date_of_birth: Optional[str] = None

class ChangePasswordInAppRequest(BaseModel):
    current_password: str
    new_password: str

class DeleteAccountRequest(BaseModel):
    password: str


# ========== PREDICTION SCHEMAS ==========

DiseaseType = Literal['CNV', 'DME', 'DRUSEN', 'NORMAL']

class Probabilities(BaseModel):
    CNV: float = Field(..., ge=0.0, le=1.0)
    DME: float = Field(..., ge=0.0, le=1.0)
    DRUSEN: float = Field(..., ge=0.0, le=1.0)
    NORMAL: float = Field(..., ge=0.0, le=1.0)

class PredictionResponse(BaseModel):
    id: str
    user_id: int
    predicted_class: DiseaseType
    confidence: float
    probabilities: Probabilities
    image_url: str
    inference_time: int  # milliseconds
    created_at: datetime
    heatmap_url: Optional[str] = None

    class Config:
        from_attributes = True

class PredictionHistoryItem(BaseModel):
    id: str
    user_id: int
    predicted_class: DiseaseType
    confidence: float
    thumbnail_url: str
    created_at: datetime

    class Config:
        from_attributes = True

class PredictionHistoryResponse(BaseModel):
    items: list[PredictionHistoryItem]
    total: int
    page: int
    page_size: int