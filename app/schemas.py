from pydantic import BaseModel, EmailStr, Field
from typing import Optional, Literal, Dict, List
from datetime import datetime


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


class UserProfile(BaseModel):
    id: int
    email: str
    full_name: Optional[str]
    mobile_number: Optional[str]
    date_of_birth: Optional[str]
    avatar_url: Optional[str] = None
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



DiseaseType = Literal['CNV', 'DME', 'DRUSEN', 'NORMAL']

class Probabilities(BaseModel):
    CNV: float = Field(..., ge=0.0, le=1.0)
    DME: float = Field(..., ge=0.0, le=1.0)
    DRUSEN: float = Field(..., ge=0.0, le=1.0)
    NORMAL: float = Field(..., ge=0.0, le=1.0)

class GradCAMAnalysis(BaseModel):
    """Schema chứa các chỉ số định lượng từ Grad-CAM"""
    analysis_status: Literal['SUCCESS', 'FAILED', 'ERROR']
    image_size_pixels: Optional[str] = Field(None, description="Kích thước ảnh gốc (Width x Height)")
    total_pixels: int = Field(..., ge=0)
    threshold: float = Field(..., ge=0.0, le=1.0, description="Ngưỡng chuẩn hóa CAM được sử dụng để tính Hot Area")
    hot_area_pixels: int = Field(..., ge=0, description="Tổng số pixel thuộc vùng 'nóng' (lớn hơn ngưỡng)")
    hot_area_ratio: float = Field(..., ge=0.0, le=1.0, description="Tỷ lệ diện tích vùng nóng (0.0 đến 1.0)")
    hot_area_percent: float = Field(..., ge=0.0, le=100.0, description="Phần trăm diện tích vùng nóng (0.0% đến 100.0%)")
    bb_width_pixels: int = Field(..., ge=0, description="Độ rộng (pixel) của Bounding Box bao quanh Hot Area")
    bb_height_pixels: int = Field(..., ge=0, description="Chiều cao (pixel) của Bounding Box bao quanh Hot Area")
    error_detail: Optional[str] = Field(None, description="Thông báo lỗi nếu phân tích thất bại")

class PredictionResponse(BaseModel):
    id: str
    user_id: int
    predicted_class: DiseaseType
    confidence: float
    probabilities: Probabilities
    image_url: str
    heatmap_url: Optional[str] = None
    analysis_result: Optional[GradCAMAnalysis] = None 
    inference_time: int  
    created_at: datetime

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
    items: List[PredictionHistoryItem]
    total: int
    page: int
    page_size: int