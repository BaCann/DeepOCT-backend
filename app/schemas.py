# app/schemas.py
from pydantic import BaseModel, EmailStr
from typing import Optional
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