from pydantic import BaseModel, EmailStr

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