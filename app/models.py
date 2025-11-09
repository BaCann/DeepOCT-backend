# app/models.py
from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean
from app.database import Base
from datetime import datetime

class User(Base):
    __tablename__ = "users"

    # Primary Key
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    
    # Authentication
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    
    # OTP for Password Reset
    otp_code = Column(String, nullable=True)
    otp_expiration = Column(DateTime, nullable=True)
    
    # Profile Information
    full_name = Column(String, nullable=True)
    mobile_number = Column(String, nullable=True)
    date_of_birth = Column(String, nullable=True)
    
    # JWT Refresh Token
    refresh_token = Column(Text, nullable=True)
    refresh_token_expire = Column(DateTime, nullable=True)
    
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    
    def __repr__(self):
        return f"<User(id={self.id}, email={self.email})>"