from sqlalchemy import Column, Integer, String, DateTime, Text
from app.database import Base
from datetime import datetime

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    otp_code = Column(String, nullable=True)
    otp_expiration = Column(DateTime, nullable=True)
    full_name = Column(String, nullable=True)  
    mobile_number = Column(String, nullable=True)  
    date_of_birth = Column(String, nullable=True)
    role = Column(String, nullable=True)
    refresh_token = Column(Text, nullable=True)
    refresh_token_expire = Column(DateTime, nullable=True)
