# app/models.py
from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, Float, JSON, ForeignKey
from sqlalchemy.orm import relationship
from app.database import Base
from datetime import datetime
from datetime import datetime, timezone
import uuid

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
    avatar_url = Column(String, nullable=True)
    
    # JWT Refresh Token
    refresh_token = Column(Text, nullable=True)
    refresh_token_expire = Column(DateTime, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    

    predictions = relationship("Prediction", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(id={self.id}, email={self.email})>"



class Prediction(Base):
    __tablename__ = "predictions"

    # Primary key
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Foreign key
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    # Prediction results
    predicted_class = Column(String, nullable=False)  # 'CNV' | 'DME' | 'DRUSEN' | 'NORMAL'
    confidence = Column(Float, nullable=False)  # 0.0 - 1.0
    probabilities = Column(JSON, nullable=False)  # { 'CNV': 0.96, 'DME': 0.02, ... }
    
    # File paths
    image_path = Column(String, nullable=False)  # Local file path
    image_url = Column(String, nullable=False)   # Public URL
    heatmap_url = Column(String, nullable=True)  # Optional heatmap URL
    
    # Metadata
    inference_time = Column(Integer, nullable=False)  # Milliseconds
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship
    user = relationship("User", back_populates="predictions")
    
    def __repr__(self):
        return f"<Prediction(id={self.id}, class={self.predicted_class}, confidence={self.confidence})>"