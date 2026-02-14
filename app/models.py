from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, Float, JSON, ForeignKey
from sqlalchemy.orm import relationship
from app.database import Base
from datetime import datetime
from datetime import datetime, timezone
import uuid

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    
    otp_code = Column(String, nullable=True)
    otp_expiration = Column(DateTime, nullable=True)
    
    full_name = Column(String, nullable=True)
    mobile_number = Column(String, nullable=True)
    date_of_birth = Column(String, nullable=True)
    avatar_url = Column(String, nullable=True)
    
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

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    predicted_class = Column(String, nullable=False)  # 'CNV' | 'DME' | 'DRUSEN' | 'NORMAL'
    confidence = Column(Float, nullable=False)  
    probabilities = Column(JSON, nullable=False)  
    
    image_path = Column(String, nullable=False)  
    image_url = Column(String, nullable=False)   
    heatmap_url = Column(String, nullable=True)  
    
    inference_time = Column(Integer, nullable=False)  
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    user = relationship("User", back_populates="predictions")
    
    def __repr__(self):
        return f"<Prediction(id={self.id}, class={self.predicted_class}, confidence={self.confidence})>"