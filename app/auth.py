from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from jose import jwt, JWTError
from passlib.context import CryptContext
from app import schemas, models, email_utils
from app.database import SessionLocal
from app.config import settings
import random

router = APIRouter()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

@router.post("/register")
def register(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter_by(email=user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed = pwd_context.hash(user.password)
    db_user = models.User(email=user.email, hashed_password=hashed)
    db.add(db_user)
    db.commit()
    return {"msg": "User registered successfully"}

@router.post("/login")
def login(user: schemas.UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter_by(email=user.email).first()
    if not db_user or not pwd_context.verify(user.password, db_user.hashed_password):
        raise HTTPException(status_code=400, detail="Invalid credentials")
    token = create_access_token({"sub": user.email}, timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
    return {"access_token": token, "token_type": "bearer"}

@router.post("/reset-password")
def reset_password(request: schemas.ResetPasswordRequest, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter_by(email=request.email).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    otp = str(random.randint(100000, 999999))
    db_user.otp_code = otp
    db_user.otp_expiration = datetime.utcnow() + timedelta(minutes=5)
    db.commit()
    email_utils.send_email_otp(request.email, otp)
    return {"msg": "OTP sent to email"}

@router.post("/reset-password/confirm")
def reset_confirm(data: schemas.ResetPasswordConfirm, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter_by(email=data.email).first()
    if not db_user or db_user.otp_code != data.otp or datetime.utcnow() > db_user.otp_expiration:
        raise HTTPException(status_code=400, detail="Invalid or expired OTP")
    db_user.hashed_password = pwd_context.hash(data.new_password)
    db_user.otp_code = None
    db_user.otp_expiration = None
    db.commit()
    return {"msg": "Password reset successful"}

