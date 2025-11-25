# app/user.py
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials 
from sqlalchemy.orm import Session
from jose import jwt, JWTError
from passlib.context import CryptContext
from app import schemas, models
from app.database import SessionLocal
from app.config import settings
from app.services.s3_service import s3_service  # ← THAY ĐỔI: Import S3 thay vì file_handler

router = APIRouter()
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")
bearer_scheme = HTTPBearer()

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ========== GET CURRENT USER HELPER ==========
def get_current_user(
    token: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: Session = Depends(get_db)
) -> models.User:
    """Extract and verify JWT token, return current user"""
    try:
        token_value = token.credentials 
        payload = jwt.decode(token_value, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        email = payload.get("sub")
        token_type = payload.get("type")
        
        if token_type != "access":
            raise HTTPException(status_code=401, detail="Invalid token type")
        
        if not email:
            raise HTTPException(status_code=401, detail="Invalid token payload")
        
        user = db.query(models.User).filter_by(email=email).first()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        if not user.is_active:
            raise HTTPException(status_code=403, detail="Account has been deactivated")
        
        return user
        
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

# ========== GET PROFILE ==========
@router.get("/profile", response_model=schemas.UserProfile)
def get_profile(current_user: models.User = Depends(get_current_user)):
    """Get current user profile"""
    return current_user

# ========== UPDATE PROFILE ==========
@router.put("/profile")
def update_profile(
    data: schemas.UpdateProfileRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update user profile"""
    if data.full_name is not None:
        current_user.full_name = data.full_name
    if data.mobile_number is not None:
        current_user.mobile_number = data.mobile_number
    if data.date_of_birth is not None:
        current_user.date_of_birth = data.date_of_birth
    
    db.commit()
    db.refresh(current_user)
    
    return {"msg": "Profile updated successfully"}

# ========== CHANGE PASSWORD (IN-APP) ==========
@router.post("/change-password")
def change_password_in_app(
    data: schemas.ChangePasswordInAppRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Change password when already logged in"""
    if not pwd_context.verify(data.current_password, current_user.hashed_password):
        raise HTTPException(status_code=400, detail="Current password is incorrect")
    
    if len(data.new_password) < 6:
        raise HTTPException(status_code=400, detail="New password must be at least 6 characters")
    
    current_user.hashed_password = pwd_context.hash(data.new_password)
    db.commit()
    
    return {"msg": "Password changed successfully"}

# ========== DELETE ACCOUNT ==========
@router.delete("/account")
def delete_account(
    data: schemas.DeleteAccountRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete user account permanently"""
    if not pwd_context.verify(data.password, current_user.hashed_password):
        raise HTTPException(status_code=400, detail="Password is incorrect")
    
    # Delete user (cascade will delete predictions)
    db.delete(current_user)
    db.commit()
    
    return {"msg": "Account deleted successfully"}

# ========== UPLOAD AVATAR ========== (SỬA ĐỔI - Dùng S3)
@router.put("/avatar", response_model=schemas.UserProfile)
async def upload_avatar(
    avatar: UploadFile = File(...),
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Upload user avatar to S3"""
    
    # Validate image
    if not avatar.content_type or not avatar.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="Invalid image file. Only images allowed."
        )
    
    # Validate file extension
    allowed_extensions = {'jpg', 'jpeg', 'png'}
    file_extension = avatar.filename.split('.')[-1].lower()
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Only JPG, JPEG, PNG files allowed"
        )
    
    try:
        # Upload avatar to S3 (will delete old one if exists)
        s3_result = await s3_service.upload_avatar(
            avatar,
            current_user.id,
            current_user.avatar_url
        )
        
        # Update user profile with new S3 URL
        current_user.avatar_url = s3_result['s3_url']
        db.commit()
        db.refresh(current_user)
        
        return current_user
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload avatar: {str(e)}"
        )