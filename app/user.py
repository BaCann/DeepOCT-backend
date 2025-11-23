# app/user.py (FULL CODE ĐÃ SỬA ĐỔI)

from fastapi import APIRouter, Depends, HTTPException
# Thêm import cho HTTPBearer để kích hoạt nút Authorize trong Swagger UI
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials 
from sqlalchemy.orm import Session
from jose import jwt, JWTError
from passlib.context import CryptContext
from app import schemas, models
from app.database import SessionLocal
from app.config import settings

router = APIRouter()
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

# Khởi tạo Security Scheme - Điều này làm nút "Authorize" xuất hiện
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
    # Sử dụng Depends(bearer_scheme) để nhận token đã được xử lý (tách "Bearer ")
    token: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: Session = Depends(get_db)
) -> models.User:
    """Extract and verify JWT token, return current user"""
    try:
        # Lấy chuỗi JWT thuần túy (Access Token)
        token_value = token.credentials 
        
        # Decode token
        payload = jwt.decode(token_value, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        email = payload.get("sub")
        token_type = payload.get("type")
        
        # Verify token type
        if token_type != "access":
            raise HTTPException(status_code=401, detail="Invalid token type")
        
        if not email:
            raise HTTPException(status_code=401, detail="Invalid token payload")
        
        # Get user from database
        user = db.query(models.User).filter_by(email=email).first()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Check if account is active
        if not user.is_active:
            raise HTTPException(status_code=403, detail="Account has been deactivated")
        
        return user
        
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

# ---

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
    # Update fields if provided
    if data.full_name is not None:
        current_user.full_name = data.full_name
    if data.mobile_number is not None:
        current_user.mobile_number = data.mobile_number
    if data.date_of_birth is not None:
        current_user.date_of_birth = data.date_of_birth
    
    # updated_at will be automatically updated by SQLAlchemy
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
    # Verify current password
    if not pwd_context.verify(data.current_password, current_user.hashed_password):
        raise HTTPException(status_code=400, detail="Current password is incorrect")
    
    # Validate new password
    if len(data.new_password) < 6:
        raise HTTPException(status_code=400, detail="New password must be at least 6 characters")
    
    # Update password
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
    # Verify password
    if not pwd_context.verify(data.password, current_user.hashed_password):
        raise HTTPException(status_code=400, detail="Password is incorrect")
    
    # Delete user
    db.delete(current_user)
    db.commit()
    
    return {"msg": "Account deleted successfully"}