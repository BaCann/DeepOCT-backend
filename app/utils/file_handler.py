# app/utils/file_handler.py
import os
import uuid
from pathlib import Path
from fastapi import UploadFile, HTTPException

UPLOAD_DIR = "uploads"
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

class FileHandler:
    @staticmethod
    def validate_image(file: UploadFile) -> bool:
        """Validate image file"""
        # Check extension
        if not file.filename:
            return False
            
        ext = file.filename.split('.')[-1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            return False
        
        # Check content type
        if not file.content_type or not file.content_type.startswith('image/'):
            return False
        
        return True
    
    @staticmethod
    async def save_image(file: UploadFile, user_id: int) -> tuple[str, str]:
        """
        Save uploaded image
        Returns: (local_path, public_url)
        """
        # Create directory structure
        user_dir = Path(UPLOAD_DIR) / str(user_id) / "images"
        user_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename
        file_id = str(uuid.uuid4())
        ext = file.filename.split('.')[-1].lower()
        filename = f"{file_id}.{ext}"
        
        # Save file
        file_path = user_dir / filename
        
        try:
            content = await file.read()
            
            # Validate file size
            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(status_code=413, detail="File too large. Maximum size is 10MB")
            
            with open(file_path, "wb") as buffer:
                buffer.write(content)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
        
        # Generate public URL (Đã thay đổi cổng từ 1.59 thành 1.102 để phù hợp với heatmap)
        # Giữ nguyên 192.168.1.59 như trong code cũ
        public_url = f"http://192.168.1.59:8000/uploads/{user_id}/images/{filename}" 
        
        return str(file_path), public_url
    
    @staticmethod
    def delete_image(image_path: str):
        """Delete image file"""
        try:
            if os.path.exists(image_path):
                os.remove(image_path)
                print(f"Deleted: {image_path}")
        except Exception as e:
            print(f"Failed to delete image: {e}")
    
    @staticmethod
    def get_thumbnail_url(image_url: str) -> str:
        """
        Get thumbnail URL
        For now, just return original image URL
        """
        return image_url

file_handler = FileHandler()