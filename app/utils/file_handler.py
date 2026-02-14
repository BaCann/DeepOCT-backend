import os
import uuid
from pathlib import Path
from fastapi import UploadFile, HTTPException
from app.config import settings

UPLOAD_DIR = "uploads"
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
MAX_FILE_SIZE = 10 * 1024 * 1024  

class FileHandler:
    @staticmethod
    def validate_image(file: UploadFile) -> bool:
        """Validate image file"""
        if not file.filename:
            return False
            
        ext = file.filename.split('.')[-1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            return False
        
        if not file.content_type or not file.content_type.startswith('image/'):
            return False
        
        return True
    
    @staticmethod
    async def save_image(file: UploadFile, user_id: int) -> tuple[str, str]:

        user_dir = Path(UPLOAD_DIR) / str(user_id) / "images"
        user_dir.mkdir(parents=True, exist_ok=True)
        
        file_id = str(uuid.uuid4())
        ext = file.filename.split('.')[-1].lower()
        filename = f"{file_id}.{ext}"
        
        file_path = user_dir / filename
        
        try:
            content = await file.read()
            
            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(status_code=413, detail="File too large. Maximum size is 10MB")
            
            with open(file_path, "wb") as buffer:
                buffer.write(content)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
        
        public_url = f"{settings.BASE_URL}/uploads/{user_id}/images/{filename}"
        
        return str(file_path), public_url
    
    @staticmethod
    async def save_avatar(file: UploadFile, user_id: int, old_avatar_url: str = None) -> tuple[str, str]:

        avatar_dir = Path(UPLOAD_DIR) / str(user_id) / "avatars"
        avatar_dir.mkdir(parents=True, exist_ok=True)
        

        if old_avatar_url:
            try:
                old_path = old_avatar_url.replace(f"{settings.BASE_URL}/", '')
                if os.path.exists(old_path):
                    os.remove(old_path)
                    print(f"Deleted old avatar: {old_path}")
            except Exception as e:
                print(f"Failed to delete old avatar: {e}")
        
        file_id = str(uuid.uuid4())
        ext = file.filename.split('.')[-1].lower()
        filename = f"avatar_{file_id}.{ext}"
        
        file_path = avatar_dir / filename
        
        try:
            content = await file.read()
            
            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(status_code=413, detail="File too large. Maximum size is 10MB")
            
            with open(file_path, "wb") as buffer:
                buffer.write(content)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save avatar: {str(e)}")
        
        public_url = f"{settings.BASE_URL}/uploads/{user_id}/avatars/{filename}"
        
        return str(file_path), public_url
    
    @staticmethod
    def get_heatmap_path(image_path: str) -> tuple[str, str]:
       
        heatmap_path = image_path.replace('/images/', '/heatmaps/').replace(
            f'.{image_path.split(".")[-1]}', '_heatmap.jpg'
        )
        
        os.makedirs(os.path.dirname(heatmap_path), exist_ok=True)
        
        heatmap_url = f"{settings.BASE_URL}/{heatmap_path}"
        
        return heatmap_path, heatmap_url
    
    @staticmethod
    def delete_image(image_path: str):
        try:
            if os.path.exists(image_path):
                os.remove(image_path)
                print(f"Deleted: {image_path}")
        except Exception as e:
            print(f"Failed to delete image: {e}")
    
    @staticmethod
    def delete_heatmap(heatmap_url: str) -> bool:

        try:
            heatmap_path = heatmap_url.replace(f"{settings.BASE_URL}/", '')
            
            if heatmap_path.startswith('http'):
                print(f"Invalid heatmap URL format: {heatmap_url}")
                return False
            
            if os.path.exists(heatmap_path):
                os.remove(heatmap_path)
                print(f"Deleted heatmap: {heatmap_path}")
                return True
            else:
                print(f"Heatmap file not found: {heatmap_path}")
                return False
                
        except Exception as e:
            print(f"Failed to delete heatmap: {e}")
            return False
    
    @staticmethod
    def get_thumbnail_url(image_url: str) -> str:

        return image_url

file_handler = FileHandler()