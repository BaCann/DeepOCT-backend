# app/services/s3_service.py
import boto3
from botocore.exceptions import ClientError
from fastapi import UploadFile, HTTPException
import os
from datetime import datetime
import uuid
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class S3Service:
    def __init__(self):
        """
        Initialize S3 client
        EC2 IAM role provides credentials automatically - no need to specify access keys
        """
        self.s3_client = boto3.client(
            's3',
            region_name=os.getenv('AWS_REGION', 'ap-southeast-1')
        )
        self.bucket_name = os.getenv('S3_BUCKET_NAME')
        
        if not self.bucket_name:
            raise ValueError("S3_BUCKET_NAME environment variable not set")
        
        logger.info(f"S3Service initialized - Bucket: {self.bucket_name}, Region: {os.getenv('AWS_REGION')}")
    
    async def upload_image(
        self, 
        file: UploadFile, 
        user_id: int,
        folder: str = "oct_images"
    ) -> dict:
        """
        Upload image to S3
        
        Args:
            file: FastAPI UploadFile object
            user_id: User ID for organizing files
            folder: S3 folder prefix (default: oct_images)
            
        Returns:
            dict: {
                's3_key': 'oct_images/user_123/20250126_143022_abc123.jpg',
                's3_url': 'https://deepoct-images-production.s3.ap-southeast-1.amazonaws.com/...',
                'bucket': 'deepoct-images-production'
            }
        """
        try:
            # Generate unique filename
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            unique_id = str(uuid.uuid4())[:8]
            file_extension = os.path.splitext(file.filename)[1] or '.jpg'
            
            # S3 key structure: oct_images/user_123/20250126_143022_abc123.jpg
            s3_key = f"{folder}/{user_id}/{timestamp}_{unique_id}{file_extension}"
            
            # Reset file pointer to beginning
            await file.seek(0)
            
            # Upload to S3
            self.s3_client.upload_fileobj(
                file.file,
                self.bucket_name,
                s3_key,
                ExtraArgs={
                    'ContentType': file.content_type or 'image/jpeg',
                    'Metadata': {
                        'original_filename': file.filename,
                        'user_id': str(user_id),
                        'uploaded_at': timestamp
                    }
                }
            )
            
            # Generate S3 URL
            s3_url = f"https://{self.bucket_name}.s3.{os.getenv('AWS_REGION')}.amazonaws.com/{s3_key}"
            
            logger.info(f"✅ Uploaded to S3: {s3_key}")
            
            return {
                "s3_key": s3_key,
                "s3_url": s3_url,
                "bucket": self.bucket_name
            }
            
        except ClientError as e:
            logger.error(f"❌ S3 upload failed: {e}")
            raise HTTPException(status_code=500, detail=f"S3 upload failed: {str(e)}")
        except Exception as e:
            logger.error(f"❌ Unexpected error during S3 upload: {e}")
            raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    
    async def download_image(self, s3_key: str) -> bytes:
        """
        Download image from S3 to memory
        
        Args:
            s3_key: S3 object key (e.g., 'oct_images/user_123/image.jpg')
            
        Returns:
            bytes: Image data
        """
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            image_data = response['Body'].read()
            
            logger.info(f"✅ Downloaded from S3: {s3_key} ({len(image_data)} bytes)")
            
            return image_data
            
        except ClientError as e:
            logger.error(f"❌ S3 download failed: {e}")
            raise HTTPException(status_code=500, detail=f"S3 download failed: {str(e)}")
    
    async def delete_image(self, s3_key: str) -> bool:
        """
        Delete image from S3
        
        Args:
            s3_key: S3 object key
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            logger.info(f"✅ Deleted from S3: {s3_key}")
            return True
            
        except ClientError as e:
            logger.error(f"❌ S3 delete failed: {e}")
            return False
    
    def generate_presigned_url(
        self, 
        s3_key: str, 
        expiration: int = 3600
    ) -> str:
        """
        Generate presigned URL for temporary access
        
        Args:
            s3_key: S3 object key
            expiration: URL expiration time in seconds (default: 3600 = 1 hour)
            
        Returns:
            str: Presigned URL valid for specified duration
        """
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.bucket_name,
                    'Key': s3_key
                },
                ExpiresIn=expiration
            )
            
            logger.info(f"✅ Generated presigned URL for: {s3_key} (expires in {expiration}s)")
            
            return url
            
        except ClientError as e:
            logger.error(f"❌ Presigned URL generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"URL generation failed: {str(e)}")
    
    async def upload_heatmap(
        self,
        heatmap_data: bytes,
        original_s3_key: str
    ) -> str:
        """
        Upload heatmap image to S3
        
        Args:
            heatmap_data: Heatmap image as bytes
            original_s3_key: Original image S3 key
            
        Returns:
            str: Heatmap S3 URL
        """
        try:
            # Generate heatmap key from original key
            # Input:  oct_images/user_123/20250126_143022_abc123.jpg
            # Output: heatmaps/user_123/20250126_143022_abc123_heatmap.jpg
            heatmap_key = original_s3_key.replace('oct_images/', 'heatmaps/')
            base, ext = os.path.splitext(heatmap_key)
            heatmap_key = f"{base}_heatmap.jpg"
            
            # Upload heatmap
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=heatmap_key,
                Body=heatmap_data,
                ContentType='image/jpeg',
                Metadata={
                    'type': 'heatmap',
                    'original_image': original_s3_key
                }
            )
            
            # Generate URL
            heatmap_url = f"https://{self.bucket_name}.s3.{os.getenv('AWS_REGION')}.amazonaws.com/{heatmap_key}"
            
            logger.info(f"✅ Uploaded heatmap to S3: {heatmap_key}")
            
            return heatmap_url
            
        except ClientError as e:
            logger.error(f"❌ Heatmap upload failed: {e}")
            raise HTTPException(status_code=500, detail=f"Heatmap upload failed: {str(e)}")
        
    async def upload_avatar(
        self,
        file: UploadFile,
        user_id: int,
        old_avatar_url: Optional[str] = None
    ) -> dict:
        """
        Upload user avatar to S3
        
        Args:
            file: Avatar file
            user_id: User ID
            old_avatar_url: Previous avatar URL to delete
            
        Returns:
            dict: {'s3_key': '...', 's3_url': '...', 'bucket': '...'}
        """
        try:
            # Delete old avatar if exists
            if old_avatar_url:
                try:
                    old_key = old_avatar_url.split('.com/')[-1]
                    await self.delete_image(old_key)
                    logger.info(f"✅ Deleted old avatar: {old_key}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to delete old avatar: {e}")
            
            # Generate avatar filename
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            unique_id = str(uuid.uuid4())[:8]
            file_extension = os.path.splitext(file.filename)[1] or '.jpg'
            
            # S3 key: avatars/user_123/20250126_143022_abc123.jpg
            s3_key = f"avatars/{user_id}/avatar_{timestamp}_{unique_id}{file_extension}"
            
            # Reset file pointer
            await file.seek(0)
            
            # Upload to S3
            self.s3_client.upload_fileobj(
                file.file,
                self.bucket_name,
                s3_key,
                ExtraArgs={
                    'ContentType': file.content_type or 'image/jpeg',
                    'Metadata': {
                        'type': 'avatar',
                        'user_id': str(user_id),
                        'uploaded_at': timestamp
                    }
                }
            )
            
            # Generate S3 URL
            s3_url = f"https://{self.bucket_name}.s3.{os.getenv('AWS_REGION')}.amazonaws.com/{s3_key}"
            
            logger.info(f"✅ Uploaded avatar to S3: {s3_key}")
            
            return {
                "s3_key": s3_key,
                "s3_url": s3_url,
                "bucket": self.bucket_name
            }
            
        except ClientError as e:
            logger.error(f"❌ Avatar upload failed: {e}")
            raise HTTPException(status_code=500, detail=f"Avatar upload failed: {str(e)}")
        except Exception as e:
            logger.error(f"❌ Unexpected error during avatar upload: {e}")
            raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

# Singleton instance
s3_service = S3Service()