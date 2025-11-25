# app/routers/predictions.py
import logging 
import uuid
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, Query
from sqlalchemy.orm import Session
from datetime import datetime
import os
import io
from PIL import Image as PILImage

from app.database import SessionLocal
from app.models import User, Prediction
from app.schemas import (
    PredictionResponse, 
    PredictionHistoryItem, 
    PredictionHistoryResponse
)
from app.user import get_current_user
from app.services.ml_service import ml_service
from app.services.s3_service import s3_service  


# Khởi tạo Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) 


router = APIRouter(prefix="/predictions", tags=["Predictions"])

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/predict", response_model=PredictionResponse)
async def create_prediction(
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Upload OCT image to S3, run ML prediction, and generate Grad-CAM.
    """
    logger.info(f"Prediction request from user {current_user.id}: {image.filename}")
    
    # 1. Validate image
    if not image.content_type or not image.content_type.startswith('image/'):
        logger.error(f"Invalid file type: {image.content_type}")
        raise HTTPException(status_code=400, detail="Invalid image file. Only images allowed.")
    
    # 2. Upload to S3
    try:
        s3_result = await s3_service.upload_image(
            file=image,
            user_id=current_user.id,
            folder="oct_images"
        )
        logger.info(f"Image uploaded to S3: {s3_result['s3_key']}")
    except Exception as e:
        logger.exception("S3 upload failed")
        raise HTTPException(status_code=500, detail=f"S3 upload failed: {str(e)}")
    
    # 3. Download image from S3 to temporary file for ML inference
    temp_path = None
    try:
        # Download image data
        image_data = await s3_service.download_image(s3_result['s3_key'])
        
        # Save to temp file (ML service needs file path)
        temp_path = f"/tmp/{uuid.uuid4()}.jpg"
        pil_image = PILImage.open(io.BytesIO(image_data)).convert('RGB')
        pil_image.save(temp_path)
        
        logger.info(f"Image downloaded to temp: {temp_path}")
        
    except Exception as e:
        logger.exception("Image processing failed")
        # Cleanup S3
        await s3_service.delete_image(s3_result['s3_key'])
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")
    
    # 4. Run ML prediction
    try:
        prediction_result = ml_service.predict(temp_path)
        logger.info(f"Prediction: {prediction_result['predicted_class']} ({prediction_result['confidence']:.2f})")
    except Exception as e:
        logger.exception("ML prediction failed")
        # Cleanup
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        await s3_service.delete_image(s3_result['s3_key'])
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    # 5. Generate Grad-CAM heatmap
    heatmap_url = None
    try:
        # Generate heatmap to temp file
        heatmap_temp_path = f"/tmp/{uuid.uuid4()}_heatmap.jpg"
        heatmap_success = ml_service.generate_gradcam(temp_path, heatmap_temp_path)
        
        if heatmap_success and os.path.exists(heatmap_temp_path):
            # Upload heatmap to S3
            with open(heatmap_temp_path, 'rb') as heatmap_file:
                heatmap_data = heatmap_file.read()
                heatmap_url = await s3_service.upload_heatmap(
                    heatmap_data,
                    s3_result['s3_key']
                )
            
            logger.info(f"Heatmap uploaded to S3: {heatmap_url}")
            
            # Clean up temp heatmap
            os.remove(heatmap_temp_path)
        else:
            logger.warning("Heatmap generation failed")
            
    except Exception as e:
        logger.warning(f"Heatmap processing failed: {e}")
    
    # Clean up temp image file
    try:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
            logger.info(f"Cleaned up temp file: {temp_path}")
    except Exception as e:
        logger.warning(f"Failed to clean temp file: {e}")
    
    # 6. Save to database
    prediction = Prediction(
        id=str(uuid.uuid4()),
        user_id=current_user.id,
        predicted_class=prediction_result['predicted_class'],
        confidence=prediction_result['confidence'],
        probabilities=prediction_result['probabilities'],
        image_path=s3_result['s3_key'],      # ← S3 key instead of disk path
        image_url=s3_result['s3_url'],       # ← S3 URL
        heatmap_url=heatmap_url,
        inference_time=prediction_result['inference_time']
    )
    
    db.add(prediction)
    db.commit()
    db.refresh(prediction)
    
    logger.info(f"Prediction saved to DB: {prediction.id}")
    
    return prediction


@router.get("/history", response_model=PredictionHistoryResponse)
async def get_prediction_history(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get paginated prediction history
    """
    logger.info(f"History request from user {current_user.id}: page={page}, page_size={page_size}")
    
    # 1. Count total predictions
    total = db.query(Prediction).filter(
        Prediction.user_id == current_user.id
    ).count()
    
    # 2. Get paginated predictions
    skip = (page - 1) * page_size
    predictions = db.query(Prediction).filter(
        Prediction.user_id == current_user.id
    ).order_by(
        Prediction.created_at.desc()
    ).offset(skip).limit(page_size).all()
    
    # 3. Convert to history items
    items = [
        PredictionHistoryItem(
            id=p.id,
            user_id=p.user_id,
            predicted_class=p.predicted_class,
            confidence=p.confidence,
            thumbnail_url=p.image_url,  # ← S3 URL directly
            created_at=p.created_at
        )
        for p in predictions
    ]
    
    logger.info(f"Returned {len(items)} items for user {current_user.id} (total: {total})")
    
    return PredictionHistoryResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size
    )


@router.get("/{prediction_id}", response_model=PredictionResponse)
async def get_prediction_detail(
    prediction_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get full prediction detail
    """
    logger.info(f"Detail request: prediction_id={prediction_id}, user={current_user.id}")
    
    prediction = db.query(Prediction).filter(
        Prediction.id == prediction_id,
        Prediction.user_id == current_user.id
    ).first()
    
    if not prediction:
        logger.warning(f"Prediction not found: {prediction_id}")
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    logger.info(f"Found prediction {prediction_id}: {prediction.predicted_class}")
    
    return prediction


@router.delete("/{prediction_id}")
async def delete_prediction(
    prediction_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Delete prediction and associated S3 images
    """
    logger.info(f"Delete request: prediction_id={prediction_id}, user={current_user.id}")
    
    prediction = db.query(Prediction).filter(
        Prediction.id == prediction_id,
        Prediction.user_id == current_user.id
    ).first()
    
    if not prediction:
        logger.warning(f"Prediction not found: {prediction_id}")
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    # Delete images from S3
    try:
        # Delete original image (image_path contains S3 key)
        await s3_service.delete_image(prediction.image_path)
        logger.info(f"Deleted S3 image: {prediction.image_path}")
        
        # Delete heatmap if exists
        if prediction.heatmap_url:
            # Extract S3 key from URL
            # URL: https://bucket.s3.region.amazonaws.com/heatmaps/user_1/image.jpg
            # Key: heatmaps/user_1/image.jpg
            heatmap_key = prediction.heatmap_url.split('.com/')[-1]
            await s3_service.delete_image(heatmap_key)
            logger.info(f"Deleted S3 heatmap: {heatmap_key}")
            
    except Exception as e:
        logger.warning(f"S3 cleanup failed: {e}")
        # Continue with DB deletion even if S3 cleanup fails
    
    # Delete from database
    db.delete(prediction)
    db.commit()
    
    logger.info(f"Prediction {prediction_id} deleted successfully")
    
    return {"msg": "Prediction deleted successfully"}