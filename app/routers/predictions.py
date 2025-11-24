# app/routers/predictions.py
import logging 
import uuid
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, Query
from sqlalchemy.orm import Session
from datetime import datetime
import os

from app.database import SessionLocal
from app.models import User, Prediction
from app.schemas import (
    PredictionResponse, 
    PredictionHistoryItem, 
    PredictionHistoryResponse
)
from app.user import get_current_user
from app.utils.file_handler import file_handler
from app.services.ml_service import ml_service


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
    Upload OCT image, run ML prediction, and generate Grad-CAM visualization.
    """
    logger.info(f"Received prediction request from user {current_user.id} for file {image.filename}")
    
    # 1. Validate image
    if not file_handler.validate_image(image):
        logger.error(f"Validation failed for user {current_user.id}. File: {image.filename}")
        raise HTTPException(
            status_code=400, 
            detail="Invalid image file. Only JPG, JPEG, PNG allowed (max 10MB)"
        )
    
    # 2. Save image
    try:
        image_path, image_url = await file_handler.save_image(image, current_user.id)
        logger.info(f"Image saved successfully: {image_path}")
    except HTTPException as e:
        logger.error(f"HTTPException while saving image for user {current_user.id}: {e.detail}")
        raise e
    except Exception as e:
        logger.exception(f"Unexpected error while saving image for user {current_user.id}")
        raise HTTPException(status_code=500, detail=f"Failed to save image: {str(e)}")
    
    # 3. Run ML prediction
    try:
        prediction_result = ml_service.predict(image_path)
        logger.info(f"ML Prediction completed: {prediction_result['predicted_class']} (Conf: {prediction_result['confidence']:.2f})")
    except Exception as e:
        # Clean up uploaded image if prediction fails
        file_handler.delete_image(image_path)
        logger.exception(f"ML Prediction failed for image: {image_path}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    # 4. Generate Grad-CAM heatmap
    heatmap_url = None
    try:
        # 4a. Lấy đường dẫn heatmap từ file_handler
        heatmap_path, heatmap_url_temp = file_handler.get_heatmap_path(image_path)
        
        # 4b. Tạo Grad-CAM visualization
        logger.info(f"Generating Grad-CAM heatmap: {heatmap_path}")
        heatmap_success = ml_service.generate_gradcam(image_path, heatmap_path)
        
        if heatmap_success and os.path.exists(heatmap_path):
            heatmap_url = heatmap_url_temp
            logger.info(f"Heatmap generated successfully: {heatmap_url}")
        else:
            logger.warning(f"Heatmap generation failed or file not created: {heatmap_path}")
            
    except Exception as e:
        logger.exception(f"Unexpected error during Heatmap generation for {image_path}: {e}")

    
    # 5. Save to database
    prediction = Prediction(
        id=str(uuid.uuid4()),
        user_id=current_user.id,
        predicted_class=prediction_result['predicted_class'],
        confidence=prediction_result['confidence'],
        probabilities=prediction_result['probabilities'],
        image_path=image_path,
        image_url=image_url,
        heatmap_url=heatmap_url,
        # analysis_result = (ĐÃ BỎ)
        inference_time=prediction_result['inference_time']
    )
    
    db.add(prediction)
    db.commit()
    db.refresh(prediction)
    
    logger.info(f"Prediction saved to database: ID {prediction.id} for user {current_user.id}")
    
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
            thumbnail_url=file_handler.get_thumbnail_url(p.image_url),
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
        logger.warning(f"Prediction ID {prediction_id} not found for user {current_user.id}")
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    logger.info(f"Found prediction {prediction_id}: Class {prediction.predicted_class}")
    
    return prediction


@router.delete("/{prediction_id}")
async def delete_prediction(
    prediction_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Delete prediction
    """
    logger.info(f"Delete request: prediction_id={prediction_id}, user={current_user.id}")
    
    prediction = db.query(Prediction).filter(
        Prediction.id == prediction_id,
        Prediction.user_id == current_user.id
    ).first()
    
    if not prediction:
        logger.warning(f"Attempted delete of non-existent prediction {prediction_id} by user {current_user.id}")
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    # Delete image file
    file_handler.delete_image(prediction.image_path)
    
    # Delete heatmap file if exists
    if prediction.heatmap_url:
        success = file_handler.delete_heatmap(prediction.heatmap_url)
        if success:
            logger.info(f"Deleted heatmap file for prediction {prediction_id}")
        else:
            logger.warning(f"Failed to delete heatmap for prediction {prediction_id}")
    
    # Delete from database
    db.delete(prediction)
    db.commit()
    
    logger.info(f"Prediction {prediction_id} deleted successfully.")
    
    return {"msg": "Prediction deleted successfully"}