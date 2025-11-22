# app/routers/predictions.py
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, Query
from sqlalchemy.orm import Session
import uuid
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
    MATCH frontend: predictionApi.predict(imageUri)
    
    Upload OCT image and run ML prediction
    Returns: PredictionResult
    """
    print(f"Received prediction request from user {current_user.id}")
    
    # 1. Validate image
    if not file_handler.validate_image(image):
        raise HTTPException(
            status_code=400, 
            detail="Invalid image file. Only JPG, JPEG, PNG allowed (max 10MB)"
        )
    
    # 2. Save image
    try:
        image_path, image_url = await file_handler.save_image(image, current_user.id)
        print(f"Image saved: {image_path}")
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save image: {str(e)}")
    
    # 3. Run ML prediction
    try:
        prediction_result = ml_service.predict(image_path)
        print(f"Prediction: {prediction_result['predicted_class']}")
    except Exception as e:
        # Clean up uploaded image if prediction fails
        file_handler.delete_image(image_path)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    # 4. Generate Grad-CAM heatmap
    heatmap_url = None
    try:
        heatmap_path = image_path.replace('/images/', '/heatmaps/').replace(
            f'.{image_path.split(".")[-1]}', '_heatmap.jpg'
        )
        os.makedirs(os.path.dirname(heatmap_path), exist_ok=True)
        heatmap_result = ml_service.generate_gradcam(image_path, heatmap_path)
        if heatmap_result:
            heatmap_url = f"http://192.168.1.102:8000/{heatmap_path}"
    except Exception as e:
        print(f"Heatmap generation failed: {e}")
    
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
        inference_time=prediction_result['inference_time']
    )
    
    db.add(prediction)
    db.commit()
    db.refresh(prediction)
    
    print(f"Prediction saved to database: {prediction.id}")
    
    return prediction


@router.get("/history", response_model=PredictionHistoryResponse)
async def get_prediction_history(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    MATCH frontend: predictionApi.getHistory(page, pageSize)
    
    Get paginated prediction history
    Returns: { items, total, page, page_size }
    """
    print(f"History request: page={page}, page_size={page_size}, user={current_user.id}")
    
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
    
    print(f"Returned {len(items)} items (total: {total})")
    
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
    MATCH frontend: predictionApi.getDetail(predictionId)
    
    Get full prediction detail
    Returns: PredictionResult
    """
    print(f"🔍 Detail request: prediction_id={prediction_id}, user={current_user.id}")
    
    prediction = db.query(Prediction).filter(
        Prediction.id == prediction_id,
        Prediction.user_id == current_user.id
    ).first()
    
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    print(f"Found prediction: {prediction.predicted_class}")
    
    return prediction


@router.delete("/{prediction_id}")
async def delete_prediction(
    prediction_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    MATCH frontend: predictionApi.delete(predictionId)
    
    Delete prediction
    Returns: { msg: string }
    """
    print(f"Delete request: prediction_id={prediction_id}, user={current_user.id}")
    
    prediction = db.query(Prediction).filter(
        Prediction.id == prediction_id,
        Prediction.user_id == current_user.id
    ).first()
    
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    # Delete files
    file_handler.delete_image(prediction.image_path)
    if prediction.heatmap_url:
        heatmap_path = prediction.heatmap_url.replace('http://192.168.1.102:8000/', '')
        if os.path.exists(heatmap_path):
            file_handler.delete_image(heatmap_path)
    
    # Delete from database
    db.delete(prediction)
    db.commit()
    
    print(f"Prediction deleted: {prediction_id}")
    
    return {"msg": "Prediction deleted successfully"}