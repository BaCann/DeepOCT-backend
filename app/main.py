# app/main.py
import time
from sqlalchemy.exc import OperationalError
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os

from app import models, database, auth, user
from app.routers import predictions  
from app.database import engine

max_tries = 10
for i in range(max_tries):
    try:
        models.Base.metadata.create_all(bind=engine)
        print(" Kết nối DB thành công và tạo bảng.")
        break
    except OperationalError as e:
        print(f" Kết nối DB thất bại ({i+1}/{max_tries}), thử lại sau 2s...")
        time.sleep(2)
else:
    raise RuntimeError(" Không thể kết nối DB sau nhiều lần thử.")

app = FastAPI(title="DeepOCT API - OCT Diagnosis System")

#  CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#  Serve uploaded files
os.makedirs("uploads", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Register routers
app.include_router(auth.router, tags=["Authentication"])
app.include_router(user.router, tags=["User Profile"])
app.include_router(predictions.router, tags=["Predictions"])  

@app.get("/")
def root():
    return {
        "message": "DeepOCT API is running",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "auth": "/login, /register, /reset-password",
            "user": "/profile, /change-password, /account",
            "predictions": "/predictions/predict, /predictions/history, /predictions/{id}"
        }
    }