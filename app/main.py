# app/main.py
import time
from sqlalchemy.exc import OperationalError
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

from app import models, database, auth, user
from app.routers import predictions  
from app.database import engine

# ========== DATABASE CONNECTION ==========
max_tries = 10
for i in range(max_tries):
    try:
        models.Base.metadata.create_all(bind=engine)
        print("Database connected successfully and tables created")
        break
    except OperationalError as e:
        print(f"Database connection failed ({i+1}/{max_tries}), retrying in 2s...")
        time.sleep(2)
else:
    raise RuntimeError("Could not connect to database after multiple attempts")

# ========== FASTAPI APP ==========
app = FastAPI(
    title="DeepOCT API",
    description="OCT Diagnosis System with AI-powered retinal disease classification",
    version="2.0.0"
)

# ========== CORS MIDDLEWARE ==========
# Read allowed origins from environment
allowed_origins = os.getenv("CORS_ORIGINS", "").strip('[]').replace('"', '').split(',')
if not allowed_origins or allowed_origins == ['']:
    # Fallback for development
    allowed_origins = ["*"]
    print("CORS: Allowing all origins (development mode)")
else:
    print(f"CORS enabled for: {allowed_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== STATIC FILES (OPTIONAL) ==========
# Uncomment nếu cần serve static files từ disk
# os.makedirs("uploads", exist_ok=True)
# app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# ========== REGISTER ROUTERS ==========
app.include_router(auth.router, tags=["Authentication"])
app.include_router(user.router, tags=["User Profile"])
app.include_router(predictions.router, tags=["Predictions"])

# ========== ROOT ENDPOINT ==========
@app.get("/")
def root():
    return {
        "message": "DeepOCT API is running",
        "version": "2.0.0",
        "environment": os.getenv("ENVIRONMENT", "development"),
        "infrastructure": {
            "database": "AWS RDS PostgreSQL",
            "storage": "AWS S3",
            "domain": os.getenv("BASE_URL", "http://localhost:8000")
        },
        "docs": "/docs",
        "redoc": "/redoc",
        "endpoints": {
            "auth": "/login, /register, /reset-password, /refresh-token",
            "user": "/profile, /avatar, /change-password, /account",
            "predictions": "/predictions/predict, /predictions/history, /predictions/{id}"
        }
    }

# ========== HEALTH CHECK ==========
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "database": "connected",
        "storage": "s3"
    }