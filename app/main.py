import time
from sqlalchemy.exc import OperationalError
from fastapi import FastAPI
from app import models, database, auth
from app.database import engine

# Thử kết nối DB và tạo bảng, retry nếu chưa sẵn sàng
max_tries = 10
for i in range(max_tries):
    try:
        models.Base.metadata.create_all(bind=engine)
        print("✅ Kết nối DB thành công và tạo bảng.")
        break
    except OperationalError as e:
        print(f"❌ Kết nối DB thất bại ({i+1}/{max_tries}), thử lại sau 2s...")
        time.sleep(2)
else:
    raise RuntimeError("❌ Không thể kết nối DB sau nhiều lần thử.")

app = FastAPI(title="FastAPI Auth System create by CanDB - DuongNVD")

app.include_router(auth.router)

