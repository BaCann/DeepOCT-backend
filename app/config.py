import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    DATABASE_URL = os.getenv("DATABASE_URL")
    SECRET_KEY = os.getenv("SECRET_KEY")
    ALGORITHM = os.getenv("ALGORITHM")
    ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES"))

    EMAIL_USERNAME = os.getenv("EMAIL_USERNAME")
    EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
    EMAIL_FROM = os.getenv("EMAIL_FROM")
    EMAIL_FROM_NAME = os.getenv("EMAIL_FROM_NAME")

settings = Settings()

