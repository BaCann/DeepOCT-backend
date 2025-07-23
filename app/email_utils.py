import smtplib
from email.message import EmailMessage
from app.config import settings

def send_email_otp(to_email: str, otp: str):
    msg = EmailMessage()
    msg["Subject"] = "Your OTP Code"
    msg["From"] = f"{settings.EMAIL_FROM_NAME} <{settings.EMAIL_FROM}>"
    msg["To"] = to_email
    msg.set_content(f"Your OTP code is: {otp}. It will expire in 5 minutes.")

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(settings.EMAIL_USERNAME, settings.EMAIL_PASSWORD)
        smtp.send_message(msg)

