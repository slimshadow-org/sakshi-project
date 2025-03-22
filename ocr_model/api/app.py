from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status, Request
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional
import cv2
import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import io
import sys
import tempfile
import requests
from PIL import Image
import uvicorn
import shutil
from pathlib import Path
import py_text_scan
from sqlalchemy import create_engine, Column, Integer, String, Boolean, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from passlib.context import CryptContext
import datetime

# --- Database Setup (SQLite) ---
DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- Database Models ---
class UserModel(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)

class FeedbackModel(Base):
    __tablename__ = "feedback"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String)
    comment = Column(Text)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

Base.metadata.create_all(bind=engine)

# --- Pydantic Schemas ---
class UserBase(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr

class UserCreate(UserBase):
    password: str = Field(..., min_length=6)

class UserResponse(UserBase):
    id: int
    is_active: bool
    is_admin: bool
    class Config:
        from_attributes = True

class UserUpdate(BaseModel):
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    is_active: Optional[bool] = None
    is_admin: Optional[bool] = None

class FeedbackBase(BaseModel):
    username: str
    comment: str

class FeedbackCreate(FeedbackBase):
    pass

class FeedbackResponse(FeedbackBase):
    id: int
    created_at: datetime.datetime
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class OCRResponse(BaseModel):
    sakshi_output: str
    word_count: int
    prediction_label: str

# --- Password Hashing ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# --- Authentication ---
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def get_current_user(db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    user = get_user_by_username(db, username=token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user

async def get_current_active_user(current_user: UserModel = Depends(get_current_user)):
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

async def get_current_admin_user(current_user: UserModel = Depends(get_current_active_user)):
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Not an administrator")
    return current_user

# --- CRUD Operations ---
def get_user(db: Session, user_id: int):
    return db.query(UserModel).filter(UserModel.id == user_id).first()

def get_user_by_username(db: Session, username: str):
    return db.query(UserModel).filter(UserModel.username == username).first()

def get_user_by_email(db: Session, email: str):
    return db.query(UserModel).filter(UserModel.email == email).first()

def get_users(db: Session, skip: int = 0, limit: int = 100):
    return db.query(UserModel).offset(skip).limit(limit).all()

def create_user(db: Session, user: UserCreate):
    hashed_password = pwd_context.hash(user.password)
    db_user = UserModel(username=user.username, email=user.email, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def update_user(db: Session, user_id: int, user: UserUpdate):
    db_user = get_user(db, user_id)
    if db_user:
        for key, value in user.dict(exclude_unset=True).items():
            setattr(db_user, key, value)
        db.commit()
        db.refresh(db_user)
    return db_user

def delete_user(db: Session, user_id: int):
    db_user = get_user(db, user_id)
    if db_user:
        db.delete(db_user)
        db.commit()
        return True
    return False

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_feedback(db: Session, feedback: FeedbackCreate):
    db_feedback = FeedbackModel(**feedback.dict())
    db.add(db_feedback)
    db.commit()
    db.refresh(db_feedback)
    return db_feedback

def get_feedback(db: Session, skip: int = 0, limit: int = 100):
    return db.query(FeedbackModel).order_by(FeedbackModel.created_at.desc()).offset(skip).limit(limit).all()

# --- FastAPI App Setup ---
app = FastAPI(
    title="Hindi OCR API",
    description="API for Hindi OCR, word detection, authentication, and feedback",
    version="1.0.0"
)

# --- Hugging Face Model and Resource URLs ---
MODEL_URL = "https://huggingface.co/sameernotes/hindi-ocr/resolve/main/hindi_ocr_model.keras"
ENCODER_URL = "https://huggingface.co/sameernotes/hindi-ocr/resolve/main/label_encoder.pkl"
FONT_URL = "https://huggingface.co/sameernotes/hindi-ocr/resolve/main/NotoSansDevanagari-Regular.ttf"
MODEL_PATH = "hindi_ocr_model.keras"
ENCODER_PATH = "label_encoder.pkl"
FONT_PATH = "NotoSansDevanagari-Regular.ttf"

def download_file(url, dest):
    if not os.path.exists(dest):
        print(f"Downloading {dest}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(dest, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {dest}")

def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)

def load_label_encoder():
    if not os.path.exists(ENCODER_PATH):
       return None
    with open(ENCODER_PATH, 'rb') as f:
        return pickle.load(f)

model = None
label_encoder = None
session_files = {}

@app.on_event("startup")
async def startup_event():
    global model, label_encoder
    download_file(MODEL_URL, MODEL_PATH)
    download_file(ENCODER_URL, ENCODER_PATH)
    download_file(FONT_URL, FONT_PATH)

    if os.path.exists(FONT_PATH):
        fm.fontManager.addfont(FONT_PATH)
        plt.rcParams['font.family'] = 'Noto Sans Devanagari'
    model = load_model()
    label_encoder = load_label_encoder()

    db = SessionLocal()
    if not get_user_by_username(db, "admin"):
        admin_user = UserCreate(username="admin", email="admin@example.com", password="adminpassword")
        create_user(db, admin_user)
        admin = get_user_by_username(db, "admin")
        admin.is_admin = True
        db.commit()
    db.close()

def detect_words(image):
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    word_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    word_count = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 10 and h > 10:
            cv2.rectangle(word_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            word_count += 1
    return word_img, word_count

def run_py_text_scan(image_path):
    buffer = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buffer
    try:
        py_text_scan.generate(image_path)
    finally:
        sys.stdout = old_stdout
    return buffer.getvalue()

def process_image(image_array):
    img = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    word_detected_img, word_count = detect_words(img)
    word_detection_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    cv2.imwrite(word_detection_path, word_detected_img)
    session_files['word_detection'] = word_detection_path

    pred_path = None
    try:
        img_resized = cv2.resize(img, (128, 32))
        img_norm = img_resized / 255.0
        img_input = img_norm[np.newaxis, ..., np.newaxis]
        if model is not None and label_encoder is not None:
            pred = model.predict(img_input)
            pred_label_idx = np.argmax(pred)
            pred_label = label_encoder.inverse_transform([pred_label_idx])[0]
            fig, ax = plt.subplots()
            ax.imshow(img, cmap='gray')
            ax.set_title(f"Predicted: {pred_label}", fontsize=12)
            ax.axis('off')
            pred_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
            plt.savefig(pred_path)
            plt.close()
            session_files['prediction'] = pred_path
        else:
            pred_label = "Model or encoder not loaded"
    except Exception as e:
        pred_label = f"Error: {str(e)}"

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        cv2.imwrite(tmp_file.name, img)
        sakshi_output = run_py_text_scan(tmp_file.name)
        os.unlink(tmp_file.name)
    return {
        "sakshi_output": sakshi_output,
        "word_detection_path": word_detection_path if 'word_detection' in session_files else None,
        "word_count": word_count,
        "prediction_path": pred_path if 'prediction' in session_files else None,
        "prediction_label": pred_label
    }

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = get_user_by_username(db, form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = user.username
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/signup", response_model=UserResponse)
async def signup(user: UserCreate, db: Session = Depends(get_db)):
    db_user_username = get_user_by_username(db, username=user.username)
    if db_user_username:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already registered")
    db_user_email = get_user_by_email(db, email=user.email)
    if db_user_email:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")
    created = create_user(db=db, user=user)
    return created

@app.post("/process/", response_model=OCRResponse)
async def process(file: UploadFile = File(...), current_user: UserModel = Depends(get_current_active_user)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    for key, filepath in session_files.items():
        if os.path.exists(filepath):
            try:
                os.unlink(filepath)
            except:
                pass
    session_files.clear()

    temp_file = tempfile.NamedTemporaryFile(delete=False)
    try:
        with temp_file as f:
            shutil.copyfileobj(file.file, f)
        image = Image.open(temp_file.name)
        image_array = np.array(image)
        result = process_image(image_array)
        return OCRResponse(
            sakshi_output=result["sakshi_output"],
            word_count=result["word_count"],
            prediction_label=result["prediction_label"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    finally:
        os.unlink(temp_file.name)

@app.get("/word-detection/")
async def get_word_detection(current_user: UserModel = Depends(get_current_active_user)):
    if 'word_detection' not in session_files or not os.path.exists(session_files['word_detection']):
        raise HTTPException(status_code=404, detail="Word detection image not found")
    return FileResponse(session_files['word_detection'])

@app.get("/prediction/")
async def get_prediction(current_user: UserModel = Depends(get_current_active_user)):
    if 'prediction' not in session_files or not os.path.exists(session_files['prediction']):
        raise HTTPException(status_code=404, detail="Prediction image not found")
    return FileResponse(session_files['prediction'])

# --- Modified Feedback Endpoint ---
# No authentication dependency is used here so that anyone can submit feedback.
@app.post("/feedback/", response_model=FeedbackResponse)
async def create_feedback_route(feedback: FeedbackCreate, db: Session = Depends(get_db)):
    return create_feedback(db=db, feedback=feedback)

# --- Admin Endpoints ---
@app.get("/admin/users/")
async def admin_get_users(skip: int = 0, limit: int = 100, current_user: UserModel = Depends(get_current_admin_user), db: Session = Depends(get_db)):
    return get_users(db, skip=skip, limit=limit)

@app.delete("/admin/users/{user_id}")
async def admin_delete_user(user_id: int, current_user: UserModel = Depends(get_current_admin_user), db: Session = Depends(get_db)):
    if delete_user(db, user_id):
        return {"detail": "User deleted successfully"}
    raise HTTPException(status_code=404, detail="User not found")

@app.get("/admin/feedback/")
async def admin_get_feedback(skip: int = 0, limit: int = 100, current_user: UserModel = Depends(get_current_admin_user), db: Session = Depends(get_db)):
    return get_feedback(db, skip=skip, limit=limit)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
