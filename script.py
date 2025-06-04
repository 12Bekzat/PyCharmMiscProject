# backend (FastAPI + face_recognition + база лиц + лог событий + API загрузки и истории)
# Файл: main.py
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import base64
import numpy as np
import cv2
from deepface import DeepFace
import tempfile
import os
from datetime import datetime

app = FastAPI()

# Разрешаем CORS для локальной разработки
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageData(BaseModel):
    image: str

# === БАЗА ЛИЦ ===
KNOWN_FACES_DIR = "known_faces"
known_face_encodings = []
known_face_names = []

# Загрузка известных лиц
def load_known_faces():
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Для deepface не нужно заранее загружать эмбеддинги, они будут созданы динамически
            known_face_names.append(os.path.splitext(filename)[0])

load_known_faces()

# === ЛОГИ ===
LOG_FILE = "events.log"
def log_event(name):
    with open(LOG_FILE, "a") as f:
        f.write(f"{datetime.now().isoformat()} - {name}\n")

def get_log():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            return f.readlines()
    return []

# === РАСПОЗНАВАНИЕ ===
@app.post("/api/recognize")
async def recognize(data: ImageData):
    try:
        header, encoded = data.image.split(",")
        decoded = base64.b64decode(encoded)
        nparr = np.frombuffer(decoded, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp:
            cv2.imwrite(temp.name, frame)
            results = DeepFace.find(img_path=temp.name, db_path=KNOWN_FACES_DIR, enforce_detection=False)

        names = []
        if len(results) > 0 and not results[0].empty:
            for _, row in results[0].iterrows():
                name = os.path.basename(row["identity"]).split(".")[0]
                names.append(name)
                log_event(name)
        else:
            names.append("Неизвестен")

        return {"faces_detected": len(names), "names": names}
    except Exception as e:
        return {"error": str(e)}

# === ЗАГРУЗКА НОВЫХ ЛИЦ ===
@app.post("/api/add_person")
async def add_person(name: str = Form(...), file: UploadFile = File(...)):
    try:
        ext = os.path.splitext(file.filename)[-1]
        filename = os.path.join(KNOWN_FACES_DIR, f"{name}{ext}")
        with open(filename, "wb") as f:
            f.write(await file.read())
        load_known_faces()
        return {"status": "added", "name": name}
    except Exception as e:
        return {"error": str(e)}

# === ПОЛУЧИТЬ ЛОГ ===
@app.get("/api/history")
def get_history():
    return {"log": get_log()}


@app.get("/api/people")
def get_all_people():
    try:
        people = []
        for filename in os.listdir(KNOWN_FACES_DIR):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                # Добавляем имя файла без расширения
                name = os.path.splitext(filename)[0]
                people.append(name)
        return {"people": people}
    except Exception as e:
        return {"error": str(e)}