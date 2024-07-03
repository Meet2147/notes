from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, String, Integer, Boolean, ForeignKey, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from passlib.context import CryptContext
import openai
import os
import uuid
import datetime

# Set your OpenAI API key

openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Database setup
DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)

class SessionToken(Base):
    __tablename__ = "session_tokens"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, index=True)
    user_id = Column(Integer)
    expires_at = Column(Integer)

class Transcription(Base):
    __tablename__ = "transcriptions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    session_id = Column(String)
    transcription = Column(String)
    summary = Column(String)
    date = Column(DateTime, default=datetime.datetime.utcnow)

class Task(Base):
    __tablename__ = "tasks"
    id = Column(Integer, primary_key=True, index=True)
    transcription_id = Column(Integer, ForeignKey("transcriptions.id"))
    task = Column(String)
    completed = Column(Boolean, default=False)

Base.metadata.create_all(bind=engine)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_session_token(user_id):
    session_id = str(uuid.uuid4())
    expires_at = int((datetime.datetime.utcnow() + datetime.timedelta(minutes=60)).timestamp())
    return session_id, expires_at

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class UserCreate(BaseModel):
    username: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    session_id: str

class AudioResponse(BaseModel):
    transcription: str
    summary: str
    tasks: list

class SessionStatus(BaseModel):
    minutes_left: int

class TaskUpdate(BaseModel):
    task_id: int
    completed: bool

@app.post("/register/")
async def register(user: UserCreate, db: Session = Depends(get_db)):
    hashed_password = get_password_hash(user.password)
    db_user = User(username=user.username, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return {"message": "User registered successfully"}

@app.post("/login/", response_model=LoginResponse)
async def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if not db_user or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=400, detail="Invalid username or password")
    
    session_id, expires_at = create_session_token(db_user.id)
    db_session = SessionToken(session_id=session_id, user_id=db_user.id, expires_at=expires_at)
    db.add(db_session)
    db.commit()
    return {"session_id": session_id}

@app.post("/process-audio/", response_model=AudioResponse)
async def process_audio(file: UploadFile = File(...), session_id: str = None, db: Session = Depends(get_db)):
    db_session = db.query(SessionToken).filter(SessionToken.session_id == session_id).first()
    if not db_session or db_session.expires_at < int(datetime.datetime.utcnow().timestamp()):
        raise HTTPException(status_code=401, detail="Session expired or invalid")
    
    file_location = f"temp_{file.filename}"
    
    with open(file_location, "wb") as f:
        f.write(await file.read())

    # Transcribe the uploaded audio
    transcription_text = transcribe_audio(file_location)

    # Generate summary and tasks
    summary_text = generate_summary(transcription_text)
    tasks_text = generate_tasks(transcription_text)

    # Save transcription and tasks to the database
    db_transcription = Transcription(user_id=db_session.user_id, session_id=session_id, transcription=transcription_text, summary=summary_text)
    db.add(db_transcription)
    db.commit()
    db.refresh(db_transcription)

    task_list = tasks_text.split("\n")  # Assuming tasks are separated by new lines
    tasks = []
    for task in task_list:
        db_task = Task(transcription_id=db_transcription.id, task=task, completed=False)
        db.add(db_task)
        db.commit()
        tasks.append({"task": task, "completed": False})

    return AudioResponse(
        transcription=transcription_text,
        summary=summary_text,
        tasks=tasks
    )

@app.get("/session-status/", response_model=SessionStatus)
async def session_status(session_id: str, db: Session = Depends(get_db)):
    db_session = db.query(SessionToken).filter(SessionToken.session_id == session_id).first()
    if not db_session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    current_time = int(datetime.datetime.utcnow().timestamp())
    if db_session.expires_at < current_time:
        raise HTTPException(status_code=401, detail="Session expired")

    minutes_left = (db_session.expires_at - current_time) // 60

    return {"minutes_left": minutes_left}

@app.put("/update-session-time/")
async def update_session_time(session_id: str, minutes_left: int, db: Session = Depends(get_db)):
    db_session = db.query(SessionToken).filter(SessionToken.session_id == session_id).first()
    if not db_session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    current_time = int(datetime.datetime.utcnow().timestamp())
    new_expires_at = current_time + minutes_left * 60
    db_session.expires_at = new_expires_at
    db.commit()
    return {"message": "Session time updated successfully"}

@app.get("/transcriptions/")
async def get_transcriptions(session_id: str, db: Session = Depends(get_db)):
    db_session = db.query(SessionToken).filter(SessionToken.session_id == session_id).first()
    if not db_session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    transcriptions = db.query(Transcription).filter(Transcription.user_id == db_session.user_id).all()
    grouped_transcriptions = {}
    for t in transcriptions:
        date_str = t.date.strftime("%Y-%m-%d")
        if date_str not in grouped_transcriptions:
            grouped_transcriptions[date_str] = []
        grouped_transcriptions[date_str].append({"id": t.id, "transcription": t.transcription, "summary": t.summary})
    
    return grouped_transcriptions

@app.get("/tasks/")
async def get_tasks(transcription_id: int, db: Session = Depends(get_db)):
    tasks = db.query(Task).filter(Task.transcription_id == transcription_id).all()
    return [{"id": t.id, "task": t.task, "completed": t.completed} for t in tasks]

@app.put("/tasks/", response_model=TaskUpdate)
async def update_task(task_update: TaskUpdate, db: Session = Depends(get_db)):
    db_task = db.query(Task).filter(Task.id == task_update.task_id).first()
    if not db_task:
        raise HTTPException(status_code=404, detail="Task not found")

    db_task.completed = task_update.completed
    db.commit()
    return task_update

def transcribe_audio(file_path):
    with open(file_path, "rb") as audio_file:
        transcription = openai.Audio.transcribe(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )
    return transcription

def generate_summary(transcript):
    prompt = (
        "Summarize the following transcript:\n\n"
        f"Transcript:\n{transcript}\n\n"
        "Summary:"
    )

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant who summarizes spoken text."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message['content']

def generate_tasks(transcript):
    prompt = (
        "From the following transcript, generate a list of tasks and to-dos:\n\n"
        f"Transcript:\n{transcript}\n\n"
        "Tasks:"
    )

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant who extracts tasks from spoken text."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message['content']

if __name__ == "____main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
