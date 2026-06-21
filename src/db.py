import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    password_hash = Column(String(128), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class Document(Base):
    __tablename__ = 'documents'
    id = Column(String(32), primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    filename = Column(String(255), nullable=False)
    pages = Column(Integer, default=0)
    chunks = Column(Integer, default=0)
    summary = Column(Text, default="")
    questions = Column(Text, default="[]")  # Store as JSON string
    pdf_path = Column(String(255), nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)

class AuditLog(Base):
    __tablename__ = 'audit_logs'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    action = Column(String(50), nullable=False)  # e.g., 'login', 'upload_doc', 'query'
    details = Column(Text, default="")
    timestamp = Column(DateTime, default=datetime.utcnow)

# Ensure data directory exists
os.makedirs("data", exist_ok=True)
engine = create_engine('sqlite:///data/enterprise_app.db', connect_args={'check_same_thread': False})
Base.metadata.create_all(engine)

SessionLocal = sessionmaker(bind=engine)

def get_session():
    return SessionLocal()
