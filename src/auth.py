import bcrypt
from src.db import get_session, User, AuditLog

def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

def verify_password(password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))

def register_user(username: str, password: str) -> bool:
    session = get_session()
    try:
        if session.query(User).filter_by(username=username).first():
            return False  # User already exists
        
        new_user = User(username=username, password_hash=hash_password(password))
        session.add(new_user)
        session.commit()
        return True
    except Exception as e:
        session.rollback()
        print(f"Registration error: {e}")
        return False
    finally:
        session.close()

def authenticate_user(username: str, password: str):
    session = get_session()
    try:
        user = session.query(User).filter_by(username=username).first()
        if user and verify_password(password, user.password_hash):
            return user.id
        return None
    finally:
        session.close()

def log_action(user_id: int, action: str, details: str = ""):
    session = get_session()
    try:
        log = AuditLog(user_id=user_id, action=action, details=details)
        session.add(log)
        session.commit()
    finally:
        session.close()
