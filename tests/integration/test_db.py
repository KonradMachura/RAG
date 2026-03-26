import os
import uuid
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.database.models import Base, User, Document
from src.core import config as cfg

def test_database_connection():
    # Use an in-memory database for testing
    engine = create_engine("sqlite:///:memory:")
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    # Create tables
    Base.metadata.create_all(bind=engine)
    
    db = SessionLocal()
    try:
        # Add a test user
        user_id = uuid.uuid4()
        user = User(
            id=user_id,
            username="test_user",
            email="test@example.com",
            hashed_password="hashed_pwd"
        )
        db.add(user)
        db.commit()
        
        # Query the user
        db_user = db.query(User).filter(User.username == "test_user").first()
        assert db_user is not None
        assert db_user.email == "test@example.com"
        assert db_user.id == user_id
    finally:
        db.close()

def test_document_model():
    engine = create_engine("sqlite:///:memory:")
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    
    db = SessionLocal()
    try:
        doc_id = uuid.uuid4()
        doc = Document(
            id=doc_id,
            file_hash="fake_hash",
            file_name="test.pdf",
            file_path="data/sources/test.pdf",
            status="pending",
            chunk_count=0
        )
        db.add(doc)
        db.commit()
        
        db_doc = db.query(Document).filter(Document.file_name == "test.pdf").first()
        assert db_doc is not None
        assert db_doc.file_hash == "fake_hash"
        assert db_doc.id == doc_id
    finally:
        db.close()
