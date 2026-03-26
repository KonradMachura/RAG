import uuid
import hashlib
from datetime import datetime, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.core import config as cfg
from src.database.models import Base, Document, User, UserDocument

engine = create_engine(cfg.DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def create_tables():
    Base.metadata.create_all(bind=engine)
    print("Tables created")


def get_db():
    """
    A function that ‘rents’ a database session for a specific request in FastAPI,
    and after everything is done (when the code in the endpoint is executed), it safely closes the connection.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def seed_database():
    db = SessionLocal()
    if db.query(User).filter(User.username == "Konrad").first():
        db.close()
        return

    STARTING_USER_ID = uuid.uuid4()
    user = User(
        id=STARTING_USER_ID,
        username="Konrad",
        email="konrad@bookipidia.com",
        hashed_password="fake_password_123"
    )

    fake_hash = hashlib.sha256(b"fake_hobbit_content").hexdigest()
    DOC_ID = uuid.uuid4()

    doc = Document(
        id=DOC_ID,
        file_hash=fake_hash,
        file_name="hobbit.pdf",
        file_path="data/sources/books/hobbit.pdf",
        status="pending",
        chunk_count=0,
        system_upload_date=datetime.now(timezone.utc)
    )

    user_doc = UserDocument(
        user_id=STARTING_USER_ID,
        document_id=DOC_ID
    )

    db.add(user)
    db.add(doc)
    db.add(user_doc)

    db.commit()
    db.close()


if __name__ == "__main__":
    create_tables()
    seed_database()