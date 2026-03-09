import uuid
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config import config as cfg
from backend.db.models import Base, Document, User

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

    # Tworzymy testowy dokument
    doc = Document(
        id=uuid.uuid4(),
        file_name="hobbit.pdf",
        file_path="data/sources/books/hobbit.pdf",
        owner_id=STARTING_USER_ID
    )
    db.add(user)
    db.add(doc)
    db.commit()
    print("Database filled.")
    db.close()

if __name__ == "__main__":
    create_tables()
    seed_database()