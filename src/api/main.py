import uuid
from pathlib import Path
from datetime import timedelta

from fastapi import FastAPI, HTTPException, Depends
from starlette import status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session, joinedload
import jwt
import transformers
transformers.logging.set_verbosity_error()

from src.api.schemas import DocumentCreate, UserDocumentResponse, DocumentUpdate, UserCreate, Token, UserBase
from src.database.models import User, Document, UserDocument
from src.database.connection import get_db
from src.core import security
from src.core import config as cfg

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI()

@app.post("/register", response_model=UserBase)
def register_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter((User.username == user.username) | (User.email == user.email)).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username or email already registered")
    hashed_password = security.get_password_hash(user.password)
    new_user = User(username=user.username, email=user.email, hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

@app.post("/token", response_model=Token)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not security.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=security.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = security.create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, security.SECRET_KEY, algorithms=[security.ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except jwt.InvalidTokenError:
        raise credentials_exception
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    return user


@app.get("/documents", response_model=list[UserDocumentResponse])
def get_my_documents(
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    my_documents = (
        db.query(UserDocument)
        .options(joinedload(UserDocument.document))
        .filter(UserDocument.user_id == current_user.id)
        .all()
    )
    return my_documents


@app.get("/documents/{doc_id}", response_model=UserDocumentResponse)
def get_document_by_id(
        doc_id: uuid.UUID,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    document = (
        db.query(UserDocument)
        .options(joinedload(UserDocument.document))
        .filter(UserDocument.user_id == current_user.id, UserDocument.document_id == doc_id)
        .first()
    )

    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    return document


@app.post("/document", response_model=UserDocumentResponse)
def add_document(
        document_in: DocumentCreate,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):

    existing_doc = db.query(Document).filter(Document.file_hash == document_in.file_hash).first()
    if existing_doc:

        existing_link = (
            db.query(UserDocument)
            .filter(UserDocument.user_id == current_user.id, UserDocument.document_id == existing_doc.id)
            .first()
        )

        if existing_link:
            raise HTTPException(status_code=400, detail="Document already exists")

        new_link = UserDocument(
            user_id=current_user.id,
            document_id=existing_doc.id
        )
        db.add(new_link)
        db.commit()
        db.refresh(new_link)
        return new_link

    target_path = cfg.DOCUMENTS_RAW_DIR / f"{document_in.file_hash}.pdf"
    relative_path = target_path.relative_to(cfg.BASE_DIR).as_posix()

    new_document = Document(
        file_name=document_in.file_name,
        file_path=relative_path,
        file_hash=document_in.file_hash,
        status="pending",
        chunk_count=0
    )
    db.add(new_document)
    db.flush()

    new_link = UserDocument(
        user_id=current_user.id,
        document_id=new_document.id
    )
    db.add(new_link)
    db.commit()
    db.refresh(new_link)
    return new_link

@app.patch("/document/{doc_id}", response_model=UserDocumentResponse)
def update_document(
        doc_id: uuid.UUID,
        update_data: DocumentUpdate,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    user_doc_link = (
        db.query(UserDocument)
        .options(joinedload(UserDocument.document))
        .filter(UserDocument.document_id == doc_id, UserDocument.user_id == current_user.id)
        .first()
    )

    if not user_doc_link:
        raise HTTPException(status_code=404, detail="Document not found")

    user_doc_link.document.chunk_count = update_data.chunk_count
    user_doc_link.document.status = "processed"
    db.commit()
    db.refresh(user_doc_link)
    return user_doc_link

@app.delete("/document/{doc_id}", response_model=UserDocumentResponse)
def delete_document(
        doc_id: uuid.UUID,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    existing_link = (
        db.query(UserDocument)
        .options(joinedload(UserDocument.document))
        .filter(UserDocument.user_id == current_user.id, UserDocument.document_id == doc_id)
        .first()
    )

    if not existing_link:
        raise HTTPException(status_code=404, detail="Document not found")

    global_document = existing_link.document

    db.delete(existing_link)
    db.flush()

    any_other_link = (
        db.query(UserDocument)
        .options(joinedload(UserDocument.document))
        .filter(UserDocument.document_id == doc_id)
        .first()
    )

    if not any_other_link:
        db.delete(global_document)

    db.commit()
    return existing_link
