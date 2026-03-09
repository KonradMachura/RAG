import uuid
from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session

from backend.api.schemas import UserBase, UserWithDocuments, DocumentCreate, DocumentResponse
from backend.db.models import User, Document
from backend.db.database import SessionLocal, get_db

app = FastAPI()

def get_current_user(db: Session = Depends(get_db)) -> type[User]:
    user = db.query(User).filter(User.username == "Konrad").first()
    if not user:
        raise HTTPException(status_code=401, detail="User doesn't exist")
    return user


@app.get("/documents", response_model=list[DocumentResponse])
def get_my_documents(
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    my_documents = db.query(Document).filter(Document.owner_id == current_user.id).all()
    return my_documents


@app.get("/documents/{doc_id}", response_model=DocumentResponse)
def get_document_by_id(
        doc_id: uuid.UUID,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    document = db.query(Document).filter(Document.id == doc_id).first()

    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    if document.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="No access to this file")

    return document


@app.post("/document", response_model=DocumentResponse)
def add_document(
        document_in: DocumentCreate,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    new_document = Document(
        file_name=document_in.file_name,
        file_path=document_in.file_path,
        owner_id=current_user.id,
        chunk_count=0
    )

    db.add(new_document)
    db.commit()
    db.refresh(new_document)
    return new_document


@app.delete("/documents/{doc_id}", response_model=DocumentResponse)
def delete_document(
        doc_id: uuid.UUID,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    document = db.query(Document).filter(Document.id == doc_id).first()

    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    if document.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="No access to this file")

    db.delete(document)
    db.commit()

    return document