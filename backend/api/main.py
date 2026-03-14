import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session

from backend.api.schemas import DocumentCreate, DocumentResponse, DocumentUpdate
from backend.db.models import User, Document
from backend.db.database import get_db
from config import config as cfg

# TODO
#  dodamć tabele asocjacyjną USER_DOCUMENTS
#  Wprowadzienie hashu pliku, dzieki temu lepiej sprawdzamy powtórki
#  oraz jak dwoch userow wgra ta sama nazwe, ale inna ksiazke to nie będzie błędu przez
#  sprawdzić, czy wszedzie scieżki sie zgadzaja
#  zrobić usuwanie i wyszukiwanie po id tak jak w api
#  jak sie zrobi część z ksiazkami to dodajemy baze danych z historia czatów, a potem dodajemy wiele userow



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
    doc_already_exists = db.query(Document).filter(
        Document.file_name == document_in.file_name,
        Document.owner_id == current_user.id
    ).first()

    if doc_already_exists:
        raise HTTPException(status_code=400, detail="Document already exists")

    input_path = Path(document_in.file_path)
    try:
        relative_path = input_path.relative_to(cfg.BASE_DIR).as_posix()
    except ValueError:
        relative_path = input_path.as_posix()

    new_document = Document(
        file_name=document_in.file_name,
        file_path=relative_path,
        owner_id=current_user.id,
        chunk_count=0
    )

    db.add(new_document)
    db.commit()
    db.refresh(new_document)
    return new_document

@app.patch("/document/{doc_id}", response_model=DocumentResponse)
def update_document(
        doc_id: uuid.UUID,
        update_data: DocumentUpdate,
        db: Session = Depends(get_db)
):
    document = db.query(Document).filter(Document.id == doc_id).first()

    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    document.chunk_count = update_data.chunk_count
    document.status = "processed"
    db.commit()
    db.refresh(document)
    return document

@app.delete("/document/{doc_id}", response_model=DocumentResponse)
def delete_document(
        doc_id: uuid.UUID,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    document = db.query(Document).filter(Document.id == doc_id).first()

#TODO dodać usuwanie z folderu konkretnego albo tutaj albo od strony streamlite
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    if document.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="No access to this file")

    db.delete(document)
    db.commit()

    return document