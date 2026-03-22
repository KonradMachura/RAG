import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session, joinedload

from backend.api.schemas import DocumentCreate, UserDocumentResponse, DocumentUpdate
from backend.db.models import User, Document, UserDocument
from backend.db.database import get_db
from config import config as cfg

# TODO
#  dodamć tabele asocjacyjną USER_DOCUMENTS
#  Wprowadzienie hashu pliku, dzieki temu lepiej sprawdzamy powtórki
#  oraz jak dwoch userow wgra ta sama nazwe, ale inna ksiazke to nie będzie błędu przez
#  jak sie zrobi część z ksiazkami to dodajemy baze danych z historia czatów, a potem dodajemy wiele userow
#  czy jesli wielu userów moze dodać inny dokument ale o takiej samej nazwie to czy nie powinniśmy zapisywać w
#  sources/books/ hasha dokumentu zamiast nazwy ???
#  refactor utils read_docs()



app = FastAPI()

def get_current_user(db: Session = Depends(get_db)) -> type[User]:
    user = db.query(User).filter(User.username == "Konrad").first()
    if not user:
        raise HTTPException(status_code=401, detail="User doesn't exist")
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
    if UserDocument.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="No access to this file")

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



    input_path = Path(document_in.file_path)
    try:
        relative_path = input_path.relative_to(cfg.BASE_DIR).as_posix()
    except ValueError:
        relative_path = input_path.as_posix()

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
        db: Session = Depends(get_db)
):
    user_doc_link = (
        db.query(UserDocument)
        .options(joinedload(UserDocument.document))
        .filter(UserDocument.document_id == doc_id)
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