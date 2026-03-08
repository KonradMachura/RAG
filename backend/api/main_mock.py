import uuid
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends

from backend.db.models import User, Document
from backend.api.schemas import DocumentCreate, DocumentResponse

app = FastAPI()

@app.get("/")
def root():
    return {"Hello": "World"}

STARTING_DOC_ID = uuid.uuid4()
STARTING_USER_ID = uuid.uuid4()

user = User(id=STARTING_USER_ID, username="Konrad")
documents = [
    Document(id=STARTING_DOC_ID,
             file_name="hobbit.pdf",
             file_path="data/sources/books/hobbit.pdf",
             status="pending",
             upload_date=datetime.now(),
             chunk_count=120,
             owner_id=STARTING_USER_ID,
             owner=user
    )
]

def get_current_user() -> User:
    return user

@app.get("/documents", response_model=list[DocumentResponse])
def get_my_documents(current_user: User = Depends(get_current_user)):
    my_documents = [doc for doc in documents if doc.owner_id == current_user.id]
    return my_documents

@app.get("/documents/{doc_id}", response_model=DocumentResponse)
def get_document_by_id(doc_id: uuid.UUID, current_user: User = Depends(get_current_user)):
    for document in documents:
        if document.id == doc_id:
            if document.owner_id != current_user.id:
                raise HTTPException(status_code=403, detail="No access to this file")

            return document

    raise HTTPException(status_code=404, detail="Document not found")

@app.post("/document", response_model=DocumentResponse)
def add_document(document: DocumentCreate):
    new_doc_id = uuid.uuid4()

    new_document = Document(
        id=new_doc_id,
        file_name=document.file_name,
        file_path=document.file_path,
        status="pending",
        upload_date=datetime.now(),
        chunk_count=120,
        owner_id=STARTING_USER_ID,
        owner=user
    )
    user.documents.append(new_document)
    documents.append(new_document)
    return new_document

@app.delete("/documents/{doc_id}", response_model=DocumentResponse)
def delete_document(doc_id: uuid.UUID, current_user: User = Depends(get_current_user)):
    for document in documents:
        if document.id == doc_id:
            if document.owner_id != current_user.id:
                raise HTTPException(status_code=403, detail="No access to this file")

            documents.remove(document)
            return document

    raise HTTPException(status_code=404, detail="Document not found")

