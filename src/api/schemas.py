from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr
from uuid import UUID

class UserBase(BaseModel):
    id: UUID
    username: str
    email: EmailStr
    created_at: datetime
    model_config = {"from_attributes": True}

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: str | None = None

class DocumentCreate(BaseModel):
    file_name: str
    file_hash: str

class DocumentGlobal(BaseModel):
    id: UUID
    file_hash: str
    file_name: str
    status: str
    chunk_count: int
    system_upload_date: datetime

    model_config = {"from_attributes": True}

class UserDocumentResponse(BaseModel):
    user_id: UUID
    document_id: UUID
    added_at: datetime

    document: DocumentGlobal

    model_config = {"from_attributes": True}

class DocumentUpdate(BaseModel):
    chunk_count: int

class UserWithDocuments(UserBase):
    documents: Optional[list[UserDocumentResponse]] = None

class QueryRequest(BaseModel):
    query: str
    document_ids: list[UUID]

class SourceDocument(BaseModel):
    content: str
    metadata: dict

class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]
