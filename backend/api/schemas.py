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

class DocumentCreate(BaseModel):
    file_name: str
    file_path: str

class DocumentResponse(BaseModel):
    id: UUID
    file_name: str
    file_path: str
    status: str
    chunk_count: int
    upload_date: datetime
    owner_id: UUID
    owner: Optional[UserBase] = None

    model_config = {"from_attributes": True}

class UserWithDocuments(UserBase):
    documents: Optional[list[DocumentResponse]] = None
