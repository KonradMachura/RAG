from __future__ import annotations
from datetime import datetime
from typing import Optional
from pydantic import BaseModel
from uuid import UUID

class UserResponse(BaseModel):
    id: UUID
    username: str

    model_config = {"from_attributes": True}

class DocumentCreate(BaseModel):
    file_name: str
    file_path: str

class DocumentResponse(BaseModel):
    id: UUID
    file_name: str
    file_path: str
    status: str
    upload_date: datetime
    owner_id: UUID
    owner: Optional[UserResponse] = None

    model_config = {"from_attributes": True}

class UserWithDocumentsResponse(UserResponse):
    documents: Optional[list[DocumentResponse]] = None