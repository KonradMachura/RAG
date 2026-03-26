from uuid import UUID, uuid4
from datetime import datetime
from sqlalchemy import String, DateTime, ForeignKey, Integer
from sqlalchemy.sql import func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, foreign


class Base(DeclarativeBase):
    pass

class UserDocument(Base):
    __tablename__ = "user_documents"

    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    document_id: Mapped[UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True)

    added_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    user: Mapped["User"] = relationship(back_populates="user_documents")
    document: Mapped["Document"] = relationship(back_populates="user_documents")

class User(Base):
    __tablename__ = "users"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    username: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    user_documents: Mapped[list["UserDocument"]] = relationship(back_populates="user", cascade="all, delete-orphan")

class Document(Base):
    __tablename__ = "documents"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    file_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)

    file_name: Mapped[str] = mapped_column(String(255))
    file_path: Mapped[str] = mapped_column(String(500))
    status: Mapped[str] = mapped_column(String(30), default="pending")
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    system_upload_date: Mapped[datetime] = mapped_column(DateTime, default=func.now())

    user_documents: Mapped[list["UserDocument"]] = relationship(back_populates="document", cascade="all, delete-orphan")