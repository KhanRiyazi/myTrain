"""
Database Configuration
SQLite database for Self API
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os

# Database file location
DATABASE_URL = "sqlite:///./train_of_success.db"

# Create engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}  # Needed for SQLite
)

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


def get_db():
    """Dependency that provides database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database - create all tables"""
    Base.metadata.create_all(bind=engine)
    print("✅ Database initialized: train_of_success.db")