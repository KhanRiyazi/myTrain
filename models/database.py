"""
SQLAlchemy Database Models
"""
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text
from sqlalchemy.sql import func
from database import Base


class IntegrationPoint(Base):
    """Integration points table"""
    __tablename__ = "integration_points"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    path = Column(String(500), nullable=False)
    type = Column(String(50), nullable=False)  # local, github, live
    status = Column(String(50), default="active")
    description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self):
        return f"<IntegrationPoint {self.name}: {self.type}>"


class ApiLog(Base):
    """API request logs table"""
    __tablename__ = "api_logs"

    id = Column(Integer, primary_key=True, index=True)
    endpoint = Column(String(200), nullable=False)
    method = Column(String(10), nullable=False)
    status_code = Column(Integer, default=200)
    message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<ApiLog {self.method} {self.endpoint}>"


class SyncHistory(Base):
    """GitHub sync history table"""
    __tablename__ = "sync_history"

    id = Column(Integer, primary_key=True, index=True)
    repo = Column(String(200), nullable=False)
    success = Column(Boolean, default=True)
    message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<SyncHistory {self.repo}: {self.success}>"