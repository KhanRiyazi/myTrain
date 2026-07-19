from pydantic import BaseModel
from typing import Optional
from datetime import datetime


# ============================================================
# API Response Schemas
# ============================================================
class IntegrationPointSchema(BaseModel):
    """Single integration point"""
    id: Optional[int] = None
    name: str
    path: str
    type: str
    status: str
    description: Optional[str] = None
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class BrainIntegration(BaseModel):
    """Complete second brain integration data"""
    local_development: IntegrationPointSchema
    github_repository: IntegrationPointSchema
    live_domain: IntegrationPointSchema
    second_brain_active: bool = True
    total_endpoints: int = 5
    total_logs: int = 0
    timestamp: str = ""


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    message: str
    version: str
    database: str
    endpoints: int
    timestamp: str


class LocalPathResponse(BaseModel):
    """Local path response"""
    path: str
    exists: bool
    workspace: str


class GitHubRepoResponse(BaseModel):
    """GitHub repository response"""
    repo: str
    url: str
    branch: str
    status: str


class DomainStatusResponse(BaseModel):
    """Domain status response"""
    domain: str
    url: str
    status: str
    ssl: bool


class SyncResponse(BaseModel):
    """GitHub sync response"""
    success: bool
    message: str
    repo: str
    sync_id: Optional[int] = None
    timestamp: str


class ApiLogSchema(BaseModel):
    """API log entry"""
    id: int
    endpoint: str
    method: str
    status_code: int
    message: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class DashboardStats(BaseModel):
    """Dashboard statistics"""
    total_integration_points: int
    total_api_logs: int
    total_syncs: int
    database_size: str
    uptime: str