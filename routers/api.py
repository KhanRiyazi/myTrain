"""
API Routes - Database-powered endpoints
"""
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from database import get_db
from models.schemas import (
    BrainIntegration,
    HealthResponse,
    LocalPathResponse,
    GitHubRepoResponse,
    DomainStatusResponse,
    SyncResponse,
    DashboardStats,
    ApiLogSchema
)
from services.brain_service import BrainService
from typing import List

router = APIRouter()
brain_service = BrainService()


# ============================================================
# HEALTH CHECK (with database status)
# ============================================================
@router.get("/health", response_model=HealthResponse)
async def health_check(db: Session = Depends(get_db)):
    """🏥 Health check with database status"""
    brain_service.log_api_call(db, "/api/v1/health", "GET", 200, "Health check")
    return brain_service.get_health(db)


# ============================================================
# ALL INTEGRATION POINTS (from database)
# ============================================================
@router.get("/integration-points", response_model=BrainIntegration)
async def get_integration_points(db: Session = Depends(get_db)):
    """📦 Get all integration points from database"""
    brain_service.log_api_call(db, "/api/v1/integration-points", "GET", 200)
    return brain_service.get_integration_points(db)


# ============================================================
# LOCAL PATH
# ============================================================
@router.get("/local-path", response_model=LocalPathResponse)
async def get_local_path(db: Session = Depends(get_db)):
    """📁 Local development path"""
    brain_service.log_api_call(db, "/api/v1/local-path", "GET", 200)
    return brain_service.get_local_path()


# ============================================================
# GITHUB REPO
# ============================================================
@router.get("/github-repo", response_model=GitHubRepoResponse)
async def get_github_repo(db: Session = Depends(get_db)):
    """🐙 GitHub repository info"""
    brain_service.log_api_call(db, "/api/v1/github-repo", "GET", 200)
    return brain_service.get_github_repo()


# ============================================================
# DOMAIN STATUS
# ============================================================
@router.get("/domain-status", response_model=DomainStatusResponse)
async def get_domain_status(db: Session = Depends(get_db)):
    """🌐 Live domain status"""
    brain_service.log_api_call(db, "/api/v1/domain-status", "GET", 200)
    return brain_service.get_domain_status()


# ============================================================
# GITHUB SYNC (saves to database)
# ============================================================
@router.post("/sync-github", response_model=SyncResponse)
async def sync_github(db: Session = Depends(get_db)):
    """🔄 Sync with GitHub - logged in database"""
    brain_service.log_api_call(db, "/api/v1/sync-github", "POST", 200, "GitHub sync triggered")
    return brain_service.sync_github(db)


# ============================================================
# DASHBOARD STATISTICS
# ============================================================
@router.get("/stats", response_model=DashboardStats)
async def get_stats(db: Session = Depends(get_db)):
    """📊 Get dashboard statistics"""
    brain_service.log_api_call(db, "/api/v1/stats", "GET", 200)
    return brain_service.get_dashboard_stats(db)


# ============================================================
# RECENT API LOGS
# ============================================================
@router.get("/logs", response_model=List[ApiLogSchema])
async def get_logs(limit: int = 20, db: Session = Depends(get_db)):
    """📋 Get recent API logs"""
    logs = brain_service.get_recent_logs(db, limit)
    return logs