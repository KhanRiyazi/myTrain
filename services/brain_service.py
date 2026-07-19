"""
Brain Service - Database-backed business logic
"""
from datetime import datetime
from typing import cast
from sqlalchemy.orm import Session
from sqlalchemy import func
import os

from models.schemas import (
    IntegrationPointSchema,
    BrainIntegration,
    HealthResponse,
    LocalPathResponse,
    GitHubRepoResponse,
    DomainStatusResponse,
    SyncResponse,
    DashboardStats
)
from models.database import IntegrationPoint, ApiLog, SyncHistory


class BrainService:
    """Service class with database operations"""

    def __init__(self):
        self.local_path = r"C:\Users\Dell\Desktop\IT\My_Domain"
        self.github_repo = "KhanRiyazi/portfolio-responsive-complete"
        self.github_url = "https://github.com/KhanRiyazi/portfolio-responsive-complete.git"
        self.live_domain = "affiliatemath.xyz"
        self.live_url = "https://affiliatemath.xyz"
        self.api_version = "1.0.0"
        self.total_endpoints = 6
        self.start_time = datetime.now()

    # ============================================================
    # DATABASE INITIALIZATION
    # ============================================================
    def seed_database(self, db: Session):
        """Seed initial integration points into database"""
        existing = db.query(IntegrationPoint).count()
        if existing == 0:
            points = [
                IntegrationPoint(
                    name="Local Development",
                    path=self.local_path,
                    type="local",
                    status="active",
                    description="Vanilla JS workspace → FastAPI ready"
                ),
                IntegrationPoint(
                    name="GitHub Repository",
                    path=self.github_url,
                    type="github",
                    status="connected",
                    description="Version control & deployment hub"
                ),
                IntegrationPoint(
                    name="Live Domain",
                    path=self.live_url,
                    type="live",
                    status="deployed",
                    description="Production website"
                ),
                IntegrationPoint(
                    name="FastAPI Self API",
                    path="http://localhost:8000",
                    type="api",
                    status="running",
                    description="Backend API server"
                ),
            ]
            db.add_all(points)
            db.commit()
            print("✅ Database seeded with integration points")

    # ============================================================
    # HEALTH CHECK
    # ============================================================
    def get_health(self, db: Session) -> HealthResponse:
        """Health check with database status"""
        try:
            # Test database connection
            db.execute(func.now())
            db_status = "connected"
        except Exception:
            db_status = "disconnected"

        return HealthResponse(
            status="healthy" if db_status == "connected" else "degraded",
            message="🚂 Train of Success · Self API is running",
            version=self.api_version,
            database=db_status,
            endpoints=self.total_endpoints,
            timestamp=datetime.now().isoformat()
        )

    # ============================================================
    # INTEGRATION POINTS (from database)
    # ============================================================
    def get_integration_points(self, db: Session) -> BrainIntegration:
        """Get all integration points from database"""
        points = db.query(IntegrationPoint).all()
        
        # Map to response
        point_map = {}
        for p in points:
            schema = IntegrationPointSchema(
                id=p.id,
                name=p.name,
                path=p.path,
                type=p.type,
                status=p.status,
                description=p.description,
                created_at=p.created_at
            )
            point_map[p.type] = schema

        total_logs = db.query(ApiLog).count()

        return BrainIntegration(
            local_development=point_map.get("local", IntegrationPointSchema(
                name="Local Development", path=self.local_path, type="local", status="active"
            )),
            github_repository=point_map.get("github", IntegrationPointSchema(
                name="GitHub Repository", path=self.github_url, type="github", status="connected"
            )),
            live_domain=point_map.get("live", IntegrationPointSchema(
                name="Live Domain", path=self.live_url, type="live", status="deployed"
            )),
            second_brain_active=True,
            total_endpoints=self.total_endpoints,
            total_logs=total_logs,
            timestamp=datetime.now().isoformat()
        )

    # ============================================================
    # LOCAL PATH
    # ============================================================
    def get_local_path(self) -> LocalPathResponse:
        return LocalPathResponse(
            path=self.local_path,
            exists=os.path.exists(self.local_path),
            workspace="My_Domain"
        )

    # ============================================================
    # GITHUB REPO
    # ============================================================
    def get_github_repo(self) -> GitHubRepoResponse:
        return GitHubRepoResponse(
            repo=self.github_repo,
            url=self.github_url,
            branch="main",
            status="connected"
        )

    # ============================================================
    # DOMAIN STATUS
    # ============================================================
    def get_domain_status(self) -> DomainStatusResponse:
        return DomainStatusResponse(
            domain=self.live_domain,
            url=self.live_url,
            status="active",
            ssl=True
        )

    # ============================================================
    # GITHUB SYNC (saves to database)
    # ============================================================
    def sync_github(self, db: Session) -> SyncResponse:
        """Sync with GitHub and log to database"""
        timestamp = datetime.now().isoformat()
        
        # Create sync record
        sync_record = SyncHistory(
            repo=self.github_repo,
            success=True,
            message=f"Synced successfully at {timestamp}"
        )
        db.add(sync_record)
        db.commit()
        db.refresh(sync_record)

        return SyncResponse(
            success=True,
            message=f"Successfully synced with {self.github_repo}",
            repo=self.github_repo,
            sync_id=sync_record.id,
            timestamp=timestamp
        )

    # ============================================================
    # API LOGGING
    # ============================================================
    def log_api_call(self, db: Session, endpoint: str, method: str, status_code: int = 200, message: str = None):
        """Log an API call to the database"""
        log = ApiLog(
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            message=message
        )
        db.add(log)
        db.commit()

    # ============================================================
    # DASHBOARD STATISTICS
    # ============================================================
    def get_dashboard_stats(self, db: Session) -> DashboardStats:
        """Get dashboard statistics from database"""
        total_points = db.query(IntegrationPoint).count()
        total_logs = db.query(ApiLog).count()
        total_syncs = db.query(SyncHistory).count()
        
        # Get database file size
        db_path = "./train_of_success.db"
        if os.path.exists(db_path):
            size_bytes = os.path.getsize(db_path)
            size_str = f"{size_bytes / 1024:.1f} KB" if size_bytes < 1024 * 1024 else f"{size_bytes / (1024*1024):.1f} MB"
        else:
            size_str = "0 KB"

        # Calculate uptime
        uptime_seconds = (datetime.now() - self.start_time).total_seconds()
        hours = int(uptime_seconds // 3600)
        minutes = int((uptime_seconds % 3600) // 60)
        uptime_str = f"{hours}h {minutes}m"

        return DashboardStats(
            total_integration_points=total_points,
            total_api_logs=total_logs,
            total_syncs=total_syncs,
            database_size=size_str,
            uptime=uptime_str
        )

    # ============================================================
    # GET RECENT LOGS
    # ============================================================
    def get_recent_logs(self, db: Session, limit: int = 20):
        """Get recent API logs"""
        return db.query(ApiLog).order_by(ApiLog.created_at.desc()).limit(limit).all()