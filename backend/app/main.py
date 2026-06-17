import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import time

from app.api.routes import router as api_router
from app.database import engine, Base, SessionLocal
from app.services.cv_pipeline import cv_monitor
from app.services.db_services import get_or_create_default_employee

app = FastAPI(
    title="Employee Activity Monitoring System API",
    description="Backend API for Camera-based Employee Activity Tracking POC",
    version="1.0.0"
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Router under /api namespace
app.include_router(api_router, prefix="/api")

# Startup event hook
@app.on_event("startup")
def startup_event():
    print("[Server] Initializing database tables...")
    # Attempt connection retries in case database server is starting up
    retries = 5
    while retries > 0:
        try:
            Base.metadata.create_all(bind=engine)
            db = SessionLocal()
            # Seed default employee
            get_or_create_default_employee(db)
            db.close()
            print("[Server] Database connection established and tables validated.")
            break
        except Exception as e:
            print(f"[Server] Database connection failed: {e}. Retrying in 2 seconds... ({retries} left)")
            time.sleep(2)
            retries -= 1
            if retries == 0:
                print("[Server] CRITICAL: Could not connect to MySQL database. Server starting with DB errors.")

    print("[Server] Starting Computer Vision Monitoring engine...")
    cv_monitor.start()

# Shutdown event hook
@app.on_event("shutdown")
def shutdown_event():
    print("[Server] Stopping Computer Vision Monitoring engine...")
    cv_monitor.stop()

# Serve static frontend files
# Note: Mount at bottom so that standard API routes take precedence
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../frontend"))
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
    print(f"[Server] Static frontend successfully mounted from: {frontend_dir}")
else:
    print(f"[Server] WARNING: Frontend directory not found at: {frontend_dir}. Static web server disabled.")
