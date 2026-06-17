from pydantic import BaseModel
from datetime import datetime, date
from typing import List, Optional

# Employee Schemas
class EmployeeBase(BaseModel):
    name: str
    department: Optional[str] = "Engineering"

class EmployeeCreate(EmployeeBase):
    pass

class EmployeeResponse(EmployeeBase):
    employee_id: int
    created_at: datetime

    class Config:
        from_attributes = True

# Activity Log (Session Segment) Schemas
class ActivityLogBase(BaseModel):
    state: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: int = 0
    confidence: float
    raw_score: float = 0.0
    smoothed_score: float = 0.0
    transition_reason: Optional[str] = None
    notes: Optional[str] = None

class ActivityLogResponse(ActivityLogBase):
    id: int
    employee_id: int
    created_at: datetime
    duration_formatted: str  # e.g., "00:15:22"

    class Config:
        from_attributes = True

# Daily Summary Schemas (Seconds-based and formatted)
class DailySummaryResponse(BaseModel):
    id: int
    employee_id: int
    date: date
    working_seconds: int
    idle_seconds: int
    absent_seconds: int
    working_time: str            # "HH:MM:SS"
    idle_time: str               # "HH:MM:SS"
    absent_time: str             # "HH:MM:SS"
    total_monitored_time: str    # "HH:MM:SS"
    productivity_score: float

    class Config:
        from_attributes = True

# Live Status Schema with diagnostic variables
class LiveStatusResponse(BaseModel):
    employee_id: int
    name: str
    department: str
    status: str
    confidence: float
    time_in_state: str           # "HH:MM:SS"
    working_time: str            # "HH:MM:SS"
    idle_time: str               # "HH:MM:SS"
    absent_time: str             # "HH:MM:SS"
    productivity_score_today: float
    first_activity: Optional[str] = None  # "HH:MM:SS"
    last_activity: Optional[str] = None   # "HH:MM:SS"
    total_monitored_time: str    # "HH:MM:SS"
    
    # Diagnostics Debug Parameters
    raw_score: float
    smoothed_score: float
    movement_threshold: float
    idle_countdown: int
    working_countdown: int
    epsilon_filter: float
