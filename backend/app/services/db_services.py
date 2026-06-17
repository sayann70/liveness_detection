import datetime
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.models import Employee, ActivityLog, DailySummary
from app.config import settings

def format_seconds_to_hhmmss(seconds: int) -> str:
    """Converts a duration in seconds into HH:MM:SS format."""
    if seconds < 0:
        seconds = 0
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def get_or_create_default_employee(db: Session) -> Employee:
    """Ensures default employee profiles (Ankur Bag and Sayan Sarkar) exist in the database."""
    # Ensure Employee 1 is Ankur Bag
    employee = db.query(Employee).filter(Employee.employee_id == settings.DEFAULT_EMPLOYEE_ID).first()
    if not employee:
        employee = Employee(
            employee_id=settings.DEFAULT_EMPLOYEE_ID,
            name="Ankur Bag",
            department="Engineering"
        )
        db.add(employee)
        db.commit()
        db.refresh(employee)
    elif employee.name == "John Doe":
        employee.name = "Ankur Bag"
        db.commit()
        db.refresh(employee)

    # Ensure Employee 2 is Sayan Sarkar
    employee2 = db.query(Employee).filter(Employee.employee_id == 2).first()
    if not employee2:
        employee2 = Employee(
            employee_id=2,
            name="Sayan Sarkar",
            department="Engineering"
        )
        db.add(employee2)
        db.commit()
        db.refresh(employee2)

    return employee

def resolve_orphaned_sessions(db: Session):
    """
    Finds and closes any un-ended state sessions (end_time is NULL) from previous runs.
    Closes them as 0-duration entries with annotations to avoid inflating active times.
    """
    orphans = db.query(ActivityLog).filter(ActivityLog.end_time == None).all()
    if orphans:
        print(f"[DB Service] Resolving {len(orphans)} orphaned sessions...")
        for o in orphans:
            o.end_time = o.start_time
            o.duration_seconds = 0
            o.notes = "Session orphaned and automatically closed on restart."
        db.commit()

def start_new_state_session(
    db: Session, 
    employee_id: int, 
    state: str, 
    confidence: float, 
    raw_score: float = 0.0,
    smoothed_score: float = 0.0,
    transition_reason: Optional[str] = None,
    notes: Optional[str] = None
) -> ActivityLog:
    """
    Transitions the subject's state by:
      1. Finding and closing the currently open session (setting end_time and duration).
      2. Inserting a new open session for the new state with tracking scores.
      3. Triggering a recalculation of daily stats.
    """
    now = datetime.datetime.now()
    
    # 1. Close open sessions
    open_sessions = db.query(ActivityLog).filter(
        ActivityLog.employee_id == employee_id,
        ActivityLog.end_time == None
    ).all()
    
    for sess in open_sessions:
        sess.end_time = now
        sess.duration_seconds = max(0, int((now - sess.start_time).total_seconds()))
    
    # 2. Insert new session with scores and reasons
    new_log = ActivityLog(
        employee_id=employee_id,
        state=state,
        start_time=now,
        end_time=None,
        duration_seconds=0,
        confidence=confidence,
        raw_score=raw_score,
        smoothed_score=smoothed_score,
        transition_reason=transition_reason,
        notes=notes
    )
    db.add(new_log)
    db.commit()
    db.refresh(new_log)
    
    # 3. Recalculate daily summary for today
    recalculate_daily_summary(db, employee_id, datetime.date.today())
    return new_log

def close_active_session(db: Session, employee_id: int) -> None:
    """
    Gracefully closes any open state session (e.g., during server shutdown).
    """
    now = datetime.datetime.now()
    open_sessions = db.query(ActivityLog).filter(
        ActivityLog.employee_id == employee_id,
        ActivityLog.end_time == None
    ).all()
    
    for sess in open_sessions:
        sess.end_time = now
        sess.duration_seconds = max(0, int((now - sess.start_time).total_seconds()))
        sess.notes = "Session closed gracefully on server stop."
    
    db.commit()
    recalculate_daily_summary(db, employee_id, datetime.date.today())

def recalculate_daily_summary(db: Session, employee_id: int, target_date: datetime.date) -> Optional[DailySummary]:
    """
    Aggregates durations (WORKING, IDLE, ABSENT) for an employee on a specific date.
    Correctly partitions sessions crossing boundaries and updates the daily_summary table.
    """
    now = datetime.datetime.now()
    start_of_day = datetime.datetime.combine(target_date, datetime.time.min)
    end_of_day = datetime.datetime.combine(target_date, datetime.time.max)
    
    # Limit aggregation upper boundary to current time if target date is today
    end_of_time = now if target_date == now.date() else end_of_day
    
    # Query all sessions overlapping with target date interval
    logs = db.query(ActivityLog).filter(
        ActivityLog.employee_id == employee_id,
        ActivityLog.start_time <= end_of_time,
        (ActivityLog.end_time == None) | (ActivityLog.end_time >= start_of_day)
    ).all()
    
    durations = {"WORKING": 0, "IDLE": 0, "ABSENT": 0}
    
    for log in logs:
        # Calculate overlap segment boundaries
        seg_start = max(log.start_time, start_of_day)
        seg_end = min(log.end_time or now, end_of_time)
        
        overlap_seconds = int((seg_end - seg_start).total_seconds())
        if overlap_seconds > 0:
            durations[log.state] += overlap_seconds
            
    w_sec = durations["WORKING"]
    i_sec = durations["IDLE"]
    a_sec = durations["ABSENT"]
    total_sec = w_sec + i_sec + a_sec
    
    # Formula: Working Seconds / (Working + Idle + Absent) * 100
    productivity_score = 0.0
    if total_sec > 0:
        productivity_score = round((w_sec / total_sec) * 100.0, 1)
        
    summary = db.query(DailySummary).filter(
        DailySummary.employee_id == employee_id,
        DailySummary.date == target_date
    ).first()
    
    try:
        if not summary:
            summary = DailySummary(
                employee_id=employee_id,
                date=target_date,
                working_seconds=w_sec,
                idle_seconds=i_sec,
                absent_seconds=a_sec,
                productivity_score=productivity_score
            )
            db.add(summary)
        else:
            summary.working_seconds = w_sec
            summary.idle_seconds = i_sec
            summary.absent_seconds = a_sec
            summary.productivity_score = productivity_score
        db.commit()
    except Exception:
        db.rollback()
        summary = db.query(DailySummary).filter(
            DailySummary.employee_id == employee_id,
            DailySummary.date == target_date
        ).first()
        if summary:
            summary.working_seconds = w_sec
            summary.idle_seconds = i_sec
            summary.absent_seconds = a_sec
            summary.productivity_score = productivity_score
            db.commit()
        else:
            raise

    db.refresh(summary)
    return summary

def get_live_status(db: Session, employee_id: int, live_cv_metrics: dict) -> dict:
    """
    Builds the real-time operational metrics payload, fetching boundaries,
    today's aggregated seconds, status durations, and developer diagnostics indicators.
    """
    employee = db.query(Employee).filter(Employee.employee_id == employee_id).first()
    if not employee:
        return {}
        
    # Recalculate today's summary to guarantee real-time figures
    summary = recalculate_daily_summary(db, employee_id, datetime.date.today())
    
    # Retrieve latest log record
    latest_log = db.query(ActivityLog).filter(
        ActivityLog.employee_id == employee_id
    ).order_by(ActivityLog.start_time.desc()).first()
    
    if latest_log:
        status = latest_log.state
        confidence = latest_log.confidence
        if latest_log.end_time is None:
            time_in_state = int((datetime.datetime.now() - latest_log.start_time).total_seconds())
        else:
            time_in_state = latest_log.duration_seconds
    else:
        status = "ABSENT"
        confidence = 0.0
        time_in_state = 0
        
    # Bounds for the monitored shift
    start_of_day = datetime.datetime.combine(datetime.date.today(), datetime.time.min)
    first_log = db.query(ActivityLog).filter(
        ActivityLog.employee_id == employee_id,
        ActivityLog.start_time >= start_of_day
    ).order_by(ActivityLog.start_time.asc()).first()
    
    first_act = first_log.start_time.strftime("%H:%M:%S") if first_log else None
    last_act = latest_log.start_time.strftime("%H:%M:%S") if latest_log else None
    
    w_sec = summary.working_seconds if summary else 0
    i_sec = summary.idle_seconds if summary else 0
    a_sec = summary.absent_seconds if summary else 0
    total_sec = w_sec + i_sec + a_sec
    productivity = summary.productivity_score if summary else 0.0
    
    return {
        "employee_id": employee.employee_id,
        "name": employee.name,
        "department": employee.department,
        "status": status,
        "confidence": confidence,
        "time_in_state": format_seconds_to_hhmmss(time_in_state),
        "working_time": format_seconds_to_hhmmss(w_sec),
        "idle_time": format_seconds_to_hhmmss(i_sec),
        "absent_time": format_seconds_to_hhmmss(a_sec),
        "productivity_score_today": productivity,
        "first_activity": first_act,
        "last_activity": last_act,
        "total_monitored_time": format_seconds_to_hhmmss(total_sec),
        
        # Merge diagnostic metrics passed from runtime
        "raw_score": live_cv_metrics.get("raw_score", 0.0),
        "smoothed_score": live_cv_metrics.get("smoothed_score", 0.0),
        "movement_threshold": live_cv_metrics.get("movement_threshold", 0.50),
        "idle_countdown": live_cv_metrics.get("idle_countdown", 0),
        "working_countdown": live_cv_metrics.get("working_countdown", 0),
        "epsilon_filter": live_cv_metrics.get("epsilon_filter", 0.0015)
    }
