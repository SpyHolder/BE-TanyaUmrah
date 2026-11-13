from typing import Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel
from app.database.session import get_db
from app.database.models import AcademicEvent

router = APIRouter(prefix="/calendar", tags=["Calendar"])

class EventCreateRequest(BaseModel):
    title: str
    description: Optional[str] = None
    start_date: str  # YYYY-MM-DD
    end_date: Optional[str] = None
    category: Optional[str] = "akademik"

@router.get("/events")
def get_calendar_events(
    month: Optional[int] = Query(None, ge=1, le=12),
    year: Optional[int] = Query(None, ge=2020, le=2030),
    category: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Get academic calendar events. Can filter by month, year, or category.
    """
    query = db.query(AcademicEvent)
    if category:
        query = query.filter(AcademicEvent.category == category)
    
    events = query.order_by(AcademicEvent.start_date.asc()).all()

    # In-memory filter for month/year on start_date string (YYYY-MM-DD)
    filtered = []
    for ev in events:
        if year and not ev.start_date.startswith(str(year)):
            continue
        if month:
            month_str = f"-{month:02d}-"
            if month_str not in ev.start_date:
                continue
        filtered.append({
            "id": ev.id,
            "title": ev.title,
            "description": ev.description,
            "start_date": ev.start_date,
            "end_date": ev.end_date,
            "category": ev.category,
            "is_approved": ev.is_approved
        })

    return {
        "status": "success",
        "total": len(filtered),
        "events": filtered
    }

@router.post("/events")
def create_calendar_event(
    req: EventCreateRequest,
    db: Session = Depends(get_db)
):
    ev = AcademicEvent(
        title=req.title,
        description=req.description,
        start_date=req.start_date,
        end_date=req.end_date,
        category=req.category,
        is_approved=True
    )
    db.add(ev)
    db.commit()
    db.refresh(ev)
    return {"status": "success", "event": ev}
