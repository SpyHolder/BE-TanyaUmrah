import calendar as month_calendar
from datetime import date
from typing import Literal, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.security import require_admin, require_student
from app.database.models import AcademicEvent
from app.database.session import get_db

router = APIRouter(prefix="/calendar", tags=["Calendar"])
EventCategory = Literal["akademik", "kemahasiswaan", "umum"]


class EventCreateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=10000)
    start_date: date
    end_date: Optional[date] = None
    category: EventCategory = "akademik"

    @model_validator(mode="after")
    def date_order(self):
        if self.end_date and self.end_date < self.start_date:
            raise ValueError("end_date tidak boleh sebelum start_date")
        if not self.title.strip():
            raise ValueError("title tidak boleh kosong")
        return self


def event_values(req: EventCreateRequest):
    return {
        **req.model_dump(exclude={"start_date", "end_date"}),
        "start_date": req.start_date.isoformat(),
        "end_date": req.end_date.isoformat() if req.end_date else None,
    }


@router.get("/events", dependencies=[Depends(require_student)])
def get_calendar_events(
    month: Optional[int] = Query(None, ge=1, le=12),
    year: Optional[int] = Query(None, ge=1900, le=9999),
    category: Optional[EventCategory] = None,
    db: Session = Depends(get_db),
):
    query = db.query(AcademicEvent).filter(AcademicEvent.is_approved.is_(True))
    if category:
        query = query.filter(AcademicEvent.category == category)
    if year:
        first_month, last_month = (month, month) if month else (1, 12)
        start = date(year, first_month, 1).isoformat()
        end = date(
            year, last_month, month_calendar.monthrange(year, last_month)[1]
        ).isoformat()
        query = query.filter(
            AcademicEvent.start_date <= end,
            func.coalesce(AcademicEvent.end_date, AcademicEvent.start_date) >= start,
        )
    events = query.order_by(AcademicEvent.start_date.asc()).all()
    if month and not year:
        # A month without a year means that month in any year; include spanning events.
        def overlaps_month(ev):
            try:
                start = date.fromisoformat(ev.start_date)
                end = date.fromisoformat(ev.end_date) if ev.end_date else start
            except ValueError:
                return False
            for event_year in range(start.year, end.year + 1):
                lo = date(event_year, month, 1)
                hi = date(
                    event_year, month, month_calendar.monthrange(event_year, month)[1]
                )
                if start <= hi and end >= lo:
                    return True
            return False

        events = [ev for ev in events if overlaps_month(ev)]
    fields = (
        "id",
        "title",
        "description",
        "start_date",
        "end_date",
        "category",
        "is_approved",
    )
    result = [{key: getattr(ev, key) for key in fields} for ev in events]
    return {"status": "success", "total": len(result), "events": result}


@router.post("/events", dependencies=[Depends(require_admin)])
def create_calendar_event(req: EventCreateRequest, db: Session = Depends(get_db)):
    ev = AcademicEvent(**event_values(req), is_approved=True)
    db.add(ev)
    db.commit()
    db.refresh(ev)
    return {"status": "success", "event": ev}
