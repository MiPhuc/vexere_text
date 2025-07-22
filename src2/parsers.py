from pydantic import BaseModel, Field
from typing import Optional

class UpdateBookingTime(BaseModel):
    ticket_id: Optional[str] = Field(..., description="ID vé cần đổi")
    new_time: Optional[str] = Field(..., description="Thời gian mới")

class CancelTicket(BaseModel):
    ticket_id: Optional[str] = Field(..., description="ID vé cần huỷ")

class RequestInvoice(BaseModel):
    ticket_id: Optional[str] = Field(..., description="ID vé cần xuất hoá đơn")
    email: Optional[str] = Field(..., description="Email nhận hoá đơn")

class SubmitComplaint(BaseModel):
    ticket_id: Optional[str] = Field(..., description="ID vé muốn khiếu nại")
    message: Optional[str] = Field(..., description="Nội dung khiếu nại")

class IntentClassifier(BaseModel):
    intent: str

INTENT_TO_PARSER = {
    "update_booking_time": UpdateBookingTime,
    "cancel_ticket": CancelTicket,
    "request_invoice": RequestInvoice,
    "submit_complaint": SubmitComplaint
}

classify_intent_parser = IntentClassifier
