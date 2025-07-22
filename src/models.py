from pydantic import BaseModel, Field
from typing import Optional

class UserIntent(BaseModel):
    intent: str = Field(None, 
    description="Phân loại ý định người dùng theo: query_booking_info, update_booking_time, cancel_ticket, request_invoice, submit_complaint hoặc other",
    enum=["query_booking_info",
          "update_booking_time", 
          "cancel_ticket", 
          "request_invoice", 
          "submit_complaint",
          "other"])

class QueryBookingInfo(BaseModel):
    question: str = Field(..., description="Tổng quát câu hỏi của người dùng")

class UpdateBookingInput(BaseModel):
    ticket_id: Optional[str] = Field(None, description="Mã vé xe")
    new_time: Optional[str] = Field(None, description="Giờ mới muốn đổi")

class CancelTicketInput(BaseModel):
    ticket_id: Optional[str] = Field(None, description="Mã vé cần huỷ")

class InvoiceRequestInput(BaseModel):
    ticket_id: Optional[str] = Field(None, description="Mã vé cần xuất hóa đơn")
    email: Optional[str] = Field(None, description="Email nhận hóa đơn")

class ComplaintInput(BaseModel):
    ticket_id: Optional[str] = Field(None, description="Mã vé liên quan khiếu nại")
    message: Optional[str] = Field(None, description="Nội dung khiếu nại")