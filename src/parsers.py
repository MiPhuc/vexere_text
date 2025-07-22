from langchain.output_parsers import PydanticOutputParser
from src.models import (
    QueryBookingInfo,
    UpdateBookingInput,
    CancelTicketInput,
    InvoiceRequestInput,
    ComplaintInput,
    UserIntent
)


classify_intent_parser = PydanticOutputParser(pydantic_object=UserIntent)

INTENT_TO_PARSER = {
    "query_booking_info": PydanticOutputParser(pydantic_object=QueryBookingInfo),
    "update_booking_time": PydanticOutputParser(pydantic_object=UpdateBookingInput),
    "cancel_ticket": PydanticOutputParser(pydantic_object=CancelTicketInput),
    "request_invoice": PydanticOutputParser(pydantic_object=InvoiceRequestInput),
    "submit_complaint": PydanticOutputParser(pydantic_object=ComplaintInput),
}
