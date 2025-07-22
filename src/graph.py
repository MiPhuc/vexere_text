from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional
from langgraph.constants import Send

from src.nodes import (
    extract_intent_node,
    extract_info_node,
    ask_missing_info_node,
    receive_user_reply_node,
    get_db_info_node,
    call_tool_node,
    QA_node,
)

class BookingState(TypedDict, total=False):
    user_id: int
    messages: List[dict]
    intent: Optional[str]
    ticket_id: Optional[str]
    new_time: Optional[str]
    email: Optional[str]
    question: Optional[str]
    message: Optional[str]
    missing_info: Optional[bool]
    result: Optional[str]
    steps: List[str]

def route_intent(state: dict) -> dict:
    if state["intent"] in ["update_booking_time", "cancel_ticket", "request_invoice", "submit_complaint"]:
        return Send("extract_info", state)
    elif state["intent"] == "query_booking_info":
        return Send("get_db_info_node", state)
    else:
        return Send("QA_node", state)

def build_graph():
    graph = StateGraph(BookingState)
    graph.add_node("extract_intent", extract_intent_node)
    graph.add_node("extract_info", extract_info_node)
    graph.add_node("ask_missing", ask_missing_info_node)
    graph.add_node("receive_user_reply", receive_user_reply_node)
    graph.add_node("get_db_info_node", get_db_info_node)
    graph.add_node("QA_node", QA_node)
    graph.add_node("call_tool", call_tool_node)

    graph.set_entry_point("extract_intent")
    graph.add_conditional_edges("extract_intent", route_intent, ["extract_info", 
                                                                  "QA_node", 
                                                                  "get_db_info_node"])
    graph.add_conditional_edges(
        "extract_info",
        lambda s: "ask_missing" if s.get("missing_info") else "call_tool"
    )
    graph.add_edge("ask_missing", "receive_user_reply")
    graph.add_edge("receive_user_reply", "extract_info")
    graph.add_edge("call_tool", END)
    graph.add_edge("get_db_info_node", END)
    graph.add_edge("QA_node", END)

    return graph.compile()