from langgraph.graph import StateGraph, END
from langgraph.constants import Send
from typing import TypedDict, List, Optional

from src2.nodes import tool_call_node, ask_missing_info_node, receive_user_reply_node

class BookingState(TypedDict, total=False):
    messages: List[dict]
    steps: List[str]
    result: Optional[str]


def build_graph():
    graph = StateGraph(BookingState)

    graph.add_node("tool_call", tool_call_node)
    # graph.add_node("ask_missing", ask_missing_info_node)
    # graph.add_node("receive_user_reply", receive_user_reply_node)

    graph.set_entry_point("tool_call")

    # graph.add_conditional_edges(
    #     "tool_call",
    #     lambda state: "ask_missing" if any(state.get(k, None) is None for k in ["ticket_id", "new_time", "email", "message"]) else END,
    #     ["ask_missing", END]
    # )
    # graph.add_edge("ask_missing", "receive_user_reply")
    # graph.add_edge("receive_user_reply", "tool_call")

    return graph.compile()
