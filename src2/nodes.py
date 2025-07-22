from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain_core.runnables import RunnablePassthrough

from src2.tools import update_booking_time, cancel_ticket, request_invoice, submit_complaint, get_booking_info
from src2.nodes_utils import search
from datetime import datetime

llm = ChatOpenAI(model="gpt-4o", temperature=0)
tools = [update_booking_time, cancel_ticket, request_invoice, submit_complaint, get_booking_info]

prompt = ChatPromptTemplate.from_messages([
    ("system", "Bạn là một trợ lý hỗ trợ khách hàng chuyên xử lý đặt vé."),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_function_messages(x.get("intermediate_steps", []))
    }
    | prompt
    | llm.bind_tools(tools)
    | OpenAIFunctionsAgentOutputParser()
)

executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def tool_call_node(state: dict) -> dict:
    user_input = next((m["content"] for m in reversed(state["messages"]) if m["role"] == "user"), "")

    # Dữ liệu tương tự từ search
    examples = search(user_input, top_k=3)
    retrieved_context = ""
    for hits in examples:
        for h in hits:
            retrieved_context += f"Câu hỏi: {h.text}\nTrả lời: {h.answer}\n-----\n"

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conversation = f"[\U0001f552 {current_time}]\n" + "\n".join(
        [f"{m['role'].capitalize()}: {m['content']}" for m in state["messages"]]
    )

    if retrieved_context:
        conversation = f"Thông tin tham khảo từ câu hỏi tương tự:\n{retrieved_context}\n\n{conversation}"

    result = executor.invoke({"input": conversation})
    state["messages"].append({"role": "assistant", "content": result["output"]})
    state.setdefault("steps", []).append(f"Tool call result: {result['output']}")
    return state


def ask_missing_info_node(state: dict) -> dict:
    state["messages"].append({
        "role": "assistant",
        "content": "Tôi cần thêm thông tin để xử lý. Bạn vui lòng cung cấp?"
    })
    return state

def receive_user_reply_node(state: dict) -> dict:
    reply = input("👤 Bạn: ")
    state["messages"].append({"role": "user", "content": reply})
    return state
