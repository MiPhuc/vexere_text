from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool

from uuid import uuid4
from datetime import datetime

from src.functions import transfer_obj_to_json, parse_response_to_sql
from src.parsers import INTENT_TO_PARSER, classify_intent_parser
from src.tools import update_booking_time, cancel_ticket, request_invoice, submit_complaint

from pymilvus import AnnSearchRequest
from pymilvus import RRFRanker
from pymilvus import MilvusClient

from services.embedding_api.function_call import get_embeddings
from config import VECTORSTORE, MODEL_NAME, DB_PATH, top_k

client = MilvusClient(VECTORSTORE)

ranker = RRFRanker()
llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
engine = create_engine(DB_PATH)
db = SQLDatabase.from_uri(DB_PATH, sample_rows_in_table_info=5)
execute_query = QuerySQLDatabaseTool(db=db)
table_info = db.get_table_info()

prompt_write_query = PromptTemplate.from_template(
    """You are an expert SQL assistant. Given a user's question, your task is to:
1. Generate a syntactically correct {dialect} query using the tables and columns below.
2. Execute the query and provide the result.

Limit results to {top_k} rows unless the user specifically asks for more.

### Tables and schema:
ticket_bookings (
  id INTEGER PRIMARY KEY,
  ticket_id TEXT UNIQUE NOT NULL,            -- mã vé duy nhất, ví dụ: "VX-1001"
  from_city TEXT NOT NULL,                   -- thành phố xuất phát
  to_city TEXT NOT NULL,                     -- thành phố đích
  departure_date TEXT NOT NULL,              -- ngày khởi hành (YYYY-MM-DD)
  bus_type TEXT,                             -- loại xe
  seat TEXT,                                 -- số ghế
  payment_method TEXT,                       -- phương thức thanh toán
  status TEXT,                               -- trạng thái vé ('pending', 'confirmed', 'canceled', 'rescheduled')
  created_at TEXT DEFAULT CURRENT_TIMESTAMP  -- thời điểm đặt vé (YYYY-MM-DD hh:mm:ss)
)

users (
  id INTEGER PRIMARY KEY,                    -- ID người dùng
  name TEXT NOT NULL,                        -- tên đầy đủ
  email TEXT UNIQUE NOT NULL                 -- email đăng ký
)

user_tickets (
  user_id INTEGER REFERENCES users(id),      -- liên kết với users.id
  ticket_id TEXT REFERENCES ticket_bookings(ticket_id),  -- liên kết với ticket_bookings.ticket_id
  PRIMARY KEY (user_id, ticket_id)
)

### Relationships:
- Một người dùng (user) có thể có nhiều vé (ticket)
- Bảng `user_tickets` map `user_id` ↔ `ticket_id`

### Output format:
Question: <user_input>
SQLQuery: <valid SQL query>
SQLResult: <result of SQL execution>
Answer: <final answer to user>

Only use the tables specified above.

Question: {input}
"""
).partial(dialect=db.dialect, table_info=table_info, top_k=5)

answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}

Answer: """
)

def extract_intent_node(state: dict) -> dict:
    format_instructions = classify_intent_parser.get_format_instructions()
    prompt = PromptTemplate.from_template(
        "Bạn là nhân viên tư vấn khác hành về dịch vụ vé xe."
        "Hãy xác định intent (mục đích) của người dùng (user) từ đoạn hội thoại dưới đây. "
        "Chọn chính xác **một trong các intent sau**:\n\n"
        "- query_booking_info: Người dùng hỏi thông tin về chuyến đi, số vé, lịch trình, đặt vé v.v.\n"
        "- update_booking_time: Người dùng muốn thay đổi thời gian khởi hành hoặc giờ đi của vé đã đặt.\n"
        "- cancel_ticket: Người dùng muốn huỷ vé đã đặt.\n"
        "- request_invoice: Người dùng muốn lấy hoặc gửi hoá đơn cho chuyến đi.\n"
        "- submit_complaint: Người dùng muốn để feedback phản ánh, khiếu nại về chuyến đi hoặc dịch vụ.\n"
        "- other: Các yêu cầu khác hoặc không rõ ràng.\n\n"
        "Dưới đây là đoạn hội thoại:\n"
        "{conversation}\n\n"
        "{format_instructions}"
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    conversation = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in state["messages"]])
    output = chain.run({"conversation": conversation, "format_instructions":format_instructions})
    state["intent"]  = transfer_obj_to_json(output, classify_intent_parser)["intent"]
    state.setdefault("steps", []).append(f"Intent: {state['intent']}")
    return state

def extract_info_node(state: dict) -> dict:
    parser = INTENT_TO_PARSER[state["intent"]]
    format_instructions = parser.get_format_instructions()
    prompt = PromptTemplate.from_template(
        "Trích thông tin cần thiết từ đoạn hội thoại:\n{conversation}\n\n{format_instructions}"
    )
    conversation = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in state["messages"]])
    chain = LLMChain(llm=llm, prompt=prompt)
    output = chain.run({"conversation": conversation, "format_instructions":format_instructions})
    parsed = transfer_obj_to_json(output, parser)

    for field, value in parsed.items():
        state[field] = value
    state["missing_info"] = any(v is None for v in parsed.values())
    state.setdefault("steps", []).append(f"Extracted: {parsed}")
    return state

def ask_missing_info_node(state: dict) -> dict:
    missing_fields = [k for k, v in state.items() if k in ( "ticket_id", "new_time", "email", "message") and v is None]
    miss_text = ', '.join(missing_fields)
    question = f"Vui lòng cung cấp: {miss_text}?"
    state["messages"].append({"role": "assistant", "content": question})
    state.setdefault("steps", []).append(f"Ask: {question}")
    return state

def receive_user_reply_node(state: dict) -> dict:
    reply = input("👤 Bạn: ")
    state["messages"].append({"role": "user", "content": reply})
    state.setdefault("steps", []).append(f"User replied: {reply}")
    return state

def call_tool_node(state: dict) -> dict:
    intent = state["intent"]
    user_id = state.get("user_id")
    if intent == "update_booking_time":
        result = update_booking_time(state["ticket_id"], state["new_time"], user_id)
    elif intent == "cancel_ticket":
        result = cancel_ticket(state["ticket_id"], user_id)
    elif intent == "request_invoice":
        result = request_invoice(state["ticket_id"], state["email"], user_id)
    elif intent == "submit_complaint":
        result = submit_complaint(state["ticket_id"], state["message"], user_id)
    else:
        result = "Intent không hợp lệ."
    state["result"] = result
    state["messages"].append({"role": "assistant", "content": result})
    state.setdefault("steps", []).append(f"Result: {result}")
    return state


def get_db_info_node(state: dict) -> dict:
    
    answer = answer_prompt | llm | StrOutputParser()
    write_query = create_sql_query_chain(llm, db, prompt_write_query)

    chain = (
        RunnablePassthrough.assign(query=write_query).assign(
            query=lambda x: parse_response_to_sql(x["query"])
        )
        | RunnablePassthrough.assign(result=execute_query)
        | answer
    )
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conversation = f"Thời điểm cuộc trò chuyện: {current_time}\n" + "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in state["messages"]])
    ans = chain.invoke({"question": conversation})
    state["messages"].append({"role": "assistant", "content": ans})
    state.setdefault("steps", []).append(f"SQL answer: {ans}")
    return state


def search(text:str, top_k:int) -> list:

    query_dense_vector, query_sparse_vector= get_embeddings([text])

    search_param_1 = {
        "data": [query_dense_vector[0]],
        "anns_field": "text_dense",
        "param": {"nprobe": 10},
        "limit": top_k
    }
    request_1 = AnnSearchRequest(**search_param_1)

    search_param_2 = {
        "data": [query_sparse_vector],
        "anns_field": "text_sparse",
        "param": {"drop_ratio_search": 0.2},
        "limit": top_k
    }
    request_2 = AnnSearchRequest(**search_param_2)

    reqs = [request_1, request_2]

    res = client.hybrid_search(
      collection_name="vexere",
      reqs=reqs,
      ranker=ranker,
      output_fields=["text", "answer"],
      limit=top_k
  )
    
    return res


def QA_node(state:dict) -> dict:
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conversation = f"Thời điểm cuộc trò chuyện: {current_time}\n" + "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in state["messages"]])
    user_text = next(
          (m["content"] for m in reversed(state["messages"]) if m["role"] == "user"),
          None
      )
    if not user_text:
        state["messages"].append({"role": "assistant", "content": "Liên hệ với nhân viên để biết thêm chi tiết."})
        state.setdefault("steps", []).append(f"SQL answer: Liên hệ với nhân viên để biết thêm chi tiết.")
        return state
    res = search(str(user_text), top_k)

    rels = ""
    for hits in res:
        for hit in hits:
            rels += f"Question: {hit.text}\nAnswer: {hit.answer}\n-----\n"

    template  = PromptTemplate.from_template(
        """Bạn là nhân viên chăm sóc khách hàng của Vexere.  
Hãy bắt đầu bằng lời chào thân thiện nếu người dùng vừa mới bắt đầu cuộc trò chuyện (ví dụ: “Chào bạn, Vexere xin hỗ trợ!”).  
Sau đó, hãy trả lời linh hoạt theo các câu trả lời mẫu dưới đây.  

Nếu không có câu trả lời mẫu nào phù hợp, hãy trả lời: "Liên hệ với nhân viên để biết thêm chi tiết."  
----------  
Câu trả lời mẫu: {rels}  
----------  
Cuộc đối thoại: {question}
"""
    )

    chain = template | llm
    ans = chain.invoke({"question": conversation,
                  "rels": rels})
    state["messages"].append({"role": "assistant", "content": ans.content})
    state.setdefault("steps", []).append(f"SQL answer: {ans.content}")
    return state

