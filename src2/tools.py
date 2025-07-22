import sqlite3
from langchain_core.tools import tool
from langchain.prompts import PromptTemplate
from langchain.chains import create_sql_query_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain.chat_models import ChatOpenAI
from datetime import datetime

from src.functions import parse_sqlquery

db_path = "./storages/mock_DB/bookings_full_info.db"

llm = ChatOpenAI(model="gpt-4o", temperature=0)
db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
execute_query = QuerySQLDatabaseTool(db=db)
table_info = db.get_table_info()

prompt_write_query = PromptTemplate.from_template(
    """Given an input question, first create a syntactically correct {dialect} query to run, 
then look at the results of the query and return the answer. 

Unless the user specifies a specific number of examples, always limit your query to at most {top_k} results. 
You can order the results by a relevant column to return the most interesting examples.

Only use the following tables:
{table_info}

Here is the description of the columns in the tables:

`id`: ID vé  
`from_city`: Thành phố đi  
`to_city`: Thành phố đến  
`departure_date`: Ngày khởi hành  
`bus_type`: Loại xe  
`seat`: Số ghế  
`payment_method`: Phương thức thanh toán  
`status`: Trạng thái vé  
`created_at`: Thời điểm đặt vé  

Question: {input}
"""
).partial(dialect=db.dialect, table_info=table_info, top_k=5)


@tool
def update_booking_time(ticket_id: str, new_time: str) -> str:
    """Thay đổi thời gian khởi hành của vé có ID đã cho."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM ticket_bookings WHERE id = ?", (ticket_id,))
    if not cursor.fetchone():
        return f"❌ Không tìm thấy vé {ticket_id}"
    cursor.execute("""
        UPDATE ticket_bookings 
        SET departure_date = ?, status = 'rescheduled' 
        WHERE id = ?
    """, (new_time, ticket_id))
    conn.commit()
    conn.close()
    return f"✅ Đã đổi vé {ticket_id} sang ngày {new_time}"


@tool
def cancel_ticket(ticket_id: str) -> str:
    """Huỷ vé có ID đã cho."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM ticket_bookings WHERE id = ?", (ticket_id,))
    if not cursor.fetchone():
        return f"❌ Không tìm thấy vé {ticket_id}"
    cursor.execute("UPDATE ticket_bookings SET status = 'canceled' WHERE id = ?", (ticket_id,))
    conn.commit()
    conn.close()
    return f"✅ Vé {ticket_id} đã được huỷ."


@tool
def request_invoice(ticket_id: str, email: str) -> str:
    """Gửi hoá đơn của vé đến email chỉ định."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM ticket_bookings WHERE id = ?", (ticket_id,))
    if not cursor.fetchone():
        return f"❌ Không tìm thấy vé {ticket_id}"
    conn.close()
    return f"📋 Hóa đơn cho vé {ticket_id} đã được gửi đến {email}."


@tool
def submit_complaint(ticket_id: str, message: str) -> str:
    """Gửi khiếu nại cho vé cụ thể với nội dung cụ thể."""
    return f"🚨 Khiếu nại cho vé {ticket_id} đã được ghi nhận: '{message}'"


@tool
def get_booking_info(question: str) -> str:
    """Truy vấn các thông tin liên quan đến thông tin vé, lịch trình, hoặc trạng thái chuyến xe."""
    answer_prompt = PromptTemplate.from_template(
        """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer:"""
    )

    answer_chain = answer_prompt | llm | StrOutputParser()
    write_query = create_sql_query_chain(llm, db, prompt_write_query)

    chain = (
        RunnablePassthrough.assign(query=write_query).assign(
            query=lambda x: parse_sqlquery(x["query"])
        )
        | RunnablePassthrough.assign(result=execute_query)
        | answer_chain
    )

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_question = f"[{current_time}] {question}"

    return chain.invoke({"question": full_question})
