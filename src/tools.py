import sqlite3

db_path = "./storages/mock_DB/bookings_full_info.db"

def update_booking_time(ticket_id: str, new_time: str, user_id: int) -> str:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Kiểm tra quyền sở hữu vé
    cursor.execute("""
        SELECT 1 FROM user_tickets WHERE ticket_id = ? AND user_id = ?
    """, (ticket_id, user_id))
    if not cursor.fetchone():
        return f"❌ Bạn không có quyền đổi vé {ticket_id}."

    cursor.execute("""
        UPDATE ticket_bookings 
        SET departure_date = ?, status = 'rescheduled' 
        WHERE id = ?
    """, (new_time, ticket_id))

    conn.commit()
    conn.close()
    return f"✅ Đã đổi vé {ticket_id} sang ngày {new_time}."

def cancel_ticket(ticket_id: str, user_id: int) -> str:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 1 FROM user_tickets WHERE ticket_id = ? AND user_id = ?
    """, (ticket_id, user_id))
    if not cursor.fetchone():
        return f"❌ Bạn không có quyền huỷ vé {ticket_id}."

    cursor.execute("UPDATE ticket_bookings SET status = 'canceled' WHERE id = ?", (ticket_id,))
    conn.commit()
    conn.close()
    return f"✅ Vé {ticket_id} đã được huỷ."

def request_invoice(ticket_id: str, email: str, user_id: int) -> str:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT 1 FROM user_tickets WHERE ticket_id = ? AND user_id = ?
    """, (ticket_id, user_id))
    if not cursor.fetchone():
        return f"❌ Bạn không có quyền lấy hoá đơn cho vé {ticket_id}."

    conn.close()
    return f"📋 Hóa đơn cho vé {ticket_id} đã được gửi đến {email}."

def submit_complaint(ticket_id: str, message: str, user_id: int) -> str:
    return f"🚨 Khiếu nại cho vé {ticket_id} của bạn đã được ghi nhận: '{message}'"