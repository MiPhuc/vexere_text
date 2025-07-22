from datetime import datetime
from datetime import datetime, timedelta, timezone
import re
import json

def get_vietnam_timestamp_iso() -> str:
    vn_tz = timezone(timedelta(hours=7))  
    vn_time = datetime.now(vn_tz)
    return vn_time.isoformat()


def iso_to_epoch(timestamp_str: str) -> int:
    dt = datetime.fromisoformat(timestamp_str)  
    return int(dt.timestamp())

def extract_after_think(text: str) -> str:
    match = re.search(r'</think>(.*)', text, re.DOTALL)
    return match.group(1).strip() if match else text

def strip_json(text):
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            data = json.loads(json_str)
            return data
        except json.JSONDecodeError as e:
            return text
    else:
      return text

def to_dict(obj):
    try:
        if isinstance(obj, list):
            return [to_dict(i) for i in obj]
        elif hasattr(obj, '__dict__'):
            return {k: to_dict(v) for k, v in obj.__dict__.items()}
        else:
            return obj
    except Exception as e:
        return obj


def transfer_obj_to_json(response, parser):
    if not isinstance(response, str):
        try:
            response = response.content
        except:
            return None
    try:
        answer = to_dict(parser.parse(response))
    except:
        answer = strip_json(extract_after_think(response))

    return answer

# def parse_sqlquery(query):
#     # print(query)
#     match = re.search(r"SQLQuery:\s*(.*)", query)
#     if match:
#         return match.group(1).replace("```","").strip()
#     else:
#         return query.replace("sql","").replace("```","").strip()


def parse_response_to_sql(response:str) -> str:
    sql_query_start = response.find("SQLQuery:")
    if sql_query_start != -1:
        response = response[sql_query_start:]
        if response.startswith("SQLQuery:"):
            response = response[len("SQLQuery:") :]
    sql_result_start = response.find("SQLResult:")
    if sql_result_start != -1:
        response = response[:sql_result_start]
    response = response.strip().strip("```").strip()
    if response.startswith("sql"):
        response = response[len("sql") :]
    return response.strip()