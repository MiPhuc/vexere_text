import json
import re


def transfer_obj_to_json(text: str, parser) -> dict:
    try:
        return parser.parse(text)
    except Exception:
        return json.loads(re.search(r"\{.*\}", text, re.DOTALL)[0])


def parse_sqlquery(raw_query_output: str) -> str:
    """
    Extracts the actual SQL query from LLM output of format:
    SQLQuery: SELECT * FROM ...
    """
    match = re.search(r"SQLQuery:\s*(.*)", raw_query_output, re.DOTALL)
    return match.group(1).strip() if match else raw_query_output
