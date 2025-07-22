from pymilvus import AnnSearchRequest, RRFRanker, MilvusClient
from services.embedding_api.function_call import get_embeddings

client = MilvusClient("./storages/vectorstore/milvus_demo.db")
ranker = RRFRanker()

def search(text: str, top_k: int = 3) -> list:
    query_dense_vector, query_sparse_vector = get_embeddings([text])

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
