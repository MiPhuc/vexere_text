import requests
import asyncio
import aiohttp
import httpx
from typing import List, Dict, Optional
import numpy as np
from scipy.sparse import csr_matrix

def get_embeddings(sentences:list, url:str = "http://localhost:3000/embeddings/"):
    payload = {"sentences": sentences}
    try:
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            dense_embeddings = np.array(result["dense_embeddings"], dtype=np.float64)
            sparse_embeddings = result["sparse_embeddings"]
            rows, cols, values = [], [], []
            for i, emb in enumerate(sparse_embeddings):
                for token_id, weight in emb.items():
                    rows.append(i)
                    cols.append(int(token_id))
                    values.append(weight)
            sparse_matrix = csr_matrix(
                (values, (rows, cols)),
                shape=(len(sparse_embeddings), 250002)  
            )
            return dense_embeddings, sparse_matrix
        else:
            raise Exception(f"API error: {response.status_code}, {response.json()['detail']}")
    except requests.exceptions.RequestException as e:
        raise Exception(f"API request failed: {e}")