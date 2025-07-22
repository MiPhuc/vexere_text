from FlagEmbedding import BGEM3FlagModel
from typing import List, Tuple
import asyncio
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from uuid import uuid4
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import logging

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("./logs/embedding/embedding.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Config
batch_size = 2
max_request = 20
max_length = 5000
request_flush_timeout = 10
request_time_out = 30
gpu_time_out = 5
port = 3000

# Request/Response Models
class EmbedRequest(BaseModel):
    sentences: List[str]

class EmbedResponse(BaseModel):
    dense_embeddings: List[List[float]]
    sparse_embeddings: List[dict]

# Model wrapper
class m3Wrapper:
    def __init__(self, model_name: str, device: str = 'cuda'):
        logger.info(f"Initializing model: {model_name}")
        self.model = BGEM3FlagModel(model_name, device=device, use_fp16=(device != 'cpu'))

    def embed(self, sentences: List[str]) -> Tuple[List[List[float]], List[dict]]:
        result = self.model.encode(sentences, batch_size=batch_size, max_length=max_length, return_sparse=True)
        dense = result['dense_vecs'].astype(np.float64).tolist()
        sparse = result.get('lexical_weights', [{}] * len(sentences))
        sparse = [{k: float(v) for k, v in emb.items()} if emb else {} for emb in sparse]
        return dense, sparse

# Request processor
class RequestProcessor:
    def __init__(self, model: m3Wrapper, max_batch_size: int, accumulation_timeout: float):
        self.model = model
        self.max_batch_size = max_batch_size
        self.accumulation_timeout = accumulation_timeout
        self.queue = asyncio.Queue(maxsize=100)
        self.response_futures = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.gpu_lock = asyncio.Semaphore(1)
        self.processing_loop_task = asyncio.create_task(self.processing_loop())

    async def processing_loop(self):
        while True:
            requests, request_ids = [], []
            start_time = asyncio.get_event_loop().time()
            while len(requests) < self.max_batch_size:
                timeout = self.accumulation_timeout - (asyncio.get_event_loop().time() - start_time)
                if timeout <= 0:
                    break
                try:
                    data, req_id = await asyncio.wait_for(self.queue.get(), timeout=timeout)
                    requests.append(data)
                    request_ids.append(req_id)
                except asyncio.TimeoutError:
                    break
            if requests:
                await self.process_requests(requests, request_ids)

    async def process_requests(self, requests, request_ids):
        tasks = [
            asyncio.create_task(self.run_with_semaphore(self.model.embed, req.sentences, req_id))
            for req, req_id in zip(requests, request_ids)
        ]
        await asyncio.gather(*tasks)

    async def run_with_semaphore(self, func, data, request_id):
        async with self.gpu_lock:
            future = self.executor.submit(func, data)
            try:
                result = await asyncio.wait_for(asyncio.wrap_future(future), timeout=gpu_time_out)
                self.response_futures[request_id].set_result(result)
            except Exception as e:
                self.response_futures[request_id].set_exception(e)
            finally:
                self.response_futures.pop(request_id, None)

    async def process_request(self, request_data: EmbedRequest):
        request_id = str(uuid4())
        self.response_futures[request_id] = asyncio.Future()
        await self.queue.put((request_data, request_id))
        return await self.response_futures[request_id]

# FastAPI setup
app = FastAPI()
model = m3Wrapper('BAAI/bge-m3')
processor = RequestProcessor(model, max_batch_size=max_request, accumulation_timeout=request_flush_timeout)

@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    try:
        return await asyncio.wait_for(call_next(request), timeout=request_time_out)
    except asyncio.TimeoutError:
        return JSONResponse(
            {'detail': 'Request processing time exceeded limit'},
            status_code=504
        )

@app.post("/embeddings/", response_model=EmbedResponse)
async def get_embeddings(request: EmbedRequest):
    try:
        dense_embeddings, sparse_embeddings = await processor.process_request(request)
        return EmbedResponse(dense_embeddings=dense_embeddings, sparse_embeddings=sparse_embeddings)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Only required if running directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
