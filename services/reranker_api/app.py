from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import asyncio
import logging

from modules import AsyncReranker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("./logs/rerank.log"),  
        logging.StreamHandler()  
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()
reranker: Optional[AsyncReranker] = None  

port = 3002

class RerankTextRequest(BaseModel):
    target_text: str
    article_texts: List[str]
    threshold: float = 0.0
    normalize: bool = True


@app.on_event("startup")
async def startup_event():
    global reranker
    reranker = AsyncReranker()
    print("Reranker model loaded once at startup.")



@app.post("/rerank-text")
async def rerank_text(req: RerankTextRequest):
    if reranker is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    try:
        results = await reranker.score_articles_text_only(
            target_text=req.target_text,
            article_texts=req.article_texts,
            threshold=req.threshold,
            normalize=req.normalize
        )
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting ReRANK server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)