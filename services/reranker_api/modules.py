import asyncio
from FlagEmbedding import FlagReranker
from concurrent.futures import ThreadPoolExecutor

from collections import defaultdict

import logging

logger = logging.getLogger(__name__)


class AsyncReranker:
    def __init__(self, model_name='AITeamVN/Vietnamese_Reranker', use_fp16=True):
        self.reranker = FlagReranker(model_name, use_fp16=use_fp16, device="cuda")
        self.executor = ThreadPoolExecutor()

    async def score_articles_text_only(
        self,
        target_text: str,
        article_texts: list[str],
        threshold: float = 0.0,
        normalize: bool = True
    ) -> list[tuple[int, float]]:
        loop = asyncio.get_event_loop()

        pairs = [[target_text, text] for text in article_texts]

        scores = await loop.run_in_executor(
            self.executor,
            lambda: self.reranker.compute_score(pairs, normalize=normalize)
        )

        results = [(i, score) for i, score in enumerate(scores) if score >= threshold]
        results.sort(key=lambda x: x[1], reverse=True)
        del pairs
        del scores
        return results