import logging

from openai import APIError, OpenAI

from retrieval_lab.errors import EmbeddingError

logger = logging.getLogger(__name__)


class Embedder:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class OpenAIEmbedder(Embedder):
    def __init__(self, client: OpenAI, model: str, batch_size: int = 64) -> None:
        self._client = client
        self._model = model
        self._batch_size = batch_size

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float] | None] = [None] * len(texts)
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            try:
                response = self._client.embeddings.create(model=self._model, input=batch)
            except APIError as exc:
                status = getattr(exc, "status_code", "unavailable")
                logger.error("embedding request failed status=%s", status)
                raise EmbeddingError() from exc
            for item in response.data:
                vectors[start + item.index] = item.embedding
        if any(vector is None for vector in vectors):
            logger.error("embedding response was incomplete")
            raise EmbeddingError("Embedding response was incomplete.")
        return [vector for vector in vectors if vector is not None]
