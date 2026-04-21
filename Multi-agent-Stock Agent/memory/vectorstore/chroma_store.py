from pathlib import Path
from uuid import uuid4

from langchain_chroma import Chroma


DEFAULT_CHROMA_DIR = Path(".runtime_cache/chroma/news")


class ChromaStore:
    def __init__(self, embedding_model, persist_directory=None, collection_name="news"):
        self.embedding_model = embedding_model
        self.persist_directory = str(Path(persist_directory or DEFAULT_CHROMA_DIR))
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        self.db = Chroma(
            collection_name=collection_name,
            embedding_function=self.embedding_model,
            persist_directory=self.persist_directory,
        )

    def similarity_search(self, query, k=3, filter=None):
        return self.db.similarity_search(query, k=k, filter=filter)

    def max_marginal_relevance_search(self, query, k=3, fetch_k=8, filter=None):
        return self.db.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k, filter=filter)

    def add_texts(self, texts, metadatas=None):
        if not texts:
            raise ValueError("No texts provided")

        ids = [str(uuid4()) for _ in texts]
        self.db.add_texts(texts=texts, metadatas=metadatas, ids=ids)
