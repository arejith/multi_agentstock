from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from memory.embeddings import get_embeddings
from memory.vectorstore.chroma_store import ChromaStore


@dataclass
class MemoryDocument:
    page_content: str
    metadata: dict


class NewsVectorStore:
    def __init__(self, persist_directory=None):
        self.documents = []
        self.store = None
        self.persist_directory = persist_directory or Path(".runtime_cache/chroma/news")

        try:
            embedding = get_embeddings()
            self.store = ChromaStore(
                embedding_model=embedding,
                persist_directory=self.persist_directory,
                collection_name="news",
            )
        except Exception:
            # Fall back to simple in-memory retrieval when embeddings or vector DB init are unavailable.
            self.store = None

    def build(self, news_list, role="data_team"):
        if role != "data_team":
            raise PermissionError("Only Data Team can write")

        texts = []
        metadatas = []
        seen_signatures = {
            self._build_signature(doc.page_content, doc.metadata)
            for doc in self.documents
        }
        stored_count = 0

        for news in news_list:
            text = news.get("text", "").strip()
            if not text:
                continue

            stored_at = news.get("timestamp") or datetime.utcnow().isoformat()
            metadata = {
                "ticker": news.get("ticker"),
                "title": news.get("title"),
                "date": news.get("date") or stored_at.split("T")[0],
                "timestamp": stored_at,
                "source": news.get("source"),
                "url": news.get("url"),
            }
            signature = self._build_signature(text, metadata)
            if signature in seen_signatures:
                continue

            texts.append(text)
            metadatas.append(metadata)
            self.documents.append(MemoryDocument(page_content=text, metadata=metadata))
            seen_signatures.add(signature)
            stored_count += 1

        if not texts:
            if news_list:
                return 0
            raise ValueError("No valid news data")

        if self.store is not None:
            try:
                self.store.add_texts(texts, metadatas)
            except Exception:
                self.store = None
        return stored_count

    def query(self, query, ticker=None, k=3):
        if self.store is not None:
            try:
                metadata_filter = {"ticker": ticker} if ticker else None
                fetch_k = max(k * 3, 8)
                results = self.store.max_marginal_relevance_search(
                    query,
                    k=k,
                    fetch_k=fetch_k,
                    filter=metadata_filter,
                )
            except Exception:
                self.store = None
                results = self.documents
        else:
            lowered_query = query.lower()
            results = [
                doc for doc in self.documents
                if lowered_query in doc.page_content.lower()
                or lowered_query in (doc.metadata.get("title") or "").lower()
            ]

        if ticker:
            results = [
                result for result in results
                if result.metadata.get("ticker") == ticker
            ]

        deduped_results = []
        seen_signatures = set()
        for result in results:
            signature = self._build_signature(result.page_content, result.metadata)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            deduped_results.append(result)

        return deduped_results[:k]

    def _build_signature(self, text, metadata):
        normalized_text = " ".join((text or "").strip().lower().split())
        normalized_title = (metadata.get("title") or "").strip().lower()
        normalized_url = (metadata.get("url") or "").strip().lower()
        normalized_ticker = (metadata.get("ticker") or "").strip().upper()
        if normalized_title:
            return ("title", normalized_ticker, normalized_title)
        if normalized_url:
            return ("url", normalized_ticker, normalized_url)
        return ("content", normalized_ticker, normalized_text)
