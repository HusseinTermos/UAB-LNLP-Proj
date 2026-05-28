import re
import uuid
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict[str, Any]


class RAG:
    def __init__(
        self,
        document: str,
        collection_name: str = "medical_rag_dense",
        qdrant_path: str | None = None,
        chunk_size: int = 250,
        chunk_overlap: int = 60,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        reset_collection: bool = True,
    ):
        """
        document:
            Initial text to index.

        collection_name:
            Qdrant collection name for dense bi-encoder vectors.

        qdrant_path:
            None      -> in-memory Qdrant
            "rag_db"  -> local persistent Qdrant folder

        chunk_size:
            Number of words per chunk.

        chunk_overlap:
            Number of overlapping words between consecutive chunks.

        embedding_model_name:
            Small sentence-transformer model for dense semantic retrieval.

        reset_collection:
            True  -> delete and rebuild the Qdrant collection on initialization.
            False -> reuse existing collection if it exists.
                     For this simple version, True is safer because we are not
                     storing a separate manifest of existing chunks.
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

        self.collection_name = collection_name
        self.vector_name = "dense"
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.model = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.model.get_embedding_dimension()

        if self.embedding_dim is None:
            test_vector = self.model.encode(["test"], normalize_embeddings=True)[0]
            self.embedding_dim = len(test_vector)

        self.client = QdrantClient(":memory:" if qdrant_path is None else qdrant_path)

        self.chunks: list[Chunk] = []
        self.bm25: BM25Okapi | None = None
        self.tokenized_corpus: list[list[str]] = []

        self._create_or_reset_dense_collection(reset_collection=reset_collection)

        if document.strip():
            self.add_text(
                document,
                source="initial_document",
                temporary=False,
            )

    # ------------------------------------------------------------------
    # Public retrieval functions
    # ------------------------------------------------------------------

    def retrieve_bm25(
        self,
        query: str,
        top_k: int = 5,
        temporary_text: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve top_k chunks using BM25 keyword retrieval.

        temporary_text:
            Optional text inserted only for this retrieval call.
            It is removed immediately afterward.
        """
        if temporary_text:
            temp_ids = self.add_text(
                temporary_text,
                source="temporary_query_context",
                temporary=True,
            )

            try:
                return self.retrieve_bm25(query=query, top_k=top_k)
            finally:
                self.remove(temp_ids)

        if self.bm25 is None or not self.chunks:
            return []

        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        ranked = sorted(
            enumerate(scores),
            key=lambda pair: float(pair[1]),
            reverse=True,
        )

        results = []
        for index, score in ranked:
            score = float(score)

            # BM25 score 0 usually means no useful lexical match.
            if score <= 0:
                continue

            chunk = self.chunks[index]
            results.append(self._make_result(chunk, score, method="bm25"))

            if len(results) >= top_k:
                break

        return results

    def retrieve_biencoder(
        self,
        query: str,
        top_k: int = 5,
        temporary_text: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve top_k chunks using dense semantic retrieval.

        This uses a sentence-transformer bi-encoder:
            query -> vector
            chunk -> vector
            Qdrant cosine search over chunk vectors
        """
        if temporary_text:
            temp_ids = self.add_text(
                temporary_text,
                source="temporary_query_context",
                temporary=True,
            )

            try:
                return self.retrieve_biencoder(query=query, top_k=top_k)
            finally:
                self.remove(temp_ids)

        if not self.chunks:
            return []

        query_vector = self.model.encode(
            [query],
            normalize_embeddings=True,
        )[0].tolist()

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using=self.vector_name,
            limit=top_k,
            with_payload=True,
        )

        results = []
        for point in response.points:
            payload = point.payload or {}

            results.append(
                {
                    "id": str(point.id),
                    "score": float(point.score),
                    "text": payload.get("text", ""),
                    "metadata": {
                        key: value
                        for key, value in payload.items()
                        if key != "text"
                    },
                    "method": "biencoder",
                }
            )

        return results

    # Alias with another common spelling.
    def retrieve_bi_encoder(
        self,
        query: str,
        top_k: int = 5,
        temporary_text: str | None = None,
    ) -> list[dict[str, Any]]:
        return self.retrieve_biencoder(
            query=query,
            top_k=top_k,
            temporary_text=temporary_text,
        )

    # Dense alias in case you prefer this name.
    def retrieve_dense(
        self,
        query: str,
        top_k: int = 5,
        temporary_text: str | None = None,
    ) -> list[dict[str, Any]]:
        return self.retrieve_biencoder(
            query=query,
            top_k=top_k,
            temporary_text=temporary_text,
        )

    # Compatibility with the older RAG class style.
    def query(
        self,
        query: str,
        top_k: int = 5,
        temporary_text: str | None = None,
        method: str = "biencoder",
    ) -> list[dict[str, Any]]:
        if method == "bm25":
            return self.retrieve_bm25(
                query=query,
                top_k=top_k,
                temporary_text=temporary_text,
            )

        if method in {"biencoder", "bi_encoder", "dense"}:
            return self.retrieve_biencoder(
                query=query,
                top_k=top_k,
                temporary_text=temporary_text,
            )

        raise ValueError("method must be one of: 'bm25', 'biencoder', 'bi_encoder', 'dense'.")

    # ------------------------------------------------------------------
    # Public add/remove functions
    # ------------------------------------------------------------------

    def add_text(
        self,
        text: str,
        source: str = "added_text",
        temporary: bool = False,
    ) -> list[str]:
        """
        Add text to both:
            1. BM25 index
            2. Dense Qdrant index

        Returns the IDs of the chunks that were added.
        """
        new_chunks = self._chunk_text(
            text=text,
            source=source,
            temporary=temporary,
        )

        if not new_chunks:
            return []

        self.chunks.extend(new_chunks)

        self._rebuild_bm25()
        self._upsert_dense_chunks(new_chunks)

        return [chunk.id for chunk in new_chunks]

    def remove(self, ids: list[str]) -> None:
        """
        Remove chunks from both:
            1. BM25 index
            2. Dense Qdrant index
        """
        if not ids:
            return

        id_set = set(ids)

        self.chunks = [
            chunk
            for chunk in self.chunks
            if chunk.id not in id_set
        ]

        self._rebuild_bm25()

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(points=ids),
            wait=True,
        )

    # ------------------------------------------------------------------
    # Optional pretty formatter
    # ------------------------------------------------------------------

    def answer(
        self,
        query: str,
        top_k: int = 5,
        method: str = "biencoder",
        temporary_text: str | None = None,
    ) -> str:
        """
        This does NOT generate a medical answer.
        It only formats retrieved evidence chunks.
        """
        results = self.query(
            query=query,
            top_k=top_k,
            method=method,
            temporary_text=temporary_text,
        )

        if not results:
            return "No relevant chunks found."

        lines = []
        for i, result in enumerate(results, start=1):
            source = result["metadata"].get("source", "unknown")
            temporary = result["metadata"].get("temporary", False)

            lines.append(
                f"[{i}] method={result['method']} "
                f"score={result['score']:.4f} "
                f"source={source} "
                f"temporary={temporary}\n"
                f"{result['text']}"
            )

        return "\n\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_or_reset_dense_collection(self, reset_collection: bool) -> None:
        exists = self.client.collection_exists(self.collection_name)

        if exists and reset_collection:
            self.client.delete_collection(self.collection_name)
            exists = False

        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    self.vector_name: models.VectorParams(
                        size=self.embedding_dim,
                        distance=models.Distance.COSINE,
                    )
                },
            )

    def _chunk_text(
        self,
        text: str,
        source: str,
        temporary: bool,
    ) -> list[Chunk]:
        words = re.findall(r"\S+", text)

        if not words:
            return []

        chunks: list[Chunk] = []
        step = self.chunk_size - self.chunk_overlap

        for local_index, start in enumerate(range(0, len(words), step)):
            end = min(start + self.chunk_size, len(words))
            chunk_text = " ".join(words[start:end])

            chunk = Chunk(
                id=str(uuid.uuid4()),
                text=chunk_text,
                metadata={
                    "source": source,
                    "temporary": temporary,
                    "chunking_method": "naive_sliding_window",
                    "local_chunk_index": local_index,
                    "start_word": start,
                    "end_word": end,
                },
            )

            chunks.append(chunk)

            if end >= len(words):
                break

        return chunks

    def _rebuild_bm25(self) -> None:
        self.tokenized_corpus = [
            self._tokenize(chunk.text)
            for chunk in self.chunks
        ]

        if not self.tokenized_corpus:
            self.bm25 = None
            return

        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _upsert_dense_chunks(self, chunks: list[Chunk]) -> None:
        texts = [chunk.text for chunk in chunks]

        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
        )

        points = []
        for chunk, vector in zip(chunks, vectors):
            payload = {
                "text": chunk.text,
                **chunk.metadata,
            }

            points.append(
                models.PointStruct(
                    id=chunk.id,
                    vector={
                        self.vector_name: vector.tolist(),
                    },
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

    def _tokenize(self, text: str) -> list[str]:
        """
        Simple tokenizer for BM25.

        Keeps words, numbers, and basic hyphenated/apostrophe terms.
        Good enough for now.
        """
        return re.findall(
            r"\b\w+(?:[-']\w+)*\b",
            text.lower(),
        )

    def _make_result(
        self,
        chunk: Chunk,
        score: float,
        method: str,
    ) -> dict[str, Any]:
        return {
            "id": chunk.id,
            "score": score,
            "text": chunk.text,
            "metadata": chunk.metadata,
            "method": method,
        }


if __name__ == "__main__":
    document = """
    Vitamin C supplementation has been studied for the prevention and treatment
    of the common cold. Some evidence suggests it may slightly reduce the duration
    of colds, but it does not consistently reduce the incidence of colds in the
    general population.

    Antibiotics are used to treat bacterial infections. They are not effective
    against viral infections such as the common cold or influenza.

    Metformin is commonly used as a first-line medication for type 2 diabetes.
    It helps lower blood glucose levels and can reduce HbA1c.
    """

    rag = RAG(document)

    claim = "Vitamin C prevents the common cold."

    print("=== BM25 ===")
    print(rag.answer(claim, method="bm25", top_k=3))

    print("\n=== BI-ENCODER ===")
    print(rag.answer(claim, method="biencoder", top_k=3))

    print("\n=== TEMPORARY TEXT EXAMPLE ===")
    temporary_reference = """
    A temporary reference says that vitamin C did not significantly reduce
    the incidence of common colds among the general population.
    """

    print(
        rag.answer(
            claim,
            method="bm25",
            top_k=3,
            temporary_text=temporary_reference,
        )
    )