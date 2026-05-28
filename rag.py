import re
import uuid
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict[str, Any]


class RAG:
    def __init__(
        self,
        document: str,
        qdrant_path: str | None = None,
        chunk_size: int = 250,
        chunk_overlap: int = 60,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cross_encoder_model_name: str = "cross-encoder/ms-marco-TinyBERT-L2-v2",
        reset_collection: bool = True,
    ):
        """
        document:
            Initial text to index.

        qdrant_path:
            None      -> in-memory Qdrant
            "rag_db"  -> local persistent Qdrant folder

        chunk_size:
            Number of words per chunk.

        chunk_overlap:
            Number of overlapping words between consecutive chunks.

        embedding_model_name:
            Small sentence-transformer model for dense semantic retrieval.

        cross_encoder_model_name:
            Small cross-encoder model for reranking BM25 + bi-encoder candidates.

        reset_collection:
            True  -> delete and rebuild the Qdrant collection on initialization.
            False -> reuse existing collection if it exists.

            For this simple version, True is safer because we are not loading
            existing Qdrant points back into self.chunks/BM25.
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

        self.collection_name = "rag_vectors"
        self.vector_name = "dense"
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.model = SentenceTransformer(embedding_model_name)
        self.cross_encoder = CrossEncoder(cross_encoder_model_name)

        self.embedding_dim = self.model.get_embedding_dimension()
        # if self.embedding_dim is None:
        #     test_vector = self.model.encode(["test"], normalize_embeddings=True)[0]
        #     self.embedding_dim = len(test_vector)

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
            results.append({
                "id": chunk.id,
                "score": score,
                "text": chunk.text,
                "metadata": chunk.metadata,
                "method": "bm25",
            })

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

    def retrieve_hybrid_candidates(
        self,
        query: str,
        bm25_k: int = 30,
        dense_k: int = 30,
        temporary_text: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve candidates from both BM25 and bi-encoder,
        then merge and deduplicate them by chunk ID.

        This does NOT rerank yet. It only creates the candidate pool
        that the cross-encoder will score.
        """
        if temporary_text:
            temp_ids = self.add_text(
                temporary_text,
                source="temporary_query_context",
                temporary=True,
            )

            try:
                return self.retrieve_hybrid_candidates(
                    query=query,
                    bm25_k=bm25_k,
                    dense_k=dense_k,
                )
            finally:
                self.remove(temp_ids)

        bm25_results = self.retrieve_bm25(query=query, top_k=bm25_k)
        dense_results = self.retrieve_biencoder(query=query, top_k=dense_k)

        merged: dict[str, dict[str, Any]] = {}

        for result in bm25_results:
            item = result.copy()
            item["retrieved_by"] = {"bm25"}
            item["bm25_score"] = item["score"]
            item["dense_score"] = None
            item["method"] = "hybrid_candidate"
            merged[item["id"]] = item

        for result in dense_results:
            if result["id"] in merged:
                merged[result["id"]]["retrieved_by"].add("biencoder")
                merged[result["id"]]["dense_score"] = result["score"]
            else:
                item = result.copy()
                item["retrieved_by"] = {"biencoder"}
                item["bm25_score"] = None
                item["dense_score"] = item["score"]
                item["method"] = "hybrid_candidate"
                merged[item["id"]] = item

        candidates = list(merged.values())

        for candidate in candidates:
            candidate["retrieved_by"] = sorted(candidate["retrieved_by"])

        return candidates

    def retrieve_cross_encoder(
        self,
        query: str,
        top_k: int = 5,
        bm25_k: int = 30,
        dense_k: int = 30,
        temporary_text: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve candidates using BM25 + bi-encoder,
        then rerank those candidates using a cross-encoder.

        Final result:
            top_k chunks ranked by cross-encoder relevance score.
        """
        candidates = self.retrieve_hybrid_candidates(
            query=query,
            bm25_k=bm25_k,
            dense_k=dense_k,
            temporary_text=temporary_text,
        )

        if not candidates:
            return []

        pairs = [
            [query, candidate["text"]]
            for candidate in candidates
        ]

        scores = self.cross_encoder.predict(pairs)

        reranked = []
        for candidate, score in zip(candidates, scores):
            item = candidate.copy()
            item["score"] = float(score)
            item["cross_encoder_score"] = float(score)
            item["method"] = "cross_encoder"
            reranked.append(item)

        reranked.sort(
            key=lambda item: item["cross_encoder_score"],
            reverse=True,
        )

        return reranked[:top_k]

    def query(
        self,
        query: str,
        top_k: int = 5,
        temporary_text: str | None = None,
        method: str = "cross_encoder",
    ) -> list[dict[str, Any]]:
        """
        Generic retrieval method.
        """
        if method == "bm25":
            return self.retrieve_bm25(
                query=query,
                top_k=top_k,
                temporary_text=temporary_text,
            )

        elif method == "biencoder":
            return self.retrieve_biencoder(
                query=query,
                top_k=top_k,
                temporary_text=temporary_text,
            )

        else:
            return self.retrieve_cross_encoder(
                query=query,
                top_k=top_k,
                temporary_text=temporary_text,
            )

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

    def answer(
        self,
        query: str,
        top_k: int = 5,
        method: str = "cross_encoder",
        temporary_text: str | None = None,
    ) -> str:
        
        """
        This does NOT generate a medical true/false answer.
        It only formats retrieved evidence chunks.
        """
        
        def _format_optional_score(score: float | None) -> str:
            if score is None:
                return "None"
            return f"{float(score):.4f}"
        
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
            metadata = result.get("metadata", {})
            source = metadata.get("source", "unknown")
            temporary = metadata.get("temporary", False)

            extra = ""

            if "bm25_score" in result:
                extra += f" bm25_score={_format_optional_score(result.get('bm25_score'))}"

            if "dense_score" in result:
                extra += f" dense_score={_format_optional_score(result.get('dense_score'))}"

            if "cross_encoder_score" in result:
                extra += f" cross_encoder_score={result['cross_encoder_score']:.4f}"

            if "retrieved_by" in result:
                extra += f" retrieved_by={result['retrieved_by']}"

            lines.append(
                f"[{i}] method={result['method']} "
                f"score={result['score']:.4f} "
                f"source={source} "
                f"temporary={temporary}"
                f"{extra}\n"
                f"{result['text']}"
            )

        return "\n\n".join(lines)

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

    Ibuprofen is a nonsteroidal anti-inflammatory drug used to reduce pain,
    fever, and inflammation. It may not be appropriate for some patients with
    kidney disease or stomach ulcers.
    """

    rag = RAG(document)

    claim = "Vitamin C prevents the common cold."

    print("=== BM25 ===")
    print(rag.answer(claim, method="bm25", top_k=3))

    print("\n=== BI-ENCODER ===")
    print(rag.answer(claim, method="biencoder", top_k=3))

    print("\n=== HYBRID CANDIDATES ===")
    print(rag.answer(claim, method="hybrid", top_k=3))

    print("\n=== CROSS-ENCODER RERANKED ===")
    print(rag.answer(claim, method="cross_encoder", top_k=3))

    print("\n=== TEMPORARY TEXT WITH CROSS-ENCODER ===")
    temporary_reference = """
    A temporary reference says that vitamin C did not significantly reduce
    the incidence of common colds among the general population.
    """

    print(
        rag.answer(
            claim,
            method="cross_encoder",
            top_k=3,
            temporary_text=temporary_reference,
        )
    )