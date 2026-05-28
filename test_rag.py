from proj.rag_old import RAG
document = """
RAG stands for retrieval-augmented generation.
It retrieves relevant context from a knowledge base before generating an answer.
TF-IDF is a sparse keyword-based representation.
Cosine similarity compares the angle between normalized vectors.
"""
rag = RAG(document)

bm25_chunks = rag.retrieve_bm25(
    "Vitamin C prevents colds",
    top_k=5,
)

dense_chunks = rag.retrieve_biencoder(
    "Vitamin C prevents colds",
    top_k=5,
)