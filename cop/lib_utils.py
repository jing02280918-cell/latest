"""Policy library: in-memory store with retrieval (default) or FAISS (legacy)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field


@dataclass
class PolicyEntry:
    condition: str
    primitive_actions: list
    option_name: str
    score_diff: float
    query: str = ""


class InMemoryPolicyLib:
    """Lightweight cross-sample policy store — no extra dependencies."""

    def __init__(self) -> None:
        self.entries: list[PolicyEntry] = []

    def add_documents(self, documents):
        return []


def construct_lib():
    if os.environ.get("COP_ENABLE_POLICY_LIB", "").lower() in {"1", "true", "yes"}:
        from langchain_huggingface import HuggingFaceEmbeddings
        import faiss
        from langchain_community.docstore.in_memory import InMemoryDocstore
        from langchain_community.vectorstores import FAISS

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
        return FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
    return InMemoryPolicyLib()


def save_policy_lib(vector_store, policy_l, action_l, option_name_l, diff_score, *, query: str = ""):
    if isinstance(vector_store, InMemoryPolicyLib):
        for ind in range(len(policy_l)):
            vector_store.entries.append(
                PolicyEntry(
                    condition=str(policy_l[ind]),
                    primitive_actions=list(action_l[ind]),
                    option_name=str(option_name_l[ind]),
                    score_diff=float(diff_score or 0),
                    query=str(query),
                )
            )
        return vector_store

    from langchain_core.documents import Document

    policy_documents = []
    for ind in range(len(policy_l)):
        policy_documents.append(
            Document(
                page_content=str(policy_l[ind]),
                metadata={
                    "option_name": option_name_l[ind],
                    "primitive_actions": json.dumps(action_l[ind]),
                    "score_diff": diff_score,
                },
            )
        )
    vector_store.add_documents(documents=policy_documents)
    return vector_store


def retreive_policy_lib(vector_store, query: str, k: int = 3) -> list[PolicyEntry]:
    if isinstance(vector_store, InMemoryPolicyLib):
        if not vector_store.entries:
            return []
        q_tokens = set(query)

        def rank(entry: PolicyEntry) -> tuple[float, float]:
            overlap = len(q_tokens & set(entry.query)) if entry.query else 0
            return (entry.score_diff, overlap)

        ranked = sorted(vector_store.entries, key=rank, reverse=True)
        return ranked[:k]

    hits = vector_store.similarity_search(query, k=k)
    out: list[PolicyEntry] = []
    for doc in hits:
        out.append(
            PolicyEntry(
                condition=doc.page_content,
                primitive_actions=json.loads(doc.metadata.get("primitive_actions", "[]")),
                option_name=str(doc.metadata.get("option_name", "")),
                score_diff=float(doc.metadata.get("score_diff", 0)),
                query=query,
            )
        )
    return out
