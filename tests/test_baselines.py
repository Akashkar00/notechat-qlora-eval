"""Phase 6 arm 4 tests — classic non-LLM baseline (PROJECT_SPEC.md §5 Phase 6 #4).

The retrieval baseline is the arm most likely to be wrong in a way that
flatters it (retrieve from the wrong pool and it trivially "wins"), so its
behaviour is pinned down here rather than trusted.
"""

import polars as pl
import pytest

from src.baselines.classic_baseline import TfidfRetrievalBaseline


def _toy_train_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "note_id": ["note-cardiac", "note-fracture", "note-derm"],
            "clinical_note": [
                "A 64-year-old man with chest pain and coronary heart disease, hypertension.",
                "A 12-year-old boy fell and sustained a fracture of the left radius bone.",
                "A 30-year-old woman with an itchy erythematous skin rash on the forearm.",
            ],
            "conversation": [
                "Doctor: Tell me about the chest pain.\nPatient: It started yesterday.",
                "Doctor: How did you break your arm?\nPatient: I fell off my bike.",
                "Doctor: How long has the rash been there?\nPatient: About a week.",
            ],
        }
    )


def test_retrieves_topically_nearest_training_note():
    retriever = TfidfRetrievalBaseline().fit(_toy_train_df())
    result = retriever.predict_one("A 70-year-old man with chest pain and coronary heart disease.")
    assert result["retrieved_note_id"] == "note-cardiac"
    assert "chest pain" in result["generated"]


def test_returns_the_retrieved_notes_conversation_verbatim():
    train_df = _toy_train_df()
    retriever = TfidfRetrievalBaseline().fit(train_df)
    result = retriever.predict_one("fracture of the radius bone after a fall")
    expected = train_df.filter(pl.col("note_id") == "note-fracture")["conversation"][0]
    assert result["generated"] == expected


def test_similarity_is_a_cosine_in_zero_to_one():
    retriever = TfidfRetrievalBaseline().fit(_toy_train_df())
    result = retriever.predict_one("skin rash itchy erythematous forearm")
    assert 0.0 <= result["retrieval_similarity"] <= 1.0
    # An almost-verbatim query should score far above an unrelated one.
    unrelated = retriever.predict_one("quarterly financial statements and revenue projections")
    assert result["retrieval_similarity"] > unrelated["retrieval_similarity"]


def test_is_deterministic():
    retriever = TfidfRetrievalBaseline().fit(_toy_train_df())
    query = "A 64-year-old man with chest pain."
    assert retriever.predict_one(query) == retriever.predict_one(query)


def test_predict_before_fit_fails_loudly():
    # Prefer failing loudly over defaulting silently (PROJECT_SPEC.md §6).
    with pytest.raises(RuntimeError):
        TfidfRetrievalBaseline().predict_one("anything")


def test_retrieval_carries_the_wrong_patients_numbers():
    """The structural weakness this arm exists to demonstrate.

    A retrieved dialogue is well-formed and on-topic but describes a
    different patient, so its numbers belong to the retrieved case. This is
    invisible to ROUGE/BERTScore and is exactly what
    `src/eval/faithfulness.py` is for — pinned as a test so the point can't
    quietly regress into "retrieval is fine, actually."
    """
    from src.eval.faithfulness import numeric_faithfulness

    retriever = TfidfRetrievalBaseline().fit(_toy_train_df())
    query_note = "A 70-year-old man with chest pain and coronary heart disease."
    result = retriever.predict_one(query_note)
    faith = numeric_faithfulness(result["generated"], query_note)
    # The query note's age (70) appears nowhere in the retrieved dialogue.
    assert faith["numeric_grounding_recall"] == 0.0
