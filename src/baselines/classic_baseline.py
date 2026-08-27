"""Phase 6 arm 4 — classic (non-LLM) baseline (PROJECT_SPEC.md §5 Phase 6 #4).

Method: TF-IDF nearest-neighbour retrieval. Fit a TF-IDF vectorizer on the
*training* notes, then for each test note return the conversation attached
to whichever training note is most cosine-similar. No neural model, no
fine-tuning, no GPU, fully deterministic.

**Why this is the right classic baseline for this task.** The spec asks for
"the field's standard tool for this exact task." Retrieval is that tool for
conditional generation over a corpus with strong house style: NoteChat's
dialogues are formulaic (a doctor restating the note's facts as questions),
so a retrieved dialogue from a clinically similar case is already
well-formed, on-style, and topically close — precisely what ROUGE and
BERTScore reward. If a fine-tuned 3B model can't clear this bar, the honest
conclusion is that an LLM was not needed here, which §5 Phase 6 explicitly
names as a legitimate and valuable finding rather than a failure.

**What it structurally cannot do:** the returned dialogue describes a
*different patient*. Names, ages, numbers, and findings will be those of the
retrieved case, not the query note. Expect high surface-form scores and poor
factual grounding — the exact gap that reference-similarity metrics are
blind to (see PROJECT_SPEC.md §4.2 and the faithfulness metric in
src/eval/faithfulness.py, which is what actually separates this arm from
the generative ones).
"""

import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class TfidfRetrievalBaseline:
    """Nearest-neighbour retrieval over training notes.

    Deliberately a plain class rather than a sklearn Pipeline: the "model"
    here is just the fitted vectorizer plus the training conversations, and
    keeping it explicit makes what's being retrieved obvious at the call
    site.
    """

    def __init__(self, max_features: int = 50_000, ngram_range: tuple[int, int] = (1, 2)):
        # Sublinear tf + English stopwords: clinical notes repeat common
        # scaffolding ("the patient was admitted...") heavily, so raw term
        # counts over-weight boilerplate that carries no retrieval signal.
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english",
            sublinear_tf=True,
        )
        self.train_matrix = None
        self.train_conversations: list[str] = []
        self.train_note_ids: list[str] = []

    def fit(self, train_df: pl.DataFrame) -> "TfidfRetrievalBaseline":
        self.train_conversations = train_df["conversation"].to_list()
        self.train_note_ids = train_df["note_id"].to_list()
        self.train_matrix = self.vectorizer.fit_transform(train_df["clinical_note"].to_list())
        return self

    def predict_one(self, clinical_note: str) -> dict:
        """Return the conversation of the most similar training note.

        Also returns which note it came from and the similarity score, so a
        results file can be audited after the fact rather than just showing
        an unexplained string.
        """
        if self.train_matrix is None:
            raise RuntimeError("fit() must be called before predict_one()")
        query_vec = self.vectorizer.transform([clinical_note])
        # linear_kernel == cosine similarity here: TfidfVectorizer L2-normalizes
        # its output by default, so the dot product is already the cosine.
        similarities = linear_kernel(query_vec, self.train_matrix).ravel()
        best_idx = int(similarities.argmax())
        return {
            "generated": self.train_conversations[best_idx],
            "retrieved_note_id": self.train_note_ids[best_idx],
            "retrieval_similarity": float(similarities[best_idx]),
        }
