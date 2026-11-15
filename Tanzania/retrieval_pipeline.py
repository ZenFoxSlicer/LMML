"""Repaired retrieval pipeline for the Tanzania agents task.

What was broken in the starter, and how it is fixed:

1. Two different embedding models. Documents were turned into vectors by
   all-MiniLM-L6-v2 and questions by e5-small-v2. Each model has its own
   "coordinate system", so comparing a MiniLM vector with an e5 vector is
   like comparing GPS coordinates with street numbers: the numbers exist but
   mean nothing together. Fix: one model (e5-small-v2) encodes both sides.

2. Missing e5 prefixes. e5 was trained to see "query: ..." for questions
   and "passage: ..." for documents. The starter only added "query:".
   Fix: documents now get the "passage: " prefix.

3. Blind 180-character chunks. Slicing text every 180 characters cuts
   sentences in half and separates the title ("CURRENT ACCESS NOTICE: X,
   City") from the access details, so a fragment cannot tell whether it
   is the current notice or an archived one, nor which city it is for.
   Fix: each document (title + full text, ~250 words, well under e5's
   512-token limit) is embedded as one piece, so one vector = one document.

4. Archived look-alikes. Even with 1-3 fixed, 12 questions matched the
   "ARCHIVED BACKGROUND" twin of the right notice (same exhibit, same city).
   Fix: drop archived documents before searching (see main()).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from flag_utils import derive_flag, load_verification_codes

# FIX 1: the same model must encode questions and documents.
MODEL = "intfloat/e5-small-v2"


@dataclass(frozen=True)
class Document:
    """One searchable source document."""

    document_id: str
    title: str
    text: str


@dataclass(frozen=True)
class Chunk:
    """One searchable fragment and its source document ID."""

    document_id: str
    text: str


def load_corpus(corpus_path: Path) -> list[Document]:
    """Read source documents from a JSON Lines corpus.

    :param corpus_path: Corpus file supplied with the task.
    :return: Searchable documents in file order.
    """
    documents: list[Document] = []
    with corpus_path.open(encoding="utf-8") as corpus_file:
        for line in corpus_file:
            row = json.loads(line)
            documents.append(Document(row["document_id"], row["title"], row["text"]))
    return documents


def load_questions(questions_path: Path) -> list[dict[str, str]]:
    """Read the ordered retrieval questions.

    :param questions_path: JSON question file supplied with the task.
    :return: Question objects in official order.
    """
    with questions_path.open(encoding="utf-8") as questions_file:
        questions = json.load(questions_file)
    if not isinstance(questions, list):
        raise ValueError("questions.json must contain a list.")
    return questions


def split_into_chunks(documents: list[Document]) -> list[Chunk]:
    """Turn every document into exactly one searchable piece.

    FIX 3: no more fixed 180-character slices; the title stays together
    with the access details so the whole notice is judged at once.

    :param documents: Complete source documents.
    :return: One chunk per document (title + text).
    """
    return [
        Chunk(document.document_id, f"{document.title}\n{document.text}")
        for document in documents
    ]


def retrieve_documents(
    chunks: list[Chunk],
    questions: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Return one source document for every question.

    :param chunks: Searchable text fragments.
    :param questions: Ordered task questions.
    :return: Top source document ID for each question.
    """
    from sentence_transformers import SentenceTransformer

    # FIX 1: a single encoder, so both sides live in the same vector space.
    encoder = SentenceTransformer(MODEL)
    # FIX 2: e5 expects "passage: " in front of documents.
    chunk_vectors = encoder.encode(
        [f"passage: {chunk.text}" for chunk in chunks],
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    query_vectors = encoder.encode(
        [f"query: {question['question']}" for question in questions],
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    # Vectors are normalised, so the dot product is the cosine similarity:
    # 1.0 = same meaning, 0 = unrelated. Each question takes its best match.
    similarities = np.asarray(query_vectors) @ np.asarray(chunk_vectors).T

    results: list[dict[str, str]] = []
    for question, row in zip(questions, similarities):
        best_chunk_index = int(np.argmax(row))
        results.append(
            {
                "question_id": question["question_id"],
                "document_id": chunks[best_chunk_index].document_id,
            }
        )
    return results


def print_sanity_check(
    results: list[dict[str, str]],
    questions: list[dict[str, str]],
    documents: list[Document],
) -> None:
    """Count answers that are the CURRENT notice for the asked exhibit and city.

    Every question names one exhibit and one city, and the right answer is
    the document titled "CURRENT ACCESS NOTICE: <exhibit>, <city>". This
    check only reads titles; it is evidence, not part of retrieval.
    """
    titles = {document.document_id: document.title for document in documents}
    cities = ["Dar es Salaam", "Arusha", "Mwanza"]
    correct = 0
    for question, result in zip(questions, results):
        title = titles[result["document_id"]]
        expected_prefix = "CURRENT ACCESS NOTICE: "
        exhibit_city = title.removeprefix(expected_prefix)
        exhibit, _, city = exhibit_city.rpartition(", ")
        asked_cities = [c for c in cities if c in question["question"]]
        ok = (
            title.startswith(expected_prefix)
            and exhibit in question["question"]
            and asked_cities == [city]
        )
        correct += ok
        if not ok:
            print(f"  CHECK {question['question_id']}: got '{title}'")
    print(f"Sanity check: {correct}/{len(results)} answers are the current notice "
          "for the asked exhibit and city.")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line file locations.

    :return: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("corpus.jsonl"))
    parser.add_argument("--questions", type=Path, default=Path("questions.json"))
    parser.add_argument("--output", type=Path, default=Path("retrieved_ids.json"))
    return parser.parse_args()


def main() -> None:
    """Run retrieval, save IDs, and print the derived flag."""
    arguments = parse_arguments()
    documents = load_corpus(arguments.corpus)
    questions = load_questions(arguments.questions)
    # FIX 4: metadata filter. Every question asks for the *current* access
    # notice. Archived background notices describe the same exhibit in the
    # same city with almost the same words, so embeddings alone pick them
    # for 12/72 questions. They state themselves that they "do not replace
    # the current access notice", so they are removed before searching.
    searchable = [
        document for document in documents
        if not document.title.startswith("ARCHIVED")
    ]
    results = retrieve_documents(split_into_chunks(searchable), questions)
    arguments.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    flag = derive_flag(results, questions, load_verification_codes(arguments.corpus))
    print(f"Saved {len(results)} results to {arguments.output}.")
    print_sanity_check(results, questions, documents)
    print(flag)


if __name__ == "__main__":
    main()
