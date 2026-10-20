"""
FINAL_05_qud_salience.py

Stage 5: Salience Scoring for QUD Candidates

Input:
    output/article03/validated_quds.json

Output:
    output/article03/salience_result.json

Expected pair fields:
    pair_id, edu1, edu2, text1, text2,
    semantic_similarity, candidates

Expected candidate fields:
    question, grounding_score,
    edu1_similarity, edu2_similarity

No additional ML model is required for this stage.
"""

import json
import re
import sys
import tempfile
from pathlib import Path
from collections import Counter


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

ARTICLE_ID = "article03"

INPUT_FILE = (
    BASE_DIR
    / "output"
    / ARTICLE_ID
    / "validated_quds.json"
)

OUTPUT_FILE = (
    BASE_DIR
    / "output"
    / ARTICLE_ID
    / "salience_result.json"
)

# All component scores are treated as 0–1 values.
# The observed semantic_similarity value in your input is 0.8449.
PAIR_SIMILARITY_RANGE = "zero_to_one"

# Weighted salience components.
WEIGHTS = {
    "grounding_score": 0.35,
    "edu1_similarity": 0.20,
    "edu2_similarity": 0.20,
    "pair_similarity": 0.15,
    "question_quality": 0.10,
}

# Minimum number of meaningful words in a question.
MIN_QUESTION_WORDS = 4

# Maximum number of words used by the question-quality heuristic.
MAX_QUESTION_WORDS = 35


# ============================================================
# 2. GENERAL UTILITIES
# ============================================================

def print_header(message):
    print("\n" + "=" * 65)
    print(message)
    print("=" * 65)


def load_json(path):
    """Load JSON with a clear error message."""

    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{path}"
        )

    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON in {path}\n"
            f"Line {error.lineno}, column {error.colno}: "
            f"{error.msg}"
        ) from error


def save_json_atomic(data, path):
    """
    Write JSON to a temporary file first, then replace the
    destination to reduce the risk of a partially written file.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            suffix=".tmp",
            delete=False,
        ) as temporary_file:

            temporary_path = Path(temporary_file.name)

            json.dump(
                data,
                temporary_file,
                ensure_ascii=False,
                indent=2,
            )

            temporary_file.write("\n")

        temporary_path.replace(path)

    finally:
        if (
            temporary_path is not None
            and temporary_path.exists()
        ):
            temporary_path.unlink()


def to_valid_score(value):
    """
    Convert a numeric score to a float in [0, 1].

    Return None if the value is missing, non-numeric,
    NaN, infinite, or outside the expected range.
    """

    if isinstance(value, bool):
        return None

    try:
        score = float(value)
    except (TypeError, ValueError):
        return None

    # Reject NaN and infinity without requiring NumPy.
    if score != score or score in (float("inf"), float("-inf")):
        return None

    if not 0.0 <= score <= 1.0:
        return None

    return score


# ============================================================
# 3. QUESTION QUALITY HEURISTIC
# ============================================================

def question_quality_score(question):
    """
    Estimate question-form quality using simple rules.

    This is a heuristic, not a trained language model.
    It checks:
        - Whether the question is present
        - Whether it has a question mark
        - Whether it starts with a useful question word
        - Whether its length is reasonable
        - Whether it contains enough words
    """

    if not isinstance(question, str):
        return 0.0

    question = question.strip()

    if not question:
        return 0.0

    words = re.findall(r"\b[\w'-]+\b", question.lower())

    if not words:
        return 0.0

    score = 0.0

    # Starts with a recognized question word.
    question_starters = {
        "who",
        "what",
        "when",
        "where",
        "why",
        "how",
        "which",
        "whom",
        "whose",
        "is",
        "are",
        "was",
        "were",
        "do",
        "does",
        "did",
        "can",
        "could",
        "will",
        "would",
        "should",
        "has",
        "have",
        "had",
    }

    if words[0] in question_starters:
        score += 0.35

    # Ends with a question mark.
    if question.endswith("?"):
        score += 0.25

    # Reasonable question length.
    word_count = len(words)

    if MIN_QUESTION_WORDS <= word_count <= MAX_QUESTION_WORDS:
        score += 0.30
    elif word_count >= 2:
        score += 0.15

    # Avoid questions consisting of only a question word
    # or a very short fragment.
    if word_count >= 4:
        score += 0.10

    return round(min(score, 1.0), 4)


# ============================================================
# 4. PAIR SIMILARITY
# ============================================================

def get_pair_similarity(pair):
    """
    Read semantic similarity from the actual input schema.

    Your validated_quds.json uses:
        semantic_similarity

    The current configuration treats this value as 0–1.
    """

    raw_score = pair.get("semantic_similarity")

    if raw_score is None:
        return None

    score = to_valid_score(raw_score)

    if score is None:
        return None

    if PAIR_SIMILARITY_RANGE == "zero_to_one":
        return score

    if PAIR_SIMILARITY_RANGE == "minus1_to_1":
        # Convert cosine similarity [-1, 1] to [0, 1].
        # This option should only be used if the input
        # actually stores untransformed cosine similarity.
        raw_cosine = float(raw_score)

        if not -1.0 <= raw_cosine <= 1.0:
            return None

        return (raw_cosine + 1.0) / 2.0

    raise ValueError(
        "PAIR_SIMILARITY_RANGE must be "
        "'zero_to_one' or 'minus1_to_1'."
    )


# ============================================================
# 5. SALIENCE SCORE
# ============================================================

def score_candidate(candidate, pair, candidate_index):
    """
    Calculate a weighted salience score for one QUD candidate.

    Formula:

        Salience =
            0.35 * grounding_score
          + 0.20 * edu1_similarity
          + 0.20 * edu2_similarity
          + 0.15 * pair_similarity
          + 0.10 * question_quality

    Every component must be available and valid.
    Incomplete candidates receive an explicit status.
    """

    if not isinstance(candidate, dict):
        return {
            "candidate_index": candidate_index,
            "salience_status": "INVALID_CANDIDATE",
            "salience_score": None,
            "salience_components": {},
            "salience_missing_fields": [
                "candidate must be a JSON object"
            ],
        }

    question = candidate.get("question", "")

    grounding = to_valid_score(
        candidate.get("grounding_score")
    )

    edu1_similarity = to_valid_score(
        candidate.get("edu1_similarity")
    )

    edu2_similarity = to_valid_score(
        candidate.get("edu2_similarity")
    )

    pair_similarity = get_pair_similarity(pair)

    quality = question_quality_score(question)

    components = {
        "grounding_score": grounding,
        "edu1_similarity": edu1_similarity,
        "edu2_similarity": edu2_similarity,
        "pair_similarity": pair_similarity,
        "question_quality": quality,
    }

    missing_fields = [
        name
        for name, value in components.items()
        if value is None
    ]

    # Record the question-quality heuristic even when
    # another required input score is unavailable.
    candidate["question_quality_score"] = quality

    if missing_fields:
        candidate["salience_status"] = "INCOMPLETE"
        candidate["salience_score"] = None
        candidate["salience_components"] = components
        candidate["salience_missing_fields"] = missing_fields

        return {
            "candidate_index": candidate_index,
            "salience_status": "INCOMPLETE",
            "salience_score": None,
            "salience_components": components,
            "salience_missing_fields": missing_fields,
        }

    # Weighted sum.
    salience = sum(
        components[name] * weight
        for name, weight in WEIGHTS.items()
    )

    salience = round(salience, 4)

    # Store the results directly on the candidate.
    candidate["salience_score"] = salience
    candidate["salience_status"] = "SCORED"
    candidate["salience_components"] = {
        name: round(value, 4)
        for name, value in components.items()
    }

    # Remove stale error information if the candidate
    # has now been scored successfully.
    candidate.pop("salience_missing_fields", None)

    return {
        "candidate_index": candidate_index,
        "salience_status": "SCORED",
        "salience_score": salience,
        "salience_components": candidate["salience_components"],
        "salience_missing_fields": [],
    }


# ============================================================
# 6. DUPLICATE QUESTION DETECTION
# ============================================================

def normalize_question(question):
    """Normalize text for duplicate detection."""

    if not isinstance(question, str):
        return ""

    normalized = question.lower().strip()
    normalized = re.sub(r"[^\w\s]", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)

    return normalized


def find_duplicate_questions(data):
    """
    Find exact duplicates after lowercase and punctuation
    normalization.

    This does not detect semantic duplicates with different
    wording.
    """

    locations = {}

    for pair_index, pair in enumerate(data):
        if not isinstance(pair, dict):
            continue

        pair_id = pair.get("pair_id", pair_index + 1)

        candidates = pair.get("candidates", [])

        if not isinstance(candidates, list):
            continue

        for candidate_index, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                continue

            question = candidate.get("question", "")
            normalized = normalize_question(question)

            if not normalized:
                continue

            locations.setdefault(normalized, []).append({
                "pair_id": pair_id,
                "candidate_index": candidate_index,
                "question": question,
            })

    duplicates = []

    for normalized, occurrences in locations.items():
        if len(occurrences) > 1:
            duplicates.append({
                "normalized_question": normalized,
                "occurrences": occurrences,
                "count": len(occurrences),
            })

    return duplicates


# ============================================================
# 7. MAIN PIPELINE
# ============================================================

def main():

    print_header("STAGE 5: QUD SALIENCE SCORING")

    print(f"Project directory : {BASE_DIR}")
    print(f"Input file        : {INPUT_FILE}")
    print(f"Output file       : {OUTPUT_FILE}")

    # --------------------------------------------------------
    # Load input
    # --------------------------------------------------------

    try:
        data = load_json(INPUT_FILE)

    except (FileNotFoundError, ValueError) as error:
        print(f"\nERROR: {error}")
        sys.exit(1)

    if not isinstance(data, list):
        print(
            "\nERROR: Expected validated_quds.json "
            "to contain a list of EDU-pair objects."
        )
        sys.exit(1)

    print(f"\nEDU pairs loaded: {len(data)}")

    # --------------------------------------------------------
    # Score candidates
    # --------------------------------------------------------

    total_candidates = 0
    scored_candidates = 0
    incomplete_candidates = 0
    invalid_pairs = 0

    all_scores = []
    missing_field_counts = Counter()
    pair_summaries = []

    for pair_index, pair in enumerate(data):

        if not isinstance(pair, dict):
            invalid_pairs += 1

            print(
                f"WARNING: Pair at index {pair_index} "
                "is not a JSON object. Skipping."
            )
            continue

        pair_id = pair.get("pair_id", pair_index + 1)

        candidates = pair.get("candidates", [])

        if not isinstance(candidates, list):
            print(
                f"WARNING: Pair {pair_id} has an invalid "
                "'candidates' field. Skipping its candidates."
            )

            pair["salience_summary"] = {
                "candidate_count": 0,
                "scored_count": 0,
                "incomplete_count": 0,
                "mean_salience": None,
                "max_salience": None,
            }

            continue

        pair_scored = 0
        pair_incomplete = 0
        pair_scores = []

        for candidate_index, candidate in enumerate(candidates):

            total_candidates += 1

            result = score_candidate(
                candidate=candidate,
                pair=pair,
                candidate_index=candidate_index,
            )

            if result["salience_status"] == "SCORED":

                scored_candidates += 1
                pair_scored += 1

                score = result["salience_score"]

                pair_scores.append(score)
                all_scores.append(score)

            else:

                incomplete_candidates += 1
                pair_incomplete += 1

                for field in result.get(
                    "salience_missing_fields", []
                ):
                    missing_field_counts[field] += 1

        # Summary for this EDU pair.
        mean_score = (
            round(sum(pair_scores) / len(pair_scores), 4)
            if pair_scores
            else None
        )

        max_score = (
            round(max(pair_scores), 4)
            if pair_scores
            else None
        )

        pair["salience_summary"] = {
            "candidate_count": len(candidates),
            "scored_count": pair_scored,
            "incomplete_count": pair_incomplete,
            "mean_salience": mean_score,
            "max_salience": max_score,
        }

        pair_summaries.append({
            "pair_id": pair_id,
            "candidate_count": len(candidates),
            "scored_count": pair_scored,
            "incomplete_count": pair_incomplete,
            "mean_salience": mean_score,
            "max_salience": max_score,
        })

        print(
            f"Pair {str(pair_id):>3} | "
            f"Candidates: {len(candidates):>3} | "
            f"Scored: {pair_scored:>3} | "
            f"Incomplete: {pair_incomplete:>3}"
        )

    # --------------------------------------------------------
    # Duplicate detection
    # --------------------------------------------------------

    duplicates = find_duplicate_questions(data)

    # --------------------------------------------------------
    # Overall summary
    # --------------------------------------------------------

    mean_salience = (
        round(sum(all_scores) / len(all_scores), 4)
        if all_scores
        else None
    )

    min_salience = (
        round(min(all_scores), 4)
        if all_scores
        else None
    )

    max_salience = (
        round(max(all_scores), 4)
        if all_scores
        else None
    )

    summary = {
        "stage": "FINAL_05_qud_salience",
        "article_id": ARTICLE_ID,
        "input_file": str(INPUT_FILE),
        "output_file": str(OUTPUT_FILE),
        "pair_count": len(data),
        "invalid_pair_count": invalid_pairs,
        "total_candidates": total_candidates,
        "scored_candidates": scored_candidates,
        "incomplete_candidates": incomplete_candidates,
        "scoring_weights": WEIGHTS,
        "pair_similarity_range": PAIR_SIMILARITY_RANGE,
        "mean_salience": mean_salience,
        "minimum_salience": min_salience,
        "maximum_salience": max_salience,
        "duplicate_question_group_count": len(duplicates),
        "missing_field_counts": dict(missing_field_counts),
    }

    # --------------------------------------------------------
    # Save output
    # --------------------------------------------------------

    output_data = {
        "article_id": ARTICLE_ID,
        "stage": "FINAL_05_qud_salience",
        "scoring_method": (
            "Weighted sum of grounding, EDU similarities, "
            "pair similarity, and question-form quality"
        ),
        "scoring_weights": WEIGHTS,
        "summary": summary,
        "pair_summaries": pair_summaries,
        "duplicate_questions": duplicates,
        "pairs": data,
    }

    try:
        save_json_atomic(output_data, OUTPUT_FILE)

    except OSError as error:
        print(f"\nERROR: Could not save output: {error}")
        sys.exit(1)

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print_header("STAGE 5 COMPLETED")

    print(f"Total EDU pairs       : {len(data)}")
    print(f"Total candidates      : {total_candidates}")
    print(f"Scored candidates     : {scored_candidates}")
    print(f"Incomplete candidates : {incomplete_candidates}")

    print(f"\nMean salience         : {mean_salience}")
    print(f"Minimum salience      : {min_salience}")
    print(f"Maximum salience      : {max_salience}")

    print(
        "Duplicate groups     : "
        f"{len(duplicates)}"
    )

    if missing_field_counts:
        print("\nMissing or invalid fields:")

        for field, count in missing_field_counts.items():
            print(f"  {field}: {count}")

    print(f"\nOutput saved to:\n{OUTPUT_FILE}")

    if total_candidates > 0 and scored_candidates == 0:
        print(
            "\nWARNING: No candidates received salience scores. "
            "Check the input score fields and pair similarity."
        )

    elif incomplete_candidates > 0:
        print(
            "\nNOTE: Some candidates could not be scored. "
            "Inspect their salience_missing_fields values."
        )

    else:
        print(
            "\nSUCCESS: Every candidate was scored successfully."
        )

    print("\nNext step: inspect salience_result.json before Stage 6.")


# ============================================================
# 8. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()