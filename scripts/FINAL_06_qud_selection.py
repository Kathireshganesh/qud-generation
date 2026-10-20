
import os
import json
import math
import re
from collections import defaultdict

from config import (
    ARTICLE_NAME,
    SALIENCE_FILE,
    SELECTED_QUD_FILE,
    OUTPUT_DIR
)


# ============================================================
# CONFIGURATION
# ============================================================

TOP_K = 2

# Ranking is based on salience.
# Keep this at 0.0 to avoid unnecessarily dropping candidates.
MIN_SALIENCE_SCORE = 0.0

# Minimum question length for structural quality checking.
MIN_QUESTION_WORDS = 4


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_question(question):
    """Normalize a question for exact duplicate detection."""
    return re.sub(
        r"[^a-z0-9]+",
        "",
        question.lower()
    )


def valid_number(value):
    """Check whether a value is a finite number between 0 and 1."""
    if isinstance(value, bool):
        return False

    try:
        number = float(value)
        return math.isfinite(number) and 0.0 <= number <= 1.0
    except (TypeError, ValueError):
        return False


def get_candidate_reasons(candidate):
    """Return structural or scoring problems for a candidate."""
    reasons = []

    question = candidate.get("question")
    score = candidate.get("salience_score")

    if not isinstance(question, str) or not question.strip():
        reasons.append("missing_or_empty_question")
    else:
        question = question.strip()

        if not question.endswith("?"):
            reasons.append("question_missing_question_mark")

        word_count = len(question.split())

        if word_count < MIN_QUESTION_WORDS:
            reasons.append("question_too_short")

    if not valid_number(score):
        reasons.append("missing_or_invalid_salience_score")
    elif float(score) < MIN_SALIENCE_SCORE:
        reasons.append("below_minimum_salience_score")

    return reasons


def atomic_save_json(filepath, content):
    """Save JSON safely to a temporary file, then replace the target."""
    directory = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(directory, exist_ok=True)

    temp_filepath = filepath + ".tmp"

    try:
        with open(
            temp_filepath,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                content,
                file,
                indent=2,
                ensure_ascii=False,
                allow_nan=False
            )

        os.replace(temp_filepath, filepath)

    finally:
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)


# ============================================================
# LOAD STAGE 5 OUTPUT
# ============================================================

if not os.path.isfile(SALIENCE_FILE):
    raise FileNotFoundError(
        f"Salience file not found:\n{SALIENCE_FILE}\n"
        "Run FINAL_05_qud_salience.py first."
    )

try:
    with open(
        SALIENCE_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        data = json.load(file)

except json.JSONDecodeError as error:
    raise ValueError(
        f"Invalid JSON in salience file: {error}"
    ) from error


if not isinstance(data, dict):
    raise ValueError(
        "Expected a JSON object from Stage 5, "
        "containing a 'pairs' list."
    )

pairs = data.get("pairs")

if not isinstance(pairs, list):
    raise ValueError(
        "The Stage 5 output does not contain a valid 'pairs' list."
    )


# ============================================================
# SELECTION
# ============================================================

selected_results = []

total_candidates = 0
selected_count = 0
excluded_count = 0
pairs_with_selection = 0
pairs_without_selection = 0

# Exact duplicates are removed within the same EDU pair.
# Cross-pair duplicates are retained to preserve EDU coverage.
global_question_occurrences = defaultdict(list)

print("=" * 75)
print("FINAL QUD SELECTION — STAGE 6")
print("=" * 75)
print(f"Article       : {ARTICLE_NAME}")
print(f"Input pairs   : {len(pairs)}")
print(f"Top K         : {TOP_K}")
print(f"Minimum score : {MIN_SALIENCE_SCORE}")


for pair_index, item in enumerate(pairs, start=1):

    if not isinstance(item, dict):
        print(
            f"\nWARNING: Skipping malformed pair at index {pair_index}."
        )
        continue

    pair_id = item.get("pair_id", pair_index)
    edu1 = item.get("edu1")
    edu2 = item.get("edu2")

    candidates = item.get("candidates", [])

    if not isinstance(candidates, list):
        candidates = []

    print("\n" + "=" * 75)
    print(f"PAIR {pair_id}: EDU {edu1} -> EDU {edu2}")
    print("=" * 75)

    total_candidates += len(candidates)

    eligible = []
    excluded_candidates = []
    seen_in_pair = set()

    # --------------------------------------------------------
    # VALIDATE AND PREPARE CANDIDATES
    # --------------------------------------------------------

    for candidate_index, candidate in enumerate(candidates):

        if not isinstance(candidate, dict):
            excluded_count += 1

            excluded_candidates.append({
                "candidate_index": candidate_index,
                "question": None,
                "salience_score": None,
                "exclusion_reasons": ["candidate_is_not_an_object"]
            })

            continue

        question = candidate.get("question")
        reasons = get_candidate_reasons(candidate)

        if reasons:
            excluded_count += 1

            excluded_candidates.append({
                "candidate_index": candidate_index,
                "question": question,
                "salience_score": candidate.get("salience_score"),
                "exclusion_reasons": reasons
            })

            continue

        normalized = normalize_question(question)

        # Ignore repeated normalized questions within this pair.
        if normalized in seen_in_pair:
            excluded_count += 1

            excluded_candidates.append({
                "candidate_index": candidate_index,
                "question": question,
                "salience_score": candidate.get("salience_score"),
                "exclusion_reasons": [
                    "duplicate_question_within_same_pair"
                ]
            })

            continue

        seen_in_pair.add(normalized)

        # Track article-wide duplicate occurrences for reporting.
        global_question_occurrences[normalized].append({
            "pair_id": pair_id,
            "candidate_index": candidate_index,
            "question": question
        })

        eligible.append({
            **candidate,
            "candidate_index": candidate_index,
            "salience_score": float(candidate["salience_score"])
        })

    # --------------------------------------------------------
    # RANK CANDIDATES
    # --------------------------------------------------------

    ranked = sorted(
        eligible,
        key=lambda candidate: (
            candidate["salience_score"],
            float(candidate.get("grounding_score", 0.0))
            if valid_number(candidate.get("grounding_score"))
            else 0.0,
            -candidate["candidate_index"]
        ),
        reverse=True
    )

    selected_candidates = ranked[:TOP_K]

    # --------------------------------------------------------
    # STORE SELECTED CANDIDATES
    # --------------------------------------------------------

    formatted_selected = []

    for rank, candidate in enumerate(
        selected_candidates,
        start=1
    ):
        formatted_candidate = {
            **candidate,
            "selection_rank": rank,
            "selection_reason": (
                f"Ranked #{rank} by salience score "
                "among structurally valid candidates."
            )
        }

        formatted_selected.append(formatted_candidate)

        selected_count += 1

        print(f"\nSELECTED QUD #{rank}")
        print(f"Question : {candidate['question']}")
        print(f"Salience : {candidate['salience_score']:.4f}")

        if valid_number(candidate.get("grounding_score")):
            print(
                f"Grounding: "
                f"{float(candidate['grounding_score']):.4f}"
            )

    # --------------------------------------------------------
    # PAIR STATUS
    # --------------------------------------------------------

    if len(formatted_selected) == TOP_K:
        selection_status = "selected_top_k"

    elif len(formatted_selected) > 0:
        selection_status = "selected_fewer_than_top_k"

    else:
        selection_status = "no_eligible_qud"

    if formatted_selected:
        pairs_with_selection += 1
    else:
        pairs_without_selection += 1

    # --------------------------------------------------------
    # PRESERVE PAIR INFORMATION
    # --------------------------------------------------------

    selected_results.append({
        "pair_id": pair_id,
        "edu1": edu1,
        "edu2": edu2,
        "text1": item.get("text1", ""),
        "text2": item.get("text2", ""),
        "pair_type": item.get("pair_type", "unknown"),
        "semantic_similarity": item.get("semantic_similarity"),
        "input_candidate_count": len(candidates),
        "eligible_candidate_count": len(eligible),
        "selected_count": len(formatted_selected),
        "selected_candidates": formatted_selected,
        "excluded_candidates": excluded_candidates,
        "selection_status": selection_status
    })

    print(f"\nEligible candidates: {len(eligible)}")
    print(f"Selected candidates: {len(formatted_selected)}")
    print(f"Excluded candidates: {len(excluded_candidates)}")
    print(f"Status: {selection_status}")


# ============================================================
# ARTICLE-WIDE DUPLICATE REPORT
# ============================================================

duplicate_questions = []

for normalized_question, occurrences in (
    global_question_occurrences.items()
):
    if len(occurrences) > 1:
        duplicate_questions.append({
            "normalized_question": normalized_question,
            "occurrences": occurrences,
            "count": len(occurrences)
        })


# ============================================================
# BUILD OUTPUT
# ============================================================

output_data = {
    "article_id": data.get("article_id", ARTICLE_NAME),
    "stage": "FINAL_06_qud_selection",
    "selection_method": "top_k_by_salience",
    "configuration": {
        "top_k": TOP_K,
        "minimum_salience_score": MIN_SALIENCE_SCORE,
        "minimum_question_words": MIN_QUESTION_WORDS,
        "duplicate_policy": (
            "Remove exact duplicates within each pair; "
            "retain cross-pair duplicates and report them."
        )
    },
    "summary": {
        "total_pairs": len(selected_results),
        "total_input_candidates": total_candidates,
        "total_selected_quds": selected_count,
        "total_excluded_candidates": excluded_count,
        "pairs_with_selection": pairs_with_selection,
        "pairs_without_selection": pairs_without_selection,
        "cross_pair_duplicate_groups": len(duplicate_questions)
    },
    "duplicate_questions": duplicate_questions,
    "pairs": selected_results
}


# ============================================================
# SAVE OUTPUT
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

atomic_save_json(
    SELECTED_QUD_FILE,
    output_data
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("STAGE 6 — QUD SELECTION COMPLETE")
print("=" * 75)

print(f"Article                   : {ARTICLE_NAME}")
print(f"Total EDU pairs           : {len(selected_results)}")
print(f"Input candidates          : {total_candidates}")
print(f"Selected QUDs             : {selected_count}")
print(f"Excluded candidates       : {excluded_count}")
print(f"Pairs with selected QUDs  : {pairs_with_selection}")
print(f"Pairs without selected QUD: {pairs_without_selection}")
print(f"Cross-pair duplicate groups: {len(duplicate_questions)}")

print("\nSaved to:")
print(SELECTED_QUD_FILE)

print("=" * 75)
