import os
import json
import re

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    ARTICLE_NAME,
    EDU_FILE,
    QUD_CANDIDATE_FILE,
    VALIDATED_QUD_FILE,
    OUTPUT_DIR
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "all-MiniLM-L6-v2"

# These are heuristic thresholds, not factual guarantees.
GROUNDING_THRESHOLD = 0.45
EDU_CONNECTION_THRESHOLD = 0.25
REDUNDANCY_THRESHOLD = 0.90


# ============================================================
# LOAD FILES
# ============================================================

for file_path, label in [
    (EDU_FILE, "EDU"),
    (QUD_CANDIDATE_FILE, "QUD candidate")
]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"{label} file not found:\n{file_path}"
        )

with open(EDU_FILE, "r", encoding="utf-8") as f:
    edus = json.load(f)

with open(QUD_CANDIDATE_FILE, "r", encoding="utf-8") as f:
    candidate_data = json.load(f)

if not edus:
    raise ValueError("No EDUs found.")

if not candidate_data:
    raise ValueError("No QUD candidates found.")

if not isinstance(candidate_data, list):
    raise ValueError(
        "QUD candidate file must contain a list of EDU pairs."
    )


# ============================================================
# LOAD SENTENCE TRANSFORMER
# ============================================================

print("=" * 70)
print("FINAL QUD VALIDATION")
print("=" * 70)

print("\nLoading MiniLM...", flush=True)

model = SentenceTransformer(MODEL_NAME)

print("MiniLM loaded successfully.", flush=True)


# ============================================================
# BASIC QUESTION CHECK
# ============================================================

def is_valid_question(question):
    """Check basic question format and reject obvious meta-questions."""

    if not isinstance(question, str):
        return False

    question = question.strip()

    if not question:
        return False

    # Must end with a question mark.
    if not question.endswith("?"):
        return False

    # Minimum length.
    if len(question.split()) < 5:
        return False

    # Reject references to internal EDU labels.
    if re.search(
        r"\b(?:EDU|TARGET)[-_ ]?(?:A|B|\d+)\b",
        question,
        re.IGNORECASE
    ):
        return False

    bad_patterns = [
        r"^what do you think",
        r"^can you explain",
        r"^according to TARGET",
        r"^what is mentioned in",
        r"^what was mentioned in",
        r"^here are",
        r"^the question is",
        r"^question:"
    ]

    for pattern in bad_patterns:
        if re.search(pattern, question, re.IGNORECASE):
            return False

    return True


# ============================================================
# NUMBER GROUNDING CHECK
# ============================================================

def extract_numbers(text):
    """
    Extract numbers from text.

    This check only checks numeric support. It cannot verify
    all factual claims in a question.
    """

    return re.findall(
        r"\b\d+(?:,\d{3})*(?:\.\d+)?%?\b",
        str(text)
    )


def numbers_are_grounded(question, source_text):
    question_numbers = extract_numbers(question)
    source_numbers = extract_numbers(source_text)

    return all(
        number in source_numbers
        for number in question_numbers
    )


# ============================================================
# EMBEDDING AND SIMILARITY HELPERS
# ============================================================

def get_embeddings(texts):
    """Generate normalized sentence embeddings."""

    return model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True
    )


def similarity_between_embeddings(embedding_a, embedding_b):
    """Return cosine similarity between two embeddings."""

    return float(
        cosine_similarity(
            [embedding_a],
            [embedding_b]
        )[0][0]
    )


def calculate_grounding(question, text1, text2):
    """
    Calculate semantic similarity between a question and
    the combined target EDU pair.
    """

    pair_text = f"{text1} {text2}"

    embeddings = get_embeddings([
        question,
        pair_text
    ])

    return similarity_between_embeddings(
        embeddings[0],
        embeddings[1]
    )


def calculate_edu_connection(question, text1, text2):
    """
    Calculate question similarity to each target EDU separately.

    Important:
    These scores indicate semantic similarity only.
    They do not prove that the question captures the
    discourse relation between the two EDUs.
    """

    embeddings = get_embeddings([
        question,
        text1,
        text2
    ])

    score_a = similarity_between_embeddings(
        embeddings[0],
        embeddings[1]
    )

    score_b = similarity_between_embeddings(
        embeddings[0],
        embeddings[2]
    )

    return score_a, score_b


# ============================================================
# REDUNDANCY REMOVAL
# ============================================================

def remove_redundant_questions(valid_candidates):
    """
    Remove semantic duplicates within one EDU pair.

    Returns:
        unique_candidates
        duplicate_candidates

    Each duplicate is retained in the rejection log.
    """

    if len(valid_candidates) <= 1:
        return valid_candidates, []

    questions = [
        candidate["question"]
        for candidate in valid_candidates
    ]

    embeddings = get_embeddings(questions)

    unique_candidates = []
    duplicate_candidates = []

    for index, candidate in enumerate(valid_candidates):

        duplicate_of = None
        duplicate_similarity = None

        for kept_index, kept_candidate in enumerate(
            unique_candidates
        ):
            similarity = similarity_between_embeddings(
                embeddings[index],
                get_embeddings([
                    kept_candidate["question"]
                ])[0]
            )

            if similarity >= REDUNDANCY_THRESHOLD:
                duplicate_of = kept_candidate["question"]
                duplicate_similarity = similarity
                break

        if duplicate_of is None:
            unique_candidates.append(candidate)

        else:
            duplicate_candidates.append({
                "question": candidate["question"],
                "reason": "semantic_duplicate_within_pair",
                "duplicate_of": duplicate_of,
                "similarity": round(
                    duplicate_similarity,
                    4
                )
            })

    return unique_candidates, duplicate_candidates


# ============================================================
# VALIDATE ONE EDU PAIR
# ============================================================

def validate_pair(item):

    required_fields = [
        "pair_id",
        "edu1",
        "edu2",
        "text1",
        "text2"
    ]

    for field in required_fields:
        if field not in item:
            raise ValueError(
                f"Missing field '{field}' in EDU pair: {item}"
            )

    edu1 = item["edu1"]
    edu2 = item["edu2"]
    text1 = str(item["text1"]).strip()
    text2 = str(item["text2"]).strip()

    candidates = item.get("candidates", [])

    if not isinstance(candidates, list):
        candidates = []

    valid_candidates = []
    rejected = []

    pair_text = f"{text1} {text2}"

    for candidate in candidates:

        # Support the normal string format and a dictionary
        # format in case the upstream output changes.
        if isinstance(candidate, dict):
            question = candidate.get("question", "")
        else:
            question = candidate

        if not isinstance(question, str):
            rejected.append({
                "question": str(question),
                "reason": "invalid_question_type"
            })
            continue

        question = question.strip()

        # ----------------------------------------------------
        # CHECK 1: QUESTION FORMAT
        # ----------------------------------------------------

        if not is_valid_question(question):
            rejected.append({
                "question": question,
                "reason": "invalid_format"
            })
            continue

        # ----------------------------------------------------
        # CHECK 2: NUMBER GROUNDING
        # ----------------------------------------------------

        if not numbers_are_grounded(
            question,
            pair_text
        ):
            rejected.append({
                "question": question,
                "reason": "unsupported_number",
                "source": "target_edu_pair"
            })
            continue

        # ----------------------------------------------------
        # CHECK 3: SEMANTIC GROUNDING
        # ----------------------------------------------------

        grounding_score = calculate_grounding(
            question,
            text1,
            text2
        )

        if grounding_score < GROUNDING_THRESHOLD:
            rejected.append({
                "question": question,
                "reason": "low_semantic_grounding",
                "grounding_score": round(
                    grounding_score,
                    4
                ),
                "threshold": GROUNDING_THRESHOLD
            })
            continue

        # ----------------------------------------------------
        # CHECK 4: CONNECTION TO BOTH EDUs
        # ----------------------------------------------------

        score_a, score_b = calculate_edu_connection(
            question,
            text1,
            text2
        )

        if (
            score_a < EDU_CONNECTION_THRESHOLD
            or score_b < EDU_CONNECTION_THRESHOLD
        ):
            rejected.append({
                "question": question,
                "reason": "weak_edu_connection",
                "edu1_similarity": round(score_a, 4),
                "edu2_similarity": round(score_b, 4),
                "threshold": EDU_CONNECTION_THRESHOLD
            })
            continue

        # ----------------------------------------------------
        # ACCEPT CANDIDATE
        # ----------------------------------------------------

        valid_candidates.append({
            "question": question,
            "grounding_score": round(
                grounding_score,
                4
            ),
            "edu1_similarity": round(
                score_a,
                4
            ),
            "edu2_similarity": round(
                score_b,
                4
            )
        })

    # --------------------------------------------------------
    # CHECK 5: REMOVE REDUNDANT QUESTIONS
    # --------------------------------------------------------

    unique_candidates, duplicate_candidates = (
        remove_redundant_questions(valid_candidates)
    )

    # Record duplicates as rejected so the summary counts
    # account for candidates removed during deduplication.
    rejected.extend(duplicate_candidates)

    # --------------------------------------------------------
    # BUILD RESULT
    # --------------------------------------------------------

    original_count = len(candidates)
    valid_count = len(unique_candidates)
    rejected_count = len(rejected)

    return {
        "pair_id": item["pair_id"],
        "edu1": edu1,
        "edu2": edu2,
        "text1": text1,
        "text2": text2,
        "pair_type": item.get(
            "pair_type",
            "unknown"
        ),
        "semantic_similarity": item.get(
            "semantic_similarity",
            None
        ),
        "original_candidates": original_count,
        "valid_candidates": valid_count,
        "rejected_candidates": rejected_count,
        "candidates": unique_candidates,
        "rejected": rejected
    }


# ============================================================
# SAVE RESULTS SAFELY AFTER EACH PAIR
# ============================================================

def save_results(results):
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    temporary_file = VALIDATED_QUD_FILE + ".tmp"

    with open(
        temporary_file,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False
        )

    # Replace the previous output only after the new JSON
    # has been written successfully.
    os.replace(
        temporary_file,
        VALIDATED_QUD_FILE
    )


# ============================================================
# PROCESS ALL EDU PAIRS
# ============================================================

validated_results = []

total_original = 0
total_valid = 0
total_rejected = 0

for index, item in enumerate(candidate_data, 1):

    print("\n" + "=" * 70)
    print(f"PAIR {index}/{len(candidate_data)}")
    print("=" * 70)

    print(
        f"EDU {item.get('edu1')} "
        f"-> EDU {item.get('edu2')}"
    )

    result = validate_pair(item)

    validated_results.append(result)

    total_original += result["original_candidates"]
    total_valid += result["valid_candidates"]
    total_rejected += result["rejected_candidates"]

    print(f"Original candidates : {result['original_candidates']}")
    print(f"Valid candidates    : {result['valid_candidates']}")
    print(f"Rejected candidates : {result['rejected_candidates']}")

    print("\nVALID QUDs:")

    for candidate in result["candidates"]:
        print(f"- {candidate['question']}")
        print(
            f"  grounding={candidate['grounding_score']} "
            f"| EDU1={candidate['edu1_similarity']} "
            f"| EDU2={candidate['edu2_similarity']}"
        )

    print("\nREJECTION REASONS:")

    if not result["rejected"]:
        print("- None")

    else:
        for rejected_candidate in result["rejected"]:
            print(
                f"- {rejected_candidate['reason']}: "
                f"{rejected_candidate['question']}"
            )

    # Save progress after each pair.
    save_results(validated_results)

    print(f"\nProgress saved: {index}/{len(candidate_data)} pairs")


# ============================================================
# FINAL SUMMARY
# ============================================================

pairs_with_valid = sum(
    1
    for item in validated_results
    if item["valid_candidates"] > 0
)

pairs_without_valid = sum(
    1
    for item in validated_results
    if item["valid_candidates"] == 0
)

print("\n" + "=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)

print(f"\nArticle                : {ARTICLE_NAME}")
print(f"Total EDU pairs        : {len(candidate_data)}")
print(f"Original QUDs          : {total_original}")
print(f"Valid QUDs             : {total_valid}")
print(f"Rejected QUDs          : {total_rejected}")
print(f"Pairs with valid QUDs  : {pairs_with_valid}")
print(f"Pairs without valid    : {pairs_without_valid}")

if total_original > 0:
    retention = total_valid / total_original * 100
    print(f"Retention rate         : {retention:.2f}%")

print("\nRejection breakdown:")

rejection_counts = {}

for pair_result in validated_results:
    for rejected_candidate in pair_result["rejected"]:
        reason = rejected_candidate["reason"]
        rejection_counts[reason] = (
            rejection_counts.get(reason, 0) + 1
        )

if rejection_counts:
    for reason, count in sorted(rejection_counts.items()):
        print(f"- {reason}: {count}")
else:
    print("- No rejected candidates")

print("\nSaved to:")
print(VALIDATED_QUD_FILE)

print("=" * 70)