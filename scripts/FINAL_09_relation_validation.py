
"""
FINAL_09_relation_validation.py

Stage 9: Audit discourse relations before constructing the discourse tree.

Input:
    output/article03/final_qud_relations.json
    output/article03/refined_quds.json

Outputs:
    output/article03/relation_validation_audit.json
    output/article03/relation_validation_audit.csv
    output/article03/relation_validation_audit.txt

Important:
    - Does not overwrite model predictions.
    - Does not automatically change predicted relation labels.
    - Groups duplicate candidate records into one EDU-pair record.
    - Flags uncertain predictions for manual review.
"""

import json
import csv
from pathlib import Path
from collections import defaultdict


# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
ARTICLE_DIR = BASE_DIR / "output" / "article03"

RELATION_FILE = ARTICLE_DIR / "final_qud_relations.json"
QUD_FILE = ARTICLE_DIR / "refined_quds.json"

OUTPUT_JSON = ARTICLE_DIR / "relation_validation_audit.json"
OUTPUT_CSV = ARTICLE_DIR / "relation_validation_audit.csv"
OUTPUT_TXT = ARTICLE_DIR / "relation_validation_audit.txt"

# Predictions below this confidence are flagged for review.
LOW_CONFIDENCE_THRESHOLD = 0.70

# Confidence alone does not prove whether a relation is correct.
# Every flagged item should be checked against the EDU texts.


# ============================================================
# 2. GENERAL HELPERS
# ============================================================

def load_json(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{path}"
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def normalize_text(value):
    if value is None:
        return ""
    return " ".join(str(value).split())


def get_results(data):
    """
    Supports either:
      {"results": [...]}
    or:
      [...]
    """
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        results = data.get("results", [])
        if isinstance(results, list):
            return results

    raise ValueError(
        "Unexpected relation JSON format. "
        "Expected a list or an object containing 'results'."
    )


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_top_probabilities(probabilities, limit=3):
    """
    Handles probability dictionaries and lists.
    Returns a readable list of the highest probabilities.
    """
    if isinstance(probabilities, dict):
        items = []

        for label, probability in probabilities.items():
            score = to_float(probability)
            if score is not None:
                items.append((str(label), score))

        items.sort(key=lambda item: item[1], reverse=True)

        return [
            {
                "relation": label,
                "probability": round(score, 4)
            }
            for label, score in items[:limit]
        ]

    if isinstance(probabilities, list):
        items = []

        for index, probability in enumerate(probabilities):
            score = to_float(probability)
            if score is not None:
                items.append((str(index), score))

        items.sort(key=lambda item: item[1], reverse=True)

        return [
            {
                "relation": label,
                "probability": round(score, 4)
            }
            for label, score in items[:limit]
        ]

    return []


def get_pair_key(record):
    """
    Group predictions for the same EDU pair.

    Candidate 1 and candidate 2 should not create separate
    discourse-tree edges when they refer to the same EDU pair.
    """
    article_id = record.get("article_id", "article03")
    pair_id = record.get("pair_id")

    edu1_id = record.get("edu1_id", record.get("edu1"))
    edu2_id = record.get("edu2_id", record.get("edu2"))

    if pair_id is not None:
        return (
            str(article_id),
            str(pair_id),
            str(edu1_id),
            str(edu2_id)
        )

    return (
        str(article_id),
        "no_pair_id",
        str(edu1_id),
        str(edu2_id)
    )


# ============================================================
# 3. LOAD INPUT FILES
# ============================================================

print("=" * 70)
print("STAGE 9: DISCOURSE RELATION VALIDATION AUDIT")
print("=" * 70)

relation_data = load_json(RELATION_FILE)
qud_data = load_json(QUD_FILE)

relation_results = get_results(relation_data)
qud_records = get_results(qud_data)

print(f"Relation prediction records: {len(relation_results)}")
print(f"QUD candidate records:       {len(qud_records)}")


# ============================================================
# 4. INDEX QUD CANDIDATES
# ============================================================

qud_by_pair = defaultdict(list)

for record in qud_records:
    key = get_pair_key(record)
    qud_by_pair[key].append(record)


# ============================================================
# 5. GROUP RELATION PREDICTIONS BY EDU PAIR
# ============================================================

grouped_predictions = defaultdict(list)

for record in relation_results:
    key = get_pair_key(record)
    grouped_predictions[key].append(record)


# ============================================================
# 6. BUILD ONE AUDIT RECORD PER EDU PAIR
# ============================================================

audit_records = []

for key, predictions in grouped_predictions.items():

    # Usually both QUD candidates have the same relation prediction.
    # Use the first model record as the representative prediction.
    first = predictions[0]

    pair_id = first.get("pair_id", "UNKNOWN")
    edu1_id = first.get("edu1_id", first.get("edu1", "UNKNOWN"))
    edu2_id = first.get("edu2_id", first.get("edu2", "UNKNOWN"))

    edu1_text = normalize_text(
        first.get("edu1_text")
        or first.get("text1")
        or first.get("edu1")
    )

    edu2_text = normalize_text(
        first.get("edu2_text")
        or first.get("text2")
        or first.get("edu2")
    )

    predicted_relation = (
        first.get("discourse_relation")
        or first.get("relation")
        or "UNKNOWN"
    )

    confidence = to_float(first.get("confidence"))

    relation_probabilities = first.get(
        "relation_probabilities", {}
    )

    top_probabilities = get_top_probabilities(
        relation_probabilities
    )

    # Detect whether duplicated records disagree.
    predicted_labels = sorted({
        str(
            item.get("discourse_relation")
            or item.get("relation")
            or "UNKNOWN"
        )
        for item in predictions
    })

    labels_consistent = len(predicted_labels) == 1

    # Locate the corresponding candidate QUDs.
    matching_quds = qud_by_pair.get(key, [])

    candidate_quds = []

    for qud in matching_quds:
        candidate_quds.append({
            "candidate_index": qud.get("candidate_index"),
            "original_qud": qud.get("original_qud"),
            "refined_qud": qud.get("refined_qud"),
            "final_qud": qud.get("final_qud")
        })

    # Build review flags. These are warnings, not corrections.
    review_reasons = []

    if confidence is None:
        review_reasons.append("MISSING_CONFIDENCE")

    elif confidence < LOW_CONFIDENCE_THRESHOLD:
        review_reasons.append("LOW_CONFIDENCE")

    if not labels_consistent:
        review_reasons.append("DUPLICATE_PREDICTIONS_DISAGREE")

    if not edu1_text or not edu2_text:
        review_reasons.append("MISSING_EDU_TEXT")

    if predicted_relation == "UNKNOWN":
        review_reasons.append("MISSING_RELATION_LABEL")

    if not candidate_quds:
        review_reasons.append("NO_MATCHING_QUD_CANDIDATES")

    audit_record = {
        "article_id": first.get("article_id", "article03"),
        "pair_id": pair_id,
        "edu1_id": edu1_id,
        "edu2_id": edu2_id,

        "edu1_text": edu1_text,
        "edu2_text": edu2_text,

        "predicted_relation": predicted_relation,
        "confidence": confidence,

        "top_probabilities": top_probabilities,
        "all_duplicate_labels": predicted_labels,
        "duplicate_prediction_count": len(predictions),
        "labels_consistent": labels_consistent,

        "candidate_qud_count": len(candidate_quds),
        "candidate_quds": candidate_quds,

        "review_required": len(review_reasons) > 0,
        "review_reasons": review_reasons,

        # Keep this empty until a human validates the relation.
        "validated_relation": None,
        "validation_status": "PENDING_MANUAL_REVIEW"
    }

    audit_records.append(audit_record)


# ============================================================
# 7. SORT BY EDU ORDER / PAIR ID
# ============================================================

def numeric_or_string(value):
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, str(value))


audit_records.sort(
    key=lambda record: (
        numeric_or_string(record.get("edu1_id")),
        numeric_or_string(record.get("edu2_id"))
    )
)


# ============================================================
# 8. CREATE AUDIT SUMMARY
# ============================================================

total_pairs = len(audit_records)

low_confidence_count = sum(
    1 for record in audit_records
    if "LOW_CONFIDENCE" in record["review_reasons"]
)

review_count = sum(
    1 for record in audit_records
    if record["review_required"]
)

inconsistent_count = sum(
    1 for record in audit_records
    if not record["labels_consistent"]
)

relation_counts = defaultdict(int)

for record in audit_records:
    relation_counts[record["predicted_relation"]] += 1

summary = {
    "article_id": "article03",
    "input_relation_records": len(relation_results),
    "unique_edu_pairs": total_pairs,
    "qud_candidate_records": len(qud_records),
    "low_confidence_threshold": LOW_CONFIDENCE_THRESHOLD,
    "low_confidence_pairs": low_confidence_count,
    "pairs_requiring_review": review_count,
    "inconsistent_duplicate_predictions": inconsistent_count,
    "relation_distribution": dict(sorted(relation_counts.items())),
    "note": (
        "Predicted relations are not automatically changed. "
        "Review flagged pairs before constructing the discourse tree."
    )
}


# ============================================================
# 9. SAVE JSON AUDIT
# ============================================================

output_data = {
    "summary": summary,
    "results": audit_records
}

with OUTPUT_JSON.open("w", encoding="utf-8") as file:
    json.dump(
        output_data,
        file,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# 10. SAVE CSV AUDIT
# ============================================================

csv_columns = [
    "article_id",
    "pair_id",
    "edu1_id",
    "edu2_id",
    "edu1_text",
    "edu2_text",
    "predicted_relation",
    "confidence",
    "top_probabilities",
    "duplicate_prediction_count",
    "labels_consistent",
    "candidate_qud_count",
    "review_required",
    "review_reasons",
    "validated_relation",
    "validation_status"
]

with OUTPUT_CSV.open(
    "w",
    newline="",
    encoding="utf-8-sig"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=csv_columns
    )

    writer.writeheader()

    for record in audit_records:
        row = {}

        for column in csv_columns:
            value = record.get(column)

            if column == "top_probabilities":
                value = json.dumps(
                    value,
                    ensure_ascii=False
                )

            elif column == "review_reasons":
                value = "; ".join(value or [])

            row[column] = value

        writer.writerow(row)


# ============================================================
# 11. SAVE HUMAN-READABLE TXT REPORT
# ============================================================

with OUTPUT_TXT.open("w", encoding="utf-8") as file:

    file.write("=" * 80 + "\n")
    file.write("STAGE 9: DISCOURSE RELATION VALIDATION AUDIT\n")
    file.write("=" * 80 + "\n\n")

    file.write(f"Input prediction records: {len(relation_results)}\n")
    file.write(f"Unique EDU pairs:          {total_pairs}\n")
    file.write(f"QUD candidate records:     {len(qud_records)}\n")
    file.write(f"Low-confidence pairs:      {low_confidence_count}\n")
    file.write(f"Pairs requiring review:    {review_count}\n")
    file.write(
        f"Inconsistent duplicates:   {inconsistent_count}\n\n"
    )

    file.write("RELATION DISTRIBUTION\n")
    file.write("-" * 80 + "\n")

    for relation, count in sorted(relation_counts.items()):
        file.write(f"{relation}: {count}\n")

    file.write("\n\nPAIR-BY-PAIR AUDIT\n")
    file.write("=" * 80 + "\n")

    for record in audit_records:

        file.write(
            f"\nPair {record['pair_id']} "
            f"(EDU {record['edu1_id']} -> EDU {record['edu2_id']})\n"
        )

        file.write("-" * 80 + "\n")
        file.write(f"EDU 1: {record['edu1_text']}\n")
        file.write(f"EDU 2: {record['edu2_text']}\n")

        file.write(
            f"Predicted relation: {record['predicted_relation']}\n"
        )

        confidence = record["confidence"]

        if confidence is None:
            file.write("Confidence: MISSING\n")
        else:
            file.write(f"Confidence: {confidence:.4f}\n")

        file.write(
            f"Duplicate records: {record['duplicate_prediction_count']}\n"
        )

        file.write(
            f"Candidate QUD count: {record['candidate_qud_count']}\n"
        )

        file.write(
            f"Review required: {record['review_required']}\n"
        )

        if record["review_reasons"]:
            file.write(
                "Review reasons: "
                + ", ".join(record["review_reasons"])
                + "\n"
            )

        if record["top_probabilities"]:
            file.write("Top probabilities:\n")

            for item in record["top_probabilities"]:
                file.write(
                    f"  {item['relation']}: "
                    f"{item['probability']:.4f}\n"
                )

        if record["candidate_quds"]:
            file.write("Candidate QUDs:\n")

            for candidate in record["candidate_quds"]:
                file.write(
                    f"  Candidate {candidate.get('candidate_index')}: "
                    f"{candidate.get('final_qud') or candidate.get('refined_qud') or candidate.get('original_qud')}\n"
                )

        file.write(
            "Validation status: PENDING_MANUAL_REVIEW\n"
        )

    file.write("\n\n")
    file.write("=" * 80 + "\n")
    file.write("END OF AUDIT\n")
    file.write("=" * 80 + "\n")


# ============================================================
# 12. PRINT RESULTS
# ============================================================

print("\nAUDIT COMPLETED")
print("-" * 70)
print(f"Unique EDU pairs:       {total_pairs}")
print(f"Low-confidence pairs:   {low_confidence_count}")
print(f"Pairs needing review:   {review_count}")
print(f"Inconsistent duplicates:{inconsistent_count}")

print("\nRelation distribution:")

for relation, count in sorted(relation_counts.items()):
    print(f"  {relation}: {count}")

print("\nGenerated files:")
print(f"  JSON: {OUTPUT_JSON}")
print(f"  CSV:  {OUTPUT_CSV}")
print(f"  TXT:  {OUTPUT_TXT}")

print("\nNext step:")
print("Review flagged pairs before constructing the discourse tree.")
print("No original prediction or QUD file was overwritten.")
