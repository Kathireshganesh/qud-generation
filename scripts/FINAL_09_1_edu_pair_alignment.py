
"""
FINAL_09_1_edu_pair_alignment.py

Stage 9.1: EDU Pair Alignment Validation

Inputs:
    output/article03/edus.json
    output/article03/relation_validation_audit.json

Outputs:
    output/article03/edu_pair_alignment_audit.json
    output/article03/edu_pair_alignment_audit.csv
    output/article03/edu_pair_alignment_audit.txt

This script:
    1. Checks EDU IDs and pair direction.
    2. Checks whether each pair is consecutive.
    3. Checks whether stored EDU text matches canonical EDU text.
    4. Flags low lexical overlap as a weak warning.
    5. Preserves all original relation labels and confidence values.
    6. Never automatically deletes or corrects a relation.
"""

import json
import csv
import re
from pathlib import Path
from collections import Counter


# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
ARTICLE_DIR = BASE_DIR / "output" / "article03"

EDUS_FILE = ARTICLE_DIR / "edus.json"
AUDIT_FILE = ARTICLE_DIR / "relation_validation_audit.json"

OUTPUT_JSON = ARTICLE_DIR / "edu_pair_alignment_audit.json"
OUTPUT_CSV = ARTICLE_DIR / "edu_pair_alignment_audit.csv"
OUTPUT_TXT = ARTICLE_DIR / "edu_pair_alignment_audit.txt"

# Weak lexical overlap is only a review signal, not proof of a bad pair.
MIN_LEXICAL_OVERLAP = 0.10


# ============================================================
# 2. HELPERS
# ============================================================

def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found:\n{path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(value):
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def normalize_for_comparison(value):
    return normalize_text(value).casefold()


def tokenize(text):
    """
    Basic word tokenization.
    This is a diagnostic signal, not a semantic similarity model.
    """
    words = re.findall(r"\b[a-zA-Z0-9]+\b", text.casefold())

    # Ignore very common function words.
    stopwords = {
        "a", "an", "the", "is", "are", "was", "were",
        "be", "been", "being", "to", "of", "in", "on",
        "at", "for", "from", "by", "with", "and", "or",
        "but", "that", "this", "it", "as", "after",
        "before", "if", "so", "will", "would", "could",
        "should", "has", "have", "had", "they", "their",
        "its", "also", "said"
    }

    return {
        word for word in words
        if word not in stopwords and len(word) > 1
    }


def lexical_overlap(text1, text2):
    """
    Jaccard overlap between content-word sets.

    A low score does not mean two EDUs are unrelated.
    It only signals that manual inspection may be useful.
    """
    words1 = tokenize(text1)
    words2 = tokenize(text2)

    if not words1 or not words2:
        return None

    union = words1 | words2

    if not union:
        return None

    return len(words1 & words2) / len(union)


def parse_edu_id(value):
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def extract_edus(data):
    """
    Supports:
      ["EDU text 1", "EDU text 2", ...]

    Also supports:
      {"edus": [...]}

    Dictionary EDU records may contain:
      {"edu_id": 1, "text": "..."}
    or:
      {"id": 1, "text": "..."}
    """
    if isinstance(data, dict):
        data = data.get("edus", data.get("results", []))

    if not isinstance(data, list):
        raise ValueError(
            "Unexpected edus.json format. Expected a list "
            "or an object containing an 'edus' list."
        )

    edu_map = {}

    for index, item in enumerate(data, start=1):

        if isinstance(item, str):
            edu_id = index
            text = item

        elif isinstance(item, dict):
            edu_id = parse_edu_id(
                item.get("edu_id", item.get("id", index))
            )

            text = (
                item.get("text")
                or item.get("edu_text")
                or item.get("sentence")
                or ""
            )

            if edu_id is None:
                edu_id = index

        else:
            continue

        edu_map[edu_id] = normalize_text(text)

    return edu_map


def extract_results(data):
    if isinstance(data, dict):
        results = data.get("results", [])
        if isinstance(results, list):
            return results

    if isinstance(data, list):
        return data

    raise ValueError(
        "Unexpected audit JSON format. Expected a list "
        "or an object containing 'results'."
    )


# ============================================================
# 3. LOAD INPUTS
# ============================================================

print("=" * 72)
print("STAGE 9.1: EDU PAIR ALIGNMENT VALIDATION")
print("=" * 72)

edu_data = load_json(EDUS_FILE)
audit_data = load_json(AUDIT_FILE)

edu_map = extract_edus(edu_data)
audit_records = extract_results(audit_data)

if not edu_map:
    raise ValueError("No EDUs were loaded from edus.json.")

print(f"Canonical EDUs loaded: {len(edu_map)}")
print(f"Relation pairs loaded: {len(audit_records)}")


# ============================================================
# 4. CHECK EACH EDU PAIR
# ============================================================

results = []

for record in audit_records:

    pair_id = record.get("pair_id", "UNKNOWN")

    edu1_id = parse_edu_id(record.get("edu1_id"))
    edu2_id = parse_edu_id(record.get("edu2_id"))

    stored_text1 = normalize_text(record.get("edu1_text"))
    stored_text2 = normalize_text(record.get("edu2_text"))

    issues = []
    warnings = []

    # --------------------------------------------------------
    # A. EDU ID validation
    # --------------------------------------------------------

    if edu1_id is None:
        issues.append("INVALID_EDU1_ID")

    if edu2_id is None:
        issues.append("INVALID_EDU2_ID")

    canonical_text1 = (
        edu_map.get(edu1_id, "")
        if edu1_id is not None else ""
    )

    canonical_text2 = (
        edu_map.get(edu2_id, "")
        if edu2_id is not None else ""
    )

    if edu1_id is not None and edu1_id not in edu_map:
        issues.append("EDU1_ID_NOT_FOUND")

    if edu2_id is not None and edu2_id not in edu_map:
        issues.append("EDU2_ID_NOT_FOUND")

    # --------------------------------------------------------
    # B. Pair direction and adjacency
    # --------------------------------------------------------

    if edu1_id is not None and edu2_id is not None:

        if edu1_id == edu2_id:
            issues.append("SELF_LINK")

        elif edu2_id < edu1_id:
            warnings.append("REVERSED_EDU_ORDER")

        if abs(edu2_id - edu1_id) != 1:
            warnings.append("NON_ADJACENT_EDU_PAIR")

    # --------------------------------------------------------
    # C. Text alignment
    # --------------------------------------------------------

    if canonical_text1 and stored_text1:
        if (
            normalize_for_comparison(canonical_text1)
            != normalize_for_comparison(stored_text1)
        ):
            issues.append("EDU1_TEXT_ID_MISMATCH")

    elif not stored_text1:
        warnings.append("MISSING_STORED_EDU1_TEXT")

    if canonical_text2 and stored_text2:
        if (
            normalize_for_comparison(canonical_text2)
            != normalize_for_comparison(stored_text2)
        ):
            issues.append("EDU2_TEXT_ID_MISMATCH")

    elif not stored_text2:
        warnings.append("MISSING_STORED_EDU2_TEXT")

    # Prefer canonical EDU text whenever the ID is valid.
    text1 = canonical_text1 or stored_text1
    text2 = canonical_text2 or stored_text2

    # --------------------------------------------------------
    # D. Lexical overlap diagnostic
    # --------------------------------------------------------

    overlap = lexical_overlap(text1, text2)

    if overlap is not None and overlap < MIN_LEXICAL_OVERLAP:
        warnings.append("LOW_LEXICAL_OVERLAP")

    if text1 and text2:
        if normalize_for_comparison(text1) == normalize_for_comparison(text2):
            issues.append("IDENTICAL_EDU_TEXT")

    # --------------------------------------------------------
    # E. Keep original model prediction unchanged
    # --------------------------------------------------------

    original_relation = record.get("predicted_relation", "UNKNOWN")
    original_confidence = record.get("confidence")

    review_required = bool(issues or warnings)

    if issues:
        status = "CHECK_ALIGNMENT"
    elif warnings:
        status = "MANUAL_REVIEW"
    else:
        status = "ALIGNMENT_CHECK_PASSED"

    results.append({
        "article_id": record.get("article_id", "article03"),
        "pair_id": pair_id,

        "edu1_id": edu1_id,
        "edu2_id": edu2_id,

        "edu1_text": text1,
        "edu2_text": text2,

        "consecutive_pair": (
            abs(edu2_id - edu1_id) == 1
            if edu1_id is not None and edu2_id is not None
            else None
        ),

        "lexical_overlap": (
            round(overlap, 4) if overlap is not None else None
        ),

        "original_relation": original_relation,
        "original_confidence": original_confidence,

        "alignment_issues": issues,
        "review_warnings": warnings,

        "review_required": review_required,
        "alignment_status": status,

        # Never auto-correct the relation or remove the pair.
        "validated_relation": None,
        "human_review_status": "PENDING"
    })


# ============================================================
# 5. SUMMARY
# ============================================================

status_counts = Counter(
    item["alignment_status"] for item in results
)

issue_counts = Counter()
warning_counts = Counter()

for item in results:
    issue_counts.update(item["alignment_issues"])
    warning_counts.update(item["review_warnings"])

review_count = sum(
    item["review_required"] for item in results
)

summary = {
    "article_id": "article03",
    "canonical_edu_count": len(edu_map),
    "total_pairs": len(results),
    "pairs_requiring_review": review_count,
    "alignment_status_counts": dict(status_counts),
    "issue_counts": dict(issue_counts),
    "warning_counts": dict(warning_counts),
    "minimum_lexical_overlap_warning_threshold": MIN_LEXICAL_OVERLAP,
    "important_note": (
        "Non-adjacency and low lexical overlap are warnings, not proof "
        "of an invalid discourse relation. Original relation labels "
        "and confidence values remain unchanged."
    )
}


# ============================================================
# 6. SAVE JSON
# ============================================================

with OUTPUT_JSON.open("w", encoding="utf-8") as file:
    json.dump(
        {"summary": summary, "results": results},
        file,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# 7. SAVE CSV
# ============================================================

csv_fields = [
    "pair_id",
    "edu1_id",
    "edu2_id",
    "edu1_text",
    "edu2_text",
    "consecutive_pair",
    "lexical_overlap",
    "original_relation",
    "original_confidence",
    "alignment_issues",
    "review_warnings",
    "review_required",
    "alignment_status",
    "human_review_status"
]

with OUTPUT_CSV.open(
    "w",
    newline="",
    encoding="utf-8-sig"
) as file:

    writer = csv.DictWriter(file, fieldnames=csv_fields)
    writer.writeheader()

    for item in results:
        row = dict(item)
        row["alignment_issues"] = "; ".join(item["alignment_issues"])
        row["review_warnings"] = "; ".join(item["review_warnings"])

        writer.writerow({
            field: row.get(field)
            for field in csv_fields
        })


# ============================================================
# 8. SAVE TXT REPORT
# ============================================================

with OUTPUT_TXT.open("w", encoding="utf-8") as file:

    file.write("=" * 78 + "\n")
    file.write("STAGE 9.1: EDU PAIR ALIGNMENT VALIDATION\n")
    file.write("=" * 78 + "\n\n")

    file.write(f"Canonical EDUs: {len(edu_map)}\n")
    file.write(f"Total EDU pairs: {len(results)}\n")
    file.write(f"Pairs requiring review: {review_count}\n\n")

    file.write("WARNING COUNTS\n")
    file.write("-" * 78 + "\n")

    for warning, count in sorted(warning_counts.items()):
        file.write(f"{warning}: {count}\n")

    file.write("\nISSUE COUNTS\n")
    file.write("-" * 78 + "\n")

    for issue, count in sorted(issue_counts.items()):
        file.write(f"{issue}: {count}\n")

    file.write("\n\nPAIR DETAILS\n")
    file.write("=" * 78 + "\n")

    for item in results:

        file.write(
            f"\nPair {item['pair_id']}: "
            f"EDU {item['edu1_id']} -> EDU {item['edu2_id']}\n"
        )

        file.write("-" * 78 + "\n")
        file.write(f"EDU 1: {item['edu1_text']}\n")
        file.write(f"EDU 2: {item['edu2_text']}\n")

        file.write(
            f"Consecutive: {item['consecutive_pair']}\n"
        )

        overlap = item["lexical_overlap"]
        file.write(
            "Lexical overlap: "
            + (f"{overlap:.4f}" if overlap is not None else "N/A")
            + "\n"
        )

        file.write(
            f"Original relation: {item['original_relation']}\n"
        )
        file.write(
            f"Original confidence: {item['original_confidence']}\n"
        )

        file.write(
            f"Alignment status: {item['alignment_status']}\n"
        )

        file.write(
            "Issues: "
            + (
                ", ".join(item["alignment_issues"])
                if item["alignment_issues"] else "None"
            )
            + "\n"
        )

        file.write(
            "Warnings: "
            + (
                ", ".join(item["review_warnings"])
                if item["review_warnings"] else "None"
            )
            + "\n"
        )

        file.write("Human review status: PENDING\n")

    file.write("\n" + "=" * 78 + "\n")
    file.write("END OF REPORT\n")


# ============================================================
# 9. PRINT SUMMARY
# ============================================================

print("\nALIGNMENT AUDIT COMPLETED")
print("-" * 72)
print(f"Canonical EDUs:          {len(edu_map)}")
print(f"Pairs checked:           {len(results)}")
print(f"Pairs requiring review:  {review_count}")

print("\nAlignment status:")
for status, count in sorted(status_counts.items()):
    print(f"  {status}: {count}")

print("\nWarnings:")
for warning, count in sorted(warning_counts.items()):
    print(f"  {warning}: {count}")

print("\nIssues:")
for issue, count in sorted(issue_counts.items()):
    print(f"  {issue}: {count}")

print("\nOutput files:")
print(f"  {OUTPUT_JSON}")
print(f"  {OUTPUT_CSV}")
print(f"  {OUTPUT_TXT}")

print("\nOriginal predictions were not changed.")
print("Review the report before constructing the discourse tree.")
