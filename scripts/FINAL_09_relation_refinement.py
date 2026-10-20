
import json
import os
import re
from collections import Counter


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = r"E:\English_QUD"
ARTICLE_NAME = "article03"

ARTICLE_DIR = os.path.join(
    BASE_DIR, "output", ARTICLE_NAME
)

QUD_FILE = os.path.join(
    ARTICLE_DIR, "refined_quds.json"
)

STAGE8_FILE = os.path.join(
    ARTICLE_DIR, "final_qud_relations.json"
)

# Do NOT overwrite the Stage 8 output.
OUTPUT_FILE = os.path.join(
    ARTICLE_DIR, "final_qud_relations_refined.json"
)

OUTPUT_TXT = os.path.join(
    ARTICLE_DIR, "final_qud_relations_refined.txt"
)

RELATIONS = [
    "Elaboration",
    "Cause",
    "Purpose",
    "Result",
    "Contrast",
    "Condition",
    "Temporal",
    "Background",
    "Comparison",
    "Concession",
    "Manner",
    "Circumstance"
]

CONFIDENCE_THRESHOLD = 0.75


# ============================================================
# LOAD JSON
# ============================================================

def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_records(data, name):
    """
    Supports either a list of records or an object
    containing a 'results' list.
    """
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        results = data.get("results")

        if isinstance(results, list):
            return results

    raise ValueError(
        f"{name} must be a list or an object containing "
        f"a 'results' list."
    )


# ============================================================
# HELPERS
# ============================================================

def normalize_id(value):
    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value).strip()


def get_qud(record):
    for key in ("final_qud", "refined_qud", "qud", "question"):
        value = record.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""


def get_edu_text(record, text_key, id_key, edus=None):
    """
    Prefer the actual EDU text stored in the QUD record.
    Use edus.json only as a fallback if supplied.
    """
    text = record.get(text_key)

    if isinstance(text, str) and text.strip():
        return text.strip()

    edu_id = normalize_id(record.get(id_key))

    if edus is not None and isinstance(edu_id, int):
        index = edu_id - 1

        if 0 <= index < len(edus):
            value = edus[index]

            if isinstance(value, str):
                return value.strip()

    return ""


# ============================================================
# DISCOURSE CUE DETECTION
# ============================================================

def detect_cues(text):
    text = text.lower()
    cues = []

    patterns = {
        "Contrast": (
            r"\bhowever\b|\bbut\b|\bnevertheless\b|"
            r"\bin contrast\b|\bon the other hand\b"
        ),
        "Concession": (
            r"\balthough\b|\beven though\b|\bdespite\b"
        ),
        "Purpose": (
            r"\bin order to\b|\bso that\b|\bfor the purpose of\b"
        ),
        "Cause": (
            r"\bbecause\b|\bdue to\b|\b owing to\b|"
            r"\bas a result of\b"
        ),
        "Result": (
            r"\bas a result\b|\btherefore\b|\bconsequently\b"
        ),
        "Condition": (
            r"\bif\b|\bunless\b|\bprovided that\b"
        ),
        "Temporal": (
            r"\bafter\b|\bbefore\b|\bwhen\b|\bsubsequently\b|"
            r"\bmeanwhile\b"
        ),
        "Comparison": (
            r"\bcompared with\b|\bin comparison\b|"
            r"\bsimilarly\b|\bwhereas\b"
        )
    }

    for relation, pattern in patterns.items():
        if re.search(pattern, text):
            cues.append(relation)

    return cues


# ============================================================
# STAGE 8 RELATION LOOKUP
# ============================================================

def build_relation_lookup(stage8_records):
    """
    Index Stage 8 predictions by pair_id and candidate_index.
    EDU IDs provide a fallback key.
    """
    lookup = {}

    for result in stage8_records:
        pair_id = normalize_id(result.get("pair_id"))
        candidate_index = normalize_id(
            result.get("candidate_index")
        )

        edu1_id = normalize_id(
            result.get("edu1_id", result.get("edu1"))
        )
        edu2_id = normalize_id(
            result.get("edu2_id", result.get("edu2"))
        )

        relation = result.get("discourse_relation")
        confidence = result.get("confidence")

        if not relation:
            continue

        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            continue

        entry = {
            "pair_id": pair_id,
            "candidate_index": candidate_index,
            "edu1_id": edu1_id,
            "edu2_id": edu2_id,
            "model_relation": relation,
            "model_confidence": confidence,
            "relation_probabilities": result.get(
                "relation_probabilities", {}
            ),
            "prediction_method": result.get(
                "prediction_method", "Stage 8 prediction"
            )
        }

        if pair_id is not None and candidate_index is not None:
            key = (pair_id, candidate_index)
            lookup.setdefault(key, []).append(entry)

        if edu1_id is not None and edu2_id is not None:
            key = (edu1_id, edu2_id, candidate_index)
            lookup.setdefault(key, []).append(entry)

    return lookup


def find_stage8_result(record, lookup):
    """
    First match pair_id + candidate_index.
    Fall back to EDU IDs + candidate_index.
    """
    pair_id = normalize_id(record.get("pair_id"))
    candidate_index = normalize_id(
        record.get("candidate_index")
    )

    edu1_id = normalize_id(
        record.get("edu1", record.get("edu1_id"))
    )
    edu2_id = normalize_id(
        record.get("edu2", record.get("edu2_id"))
    )

    if pair_id is not None and candidate_index is not None:
        matches = lookup.get(
            (pair_id, candidate_index), []
        )

        if matches:
            return matches[0]

    if edu1_id is not None and edu2_id is not None:
        matches = lookup.get(
            (edu1_id, edu2_id, candidate_index), []
        )

        if matches:
            return matches[0]

    return None


# ============================================================
# RELATION REFINEMENT
# ============================================================

def refine_relation(edu1, edu2, model_relation, confidence):
    combined_text = f"{edu1} {edu2}".strip()
    cues = detect_cues(combined_text)

    final_relation = model_relation
    reason = "Stage 8 DeBERTa prediction"

    # Only consider overriding a low-confidence prediction.
    # If several cue types occur, retain the model prediction
    # rather than choosing an arbitrary cue.
    if confidence < CONFIDENCE_THRESHOLD and len(cues) == 1:
        cue_relation = cues[0]

        if cue_relation in RELATIONS:
            final_relation = cue_relation
            reason = (
                f"Low-confidence prediction; "
                f"single explicit cue: {cue_relation}"
            )

    elif confidence < CONFIDENCE_THRESHOLD and len(cues) > 1:
        reason = (
            "Low-confidence prediction; multiple cues detected, "
            "so the model prediction was retained"
        )

    return final_relation, reason, cues


# ============================================================
# MAIN PROCESS
# ============================================================

def main():
    print("=" * 70)
    print("FINAL RELATION REFINEMENT")
    print("=" * 70)

    qud_data = get_records(
        load_json(QUD_FILE), "QUD file"
    )

    stage8_data = get_records(
        load_json(STAGE8_FILE), "Stage 8 relation file"
    )

    print("QUD records:", len(qud_data))
    print("Stage 8 predictions:", len(stage8_data))

    lookup = build_relation_lookup(stage8_data)

    final_results = []
    unmatched_records = []

    print("\nRefining existing Stage 8 predictions...\n")

    for index, record in enumerate(qud_data, start=1):
        if not isinstance(record, dict):
            unmatched_records.append({
                "record_index": index,
                "reason": "QUD record is not a JSON object"
            })
            continue

        stage8_result = find_stage8_result(record, lookup)

        if stage8_result is None:
            unmatched_records.append({
                "record_index": index,
                "pair_id": record.get("pair_id"),
                "candidate_index": record.get("candidate_index"),
                "reason": "No matching Stage 8 relation prediction"
            })
            continue

        edu1_id = normalize_id(
            record.get("edu1", record.get("edu1_id"))
        )
        edu2_id = normalize_id(
            record.get("edu2", record.get("edu2_id"))
        )

        edu1 = get_edu_text(
            record, "text1", "edu1"
        )
        edu2 = get_edu_text(
            record, "text2", "edu2"
        )

        # If the QUD record does not contain EDU text,
        # fall back to the Stage 8 record's text.
        if not edu1:
            value = stage8_result.get("edu1")
            edu1 = value.strip() if isinstance(value, str) else ""

        if not edu2:
            value = stage8_result.get("edu2")
            edu2 = value.strip() if isinstance(value, str) else ""

        qud = get_qud(record)

        model_relation = stage8_result["model_relation"]
        confidence = stage8_result["model_confidence"]

        final_relation, reason, cues = refine_relation(
            edu1,
            edu2,
            model_relation,
            confidence
        )

        result = {
            "article_id": record.get("article_id", ARTICLE_NAME),
            "pair_id": record.get(
                "pair_id", stage8_result.get("pair_id")
            ),
            "candidate_index": record.get(
                "candidate_index",
                stage8_result.get("candidate_index")
            ),
            "edu1_id": edu1_id,
            "edu2_id": edu2_id,
            "edu1": edu1,
            "edu2": edu2,
            "qud": qud,
            "model_relation": model_relation,
            "model_confidence": round(confidence, 4),
            "final_relation": final_relation,
            "refinement_reason": reason,
            "discourse_cues": cues,
            "relation_probabilities": stage8_result.get(
                "relation_probabilities", {}
            ),
            "prediction_method": stage8_result.get(
                "prediction_method", "Stage 8 prediction"
            )
        }

        final_results.append(result)

        print(
            f"PAIR {result['pair_id']} | "
            f"CANDIDATE {result['candidate_index']} | "
            f"{model_relation} -> {final_relation} | "
            f"confidence={confidence:.3f}"
        )

    # Do not silently produce a seemingly complete output
    # when records could not be matched.
    if unmatched_records:
        print("\nWARNING:", len(unmatched_records),
              "QUD records could not be matched.")

        for item in unmatched_records:
            print(item)

    # ========================================================
    # SAVE JSON
    # ========================================================

    output_data = {
        "article_id": ARTICLE_NAME,
        "total_qud_records": len(qud_data),
        "processed_records": len(final_results),
        "unmatched_records": unmatched_records,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "results": final_results
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(
            output_data,
            f,
            indent=2,
            ensure_ascii=False
        )

    # ========================================================
    # SAVE TEXT
    # ========================================================

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        for result in final_results:
            f.write(f"Pair ID: {result['pair_id']}\n")
            f.write(
                f"Candidate Index: {result['candidate_index']}\n"
            )
            f.write(f"EDU 1: {result['edu1']}\n")
            f.write(f"EDU 2: {result['edu2']}\n")
            f.write(f"QUD: {result['qud']}\n")
            f.write(
                f"Model Relation: {result['model_relation']}\n"
            )
            f.write(
                f"Confidence: {result['model_confidence']}\n"
            )
            f.write(
                f"Final Relation: {result['final_relation']}\n"
            )
            f.write(
                f"Refinement Reason: "
                f"{result['refinement_reason']}\n"
            )
            f.write(
                f"Discourse Cues: "
                f"{', '.join(result['discourse_cues']) or 'None'}\n"
            )
            f.write("-" * 60 + "\n")

    # ========================================================
    # SUMMARY
    # ========================================================

    counts = Counter(
        result["final_relation"]
        for result in final_results
    )

    print("\n" + "=" * 70)
    print("FINAL RELATION SUMMARY")
    print("=" * 70)

    for relation in RELATIONS:
        print(f"{relation:15s}: {counts.get(relation, 0)}")

    print("\nTotal QUD records:", len(qud_data))
    print("Successfully processed:", len(final_results))
    print("Unmatched records:", len(unmatched_records))
    print("\nJSON saved:", OUTPUT_FILE)
    print("TXT saved:", OUTPUT_TXT)

    if unmatched_records:
        print(
            "\nWARNING: Check the pair_id and candidate_index "
            "values for unmatched records."
        )
    else:
        print("\nAll QUD records matched successfully.")

    print("\nDONE.")


if __name__ == "__main__":
    main()
