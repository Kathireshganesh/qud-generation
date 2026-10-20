
import os
import json
import torch
from collections import Counter
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from config import ARTICLE_NAME, REFINED_QUD_FILE, OUTPUT_DIR


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"E:\English_QUD"
MODEL_DIR = os.path.join(
    BASE_DIR, "output", "relation_training",
    "FINAL_deberta_relation_model"
)

EDU_FILE = os.path.join(OUTPUT_DIR, "edus.json")
QUD_FILE = REFINED_QUD_FILE

OUTPUT_JSON = os.path.join(
    OUTPUT_DIR, "final_qud_relations.json"
)
OUTPUT_TXT = os.path.join(
    OUTPUT_DIR, "final_qud_relations.txt"
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_qud(record):
    """Return the best available QUD text."""
    for key in ("final_qud", "refined_qud", "selected_qud",
                "qud", "original_qud"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def main():
    print("=" * 70)
    print("STAGE 8: DISCOURSE RELATION PREDICTION")
    print("=" * 70)
    print("Article:", ARTICLE_NAME)
    print("Model:", MODEL_DIR)
    print("Device:", "cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.isdir(MODEL_DIR):
        raise FileNotFoundError(
            f"Trained relation model not found: {MODEL_DIR}"
        )
    if not os.path.isfile(EDU_FILE):
        raise FileNotFoundError(f"EDU file not found: {EDU_FILE}")
    if not os.path.isfile(QUD_FILE):
        raise FileNotFoundError(f"QUD file not found: {QUD_FILE}")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load model
    print("\nLoading DeBERTa...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR
    ).to(device)
    model.eval()

    id2label = {
        int(k): v for k, v in model.config.id2label.items()
    }

    print("Model loaded.")
    print("Relation labels:", id2label)

    # Load EDUs. Their IDs are their 1-based positions in this list.
    edus = load_json(EDU_FILE)
    if not isinstance(edus, list):
        raise ValueError("edus.json must contain a list of EDU texts.")

    edu_map = {
        i: str(text).strip()
        for i, text in enumerate(edus, start=1)
    }

    # Load refined QUD candidate records
    qud_records = load_json(QUD_FILE)
    if not isinstance(qud_records, list):
        raise ValueError("refined_quds.json must contain a list.")

    print(f"\nEDUs loaded: {len(edu_map)}")
    print(f"QUD candidate records: {len(qud_records)}")

    results = []
    skipped = []

    print("\nRunning relation predictions...")

    for index, record in enumerate(qud_records, start=1):
        if not isinstance(record, dict):
            skipped.append({
                "record_index": index,
                "reason": "Record is not a JSON object",
            })
            continue

        # Actual schema in refined_quds.json
        edu1_id = record.get("edu1")
        edu2_id = record.get("edu2")
        qud = get_qud(record)

        try:
            edu1_id = int(edu1_id)
            edu2_id = int(edu2_id)
        except (TypeError, ValueError):
            skipped.append({
                "record_index": index,
                "reason": "Invalid edu1/edu2 ID",
            })
            continue

        if edu1_id not in edu_map or edu2_id not in edu_map:
            skipped.append({
                "record_index": index,
                "reason": "EDU ID not found in edus.json",
            })
            continue

        if not qud:
            skipped.append({
                "record_index": index,
                "reason": "No usable QUD text",
            })
            continue

        edu1 = edu_map[edu1_id]
        edu2 = edu_map[edu2_id]

        # IMPORTANT:
        # This trained classifier receives EDU1 + EDU2 only.
        # QUD text is saved as candidate metadata, not model input.
        encoded = tokenizer(
            edu1,
            edu2,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        with torch.inference_mode():
            output = model(**encoded)
            probs = torch.softmax(output.logits, dim=-1)[0]

        predicted_id = int(torch.argmax(probs).item())
        relation = id2label.get(
            predicted_id, str(predicted_id)
        )
        confidence = float(probs[predicted_id].item())

        relation_probabilities = {
            id2label.get(i, str(i)): round(float(p), 6)
            for i, p in enumerate(probs.cpu().tolist())
        }

        result = {
            "record_index": index,
            "article_id": record.get("article_id", ARTICLE_NAME),
            "pair_id": record.get("pair_id"),
            "candidate_index": record.get("candidate_index"),
            "edu1_id": edu1_id,
            "edu2_id": edu2_id,
            "edu1": edu1,
            "edu2": edu2,
            "qud": qud,
            "pair_type": record.get("pair_type"),
            "salience_score": record.get("salience_score"),
            "selection_rank": record.get("selection_rank"),
            "meaning_similarity": record.get("meaning_similarity"),
            "discourse_relation": relation,
            "confidence": round(confidence, 6),
            "relation_probabilities": relation_probabilities,
            "prediction_method": "edu_pair_only",
        }
        results.append(result)

        print(
            f"[{index}/{len(qud_records)}] "
            f"Pair={result['pair_id']} "
            f"Candidate={result['candidate_index']} "
            f"Relation={relation} "
            f"Confidence={confidence:.4f}"
        )

    if not results:
        raise ValueError(
            "No valid predictions generated. Check the EDU and QUD IDs."
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_data = {
        "article_id": ARTICLE_NAME,
        "prediction_method": "edu_pair_only",
        "note": (
            "The trained classifier receives the two EDUs only. "
            "QUD candidates and salience values are preserved as metadata; "
            "predictions are not QUD-conditioned."
        ),
        "total_candidates": len(qud_records),
        "successful_predictions": len(results),
        "skipped_records": skipped,
        "results": results,
    }
    save_json(OUTPUT_JSON, output_data)

    # Human-readable output
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("ENGLISH QUD DISCOURSE RELATION PREDICTIONS\n")
        f.write("=" * 70 + "\n")
        f.write(f"Article: {ARTICLE_NAME}\n")
        f.write(f"Candidates: {len(results)}\n")
        f.write("Prediction input: EDU pair only\n\n")

        for r in results:
            f.write("-" * 70 + "\n")
            f.write(
                f"Pair ID: {r['pair_id']} | "
                f"Candidate: {r['candidate_index']}\n"
            )
            f.write(f"EDU {r['edu1_id']}: {r['edu1']}\n")
            f.write(f"EDU {r['edu2_id']}: {r['edu2']}\n")
            f.write(f"QUD: {r['qud']}\n")
            f.write(
                f"Salience: {r['salience_score']}\n"
                f"Relation: {r['discourse_relation']}\n"
                f"Confidence: {r['confidence']:.6f}\n"
            )
            f.write("Relation probabilities:\n")
            for label, probability in r[
                "relation_probabilities"
            ].items():
                f.write(f"  {label}: {probability:.6f}\n")
            f.write("\n")

        f.write("=" * 70 + "\n")
        f.write(f"Skipped records: {len(skipped)}\n")

    counts = Counter(
        r["discourse_relation"] for r in results
    )

    unique_pairs = {
        r["pair_id"] for r in results
        if r["pair_id"] is not None
    }

    print("\n" + "=" * 70)
    print("STAGE 8 COMPLETE")
    print("=" * 70)
    print("Successful predictions:", len(results))
    print("Unique EDU pairs:", len(unique_pairs))
    print("Skipped records:", len(skipped))
    print("\nRelation distribution:")
    for label, count in sorted(counts.items()):
        print(f"  {label}: {count}")

    print("\nJSON:", OUTPUT_JSON)
    print("TXT :", OUTPUT_TXT)


if __name__ == "__main__":
    main()
