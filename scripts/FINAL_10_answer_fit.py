import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output" / "article03"

INPUT_FILE = OUTPUT_DIR / "refined_quds.json"
OUTPUT_JSON = OUTPUT_DIR / "answer_fit_scores.json"
OUTPUT_TXT = OUTPUT_DIR / "answer_fit_scores.txt"


def main():
    print("Loading QUD candidates...")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise ValueError("refined_quds.json must contain a list.")

    print("Total candidates:", len(records))

    print("Loading MiniLM model...")

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
        local_files_only=True
    )

    results = []
    groups = {}

    for item in records:
        if not isinstance(item, dict):
            continue

        pair_id = item.get("pair_id")
        qud = item.get("final_qud") or item.get("refined_qud")
        answer = item.get("text2")

        if pair_id is None or not qud or not answer:
            continue

        groups.setdefault(pair_id, []).append({
            "candidate_index": item.get("candidate_index"),
            "edu1": item.get("edu1"),
            "edu2": item.get("edu2"),
            "text1": item.get("text1"),
            "text2": answer,
            "qud": qud,
            "salience_score": float(item.get("salience_score", 0))
        })

    print("Valid EDU pairs:", len(groups))

    for pair_id, candidates in groups.items():
        texts = []

        for candidate in candidates:
            texts.append(candidate["qud"])
            texts.append(candidate["text2"])

        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        pair_results = []

        for i, candidate in enumerate(candidates):
            score = float(
                cosine_similarity(
                    [embeddings[2 * i]],
                    [embeddings[2 * i + 1]]
                )[0][0]
            )

            result = {
                "pair_id": pair_id,
                **candidate,
                "answer_fit_score": round(score, 6),
                "salience_times_answer_fit": round(
                    candidate["salience_score"] * max(0, score), 6
                ),
                "provisionally_selected": False
            }

            pair_results.append(result)

        # Select the highest-salience QUD provisionally.
        selected = max(
            pair_results,
            key=lambda x: x["salience_score"]
        )
        selected["provisionally_selected"] = True

        results.extend(pair_results)

    output = {
        "stage": 10,
        "stage_name": "QUD Answer-Fit Scoring",
        "model": "all-MiniLM-L6-v2",
        "selection_method": "Highest salience, provisional",
        "total_pairs": len(groups),
        "total_candidates": len(results),
        "records": results
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("STAGE 10: QUD ANSWER-FIT SCORING\n")
        f.write("=" * 60 + "\n")

        for result in results:
            f.write(f"\nPair: {result['pair_id']}\n")
            f.write(f"Candidate: {result['candidate_index']}\n")
            f.write(f"QUD: {result['qud']}\n")
            f.write(f"EDU2: {result['text2']}\n")
            f.write(f"Salience: {result['salience_score']:.4f}\n")
            f.write(f"Answer fit: {result['answer_fit_score']:.4f}\n")
            f.write(
                f"Combined score: "
                f"{result['salience_times_answer_fit']:.4f}\n"
            )
            f.write(
                f"Selected provisionally: "
                f"{result['provisionally_selected']}\n"
            )
            f.write("-" * 60 + "\n")

    print("\nSTAGE 10 COMPLETED")
    print("Pairs:", len(groups))
    print("Candidates scored:", len(results))
    print("JSON:", OUTPUT_JSON)
    print("TXT:", OUTPUT_TXT)


if __name__ == "__main__":
    main()