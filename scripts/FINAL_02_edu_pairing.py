
import json
import os
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config import EDU_FILE, PAIR_FILE, OUTPUT_DIR


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "all-MiniLM-L6-v2"

NON_ADJACENT_THRESHOLD = 0.60
TOP_NON_ADJACENT_PER_EDU = 2


# ============================================================
# LOAD EDUs
# ============================================================

if not os.path.isfile(EDU_FILE):
    raise FileNotFoundError(
        f"EDU file not found:\n{EDU_FILE}\n"
        "Run FINAL_01_sentence_edu.py first."
    )

with open(EDU_FILE, "r", encoding="utf-8") as file:
    edus = json.load(file)

if not isinstance(edus, list) or not edus:
    raise ValueError("EDU file must contain a non-empty JSON list.")

if not all(isinstance(edu, str) and edu.strip() for edu in edus):
    raise ValueError("Every EDU must be a non-empty string.")

edus = [edu.strip() for edu in edus]
n = len(edus)

print("=" * 70)
print("STAGE 2: EDU PAIRING")
print("=" * 70)
print(f"Article       : {Path(OUTPUT_DIR).name}")
print(f"Total EDUs    : {n}")
print(f"Similarity threshold: {NON_ADJACENT_THRESHOLD}")
print(f"Top semantic pairs per EDU: {TOP_NON_ADJACENT_PER_EDU}")


# ============================================================
# LOAD SEMANTIC MODEL
# ============================================================

print("\nLoading MiniLM model...")

model = SentenceTransformer(MODEL_NAME, device="cpu")

print("MiniLM loaded successfully.")


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

print("\nCreating EDU embeddings...")

embeddings = model.encode(
    edus,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True,
    batch_size=16
)

print("Embeddings created successfully.")


# ============================================================
# COMPUTE COSINE SIMILARITY
# ============================================================

similarity = cosine_similarity(embeddings)


# ============================================================
# SELECT PAIRS
# ============================================================

selected = {}

# ------------------------------------------------------------
# METHOD 1: ADJACENT PAIRS
# ------------------------------------------------------------

for i in range(n - 1):
    j = i + 1

    key = (i + 1, j + 1)

    selected[key] = {
        "edu1": i + 1,
        "edu2": j + 1,
        "text1": edus[i],
        "text2": edus[j],
        "semantic_similarity": round(
            float(similarity[i, j]), 4
        ),
        "distance": 1,
        "pair_type": "adjacent"
    }


# ------------------------------------------------------------
# METHOD 2: NON-ADJACENT SEMANTIC PAIRS
# ------------------------------------------------------------

for i in range(n):
    candidates = []

    for j in range(n):

        # Skip self-pairs.
        if i == j:
            continue

        # Adjacent pairs already exist.
        if abs(i - j) == 1:
            continue

        score = float(similarity[i, j])

        if score >= NON_ADJACENT_THRESHOLD:
            candidates.append((score, j))

    # Sort by highest similarity, then lower EDU index.
    candidates.sort(key=lambda item: (-item[0], item[1]))

    # Keep at most TOP_NON_ADJACENT_PER_EDU candidates.
    for score, j in candidates[:TOP_NON_ADJACENT_PER_EDU]:

        edu1 = min(i, j) + 1
        edu2 = max(i, j) + 1
        key = (edu1, edu2)

        # Each unordered pair is stored only once.
        if key not in selected:
            selected[key] = {
                "edu1": edu1,
                "edu2": edu2,
                "text1": edus[edu1 - 1],
                "text2": edus[edu2 - 1],
                "semantic_similarity": round(score, 4),
                "distance": edu2 - edu1,
                "pair_type": "global_semantic"
            }


# ============================================================
# VALIDATE AND SORT PAIRS
# ============================================================

pairs = list(selected.values())

for pair in pairs:
    edu1 = pair["edu1"]
    edu2 = pair["edu2"]

    if not (1 <= edu1 < edu2 <= n):
        raise ValueError(f"Invalid EDU pair: {edu1}, {edu2}")

    # Refresh texts from the loaded EDU list.
    pair["text1"] = edus[edu1 - 1]
    pair["text2"] = edus[edu2 - 1]
    pair["distance"] = edu2 - edu1

pairs.sort(key=lambda pair: (pair["edu1"], pair["edu2"]))


# ============================================================
# SAVE OUTPUT
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(PAIR_FILE, "w", encoding="utf-8") as file:
    json.dump(pairs, file, indent=2, ensure_ascii=False)


# ============================================================
# STATISTICS
# ============================================================

adjacent_count = sum(
    pair["pair_type"] == "adjacent"
    for pair in pairs
)

semantic_count = sum(
    pair["pair_type"] == "global_semantic"
    for pair in pairs
)

invalid_pairs = [
    pair for pair in pairs
    if not (
        1 <= pair["edu1"] < pair["edu2"] <= n
        and pair["text1"] == edus[pair["edu1"] - 1]
        and pair["text2"] == edus[pair["edu2"] - 1]
    )
]

if invalid_pairs:
    raise ValueError(
        f"Validation failed: {len(invalid_pairs)} invalid pairs."
    )


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("STAGE 2 COMPLETED SUCCESSFULLY")
print("=" * 70)

print(f"Total EDUs       : {n}")
print(f"Total pairs      : {len(pairs)}")
print(f"Adjacent pairs   : {adjacent_count}")
print(f"Semantic pairs   : {semantic_count}")
print(f"Output file      : {PAIR_FILE}")

print("\nSELECTED PAIRS")
print("-" * 70)

for pair in pairs:
    print(
        f"EDU {pair['edu1']} -> EDU {pair['edu2']} | "
        f"similarity={pair['semantic_similarity']:.4f} | "
        f"distance={pair['distance']} | "
        f"type={pair['pair_type']}"
    )

print("\nVALIDATION")
print("-" * 70)
print("All pairs reference valid EDUs.")
print("No self-pairs or duplicate unordered pairs.")
print("All EDUs are eligible for semantic pairing.")
print("=" * 70)
