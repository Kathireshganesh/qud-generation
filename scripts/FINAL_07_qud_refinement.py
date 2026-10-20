
import os

# Set before importing Transformers to avoid tokenizer parallelism warnings.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import re
import tempfile
import traceback
from collections import Counter

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    ARTICLE_NAME,
    SELECTED_QUD_FILE,
    REFINED_QUD_FILE,
)


# ============================================================
# STAGE 7 — QUD REFINEMENT
# Qwen2.5-0.5B-Instruct + MiniLM validation
# ============================================================

MODEL_PATH = r"E:\English_QUD\models\Qwen2.5-0.5B-Instruct"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

MAX_INPUT_TOKENS = 2048
MAX_NEW_TOKENS = 80

MIN_MEANING_SIMILARITY = 0.70
MIN_EDU_SIMILARITY = 0.25

# Conservative generation settings.
# Sampling-only parameters are disabled to avoid the known warning.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


# ============================================================
# FILE HELPERS
# ============================================================

def read_json(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json_atomic(path, data):
    """Write a complete JSON file before replacing the previous output."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".tmp",
            prefix="stage7_",
            dir=directory,
            delete=False,
        ) as temp_file:
            json.dump(
                data,
                temp_file,
                ensure_ascii=False,
                indent=2,
            )
            temp_file.flush()
            os.fsync(temp_file.fileno())
            temp_path = temp_file.name

        os.replace(temp_path, path)

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


# ============================================================
# TEXT HELPERS
# ============================================================

def clean_question(text):
    if not isinstance(text, str):
        return ""

    text = text.strip()

    text = re.sub(
        r"^```(?:text)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s*```$", "", text)

    text = re.sub(
        r"^(?:refined question|question|output)\s*:\s*",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    )

    text = text.strip().strip("\"'").strip()
    text = re.sub(r"\s+", " ", text)

    # Keep only the first complete question.
    match = re.search(r"(.+?\?)", text)
    if match:
        text = match.group(1).strip()

    return text


def extract_numbers(text):
    return re.findall(
        r"\b\d+(?:[.,]\d+)?%?\b",
        text or "",
    )


def is_question(text):
    if not isinstance(text, str) or not text.strip():
        return False

    words = text.split()

    return (
        4 <= len(words) <= 45
        and text.endswith("?")
        and "\n" not in text
    )


def normalize_text(text):
    return re.sub(
        r"\s+",
        " ",
        str(text or "").strip().lower(),
    )


# ============================================================
# INPUT PREPARATION
# ============================================================

def prepare_tasks(data):
    """
    Expected Stage 6 structure:
    {
        "article_id": "...",
        "pairs": [
            {
                "pair_id": ...,
                "edu1": ...,
                "edu2": ...,
                "text1": "...",
                "text2": "...",
                "selected_candidates": [...]
            }
        ]
    }
    """

    if not isinstance(data, dict):
        raise ValueError(
            "Invalid Stage 6 input: expected a JSON object."
        )

    pairs = data.get("pairs")

    if not isinstance(pairs, list):
        raise ValueError(
            "Stage 6 input must contain a 'pairs' list."
        )

    tasks = []
    article_id = data.get("article_id", ARTICLE_NAME)

    for pair in pairs:
        if not isinstance(pair, dict):
            continue

        pair_id = pair.get("pair_id", "unknown_pair")

        edu1 = pair.get("edu1", "")
        edu2 = pair.get("edu2", "")

        # Prefer actual EDU text. edu1/edu2 may be numeric IDs.
        text1 = pair.get("text1")
        text2 = pair.get("text2")

        if not isinstance(text1, str):
            text1 = edu1 if isinstance(edu1, str) else ""

        if not isinstance(text2, str):
            text2 = edu2 if isinstance(edu2, str) else ""

        candidates = pair.get("selected_candidates", [])

        if not isinstance(candidates, list):
            continue

        for candidate_index, candidate in enumerate(
            candidates, start=1
        ):
            if not isinstance(candidate, dict):
                continue

            original = (
                candidate.get("question")
                or candidate.get("selected_qud")
                or candidate.get("qud")
                or ""
            )

            tasks.append({
                "article_id": article_id,
                "pair_id": pair_id,
                "candidate_index": candidate_index,
                "edu1": edu1,
                "edu2": edu2,
                "text1": text1,
                "text2": text2,
                "pair_type": pair.get("pair_type", ""),
                "original_qud": (
                    original.strip()
                    if isinstance(original, str)
                    else ""
                ),
                "salience_score": candidate.get(
                    "salience_score",
                    candidate.get("salience"),
                ),
                "selection_rank": candidate.get(
                    "selection_rank",
                    candidate_index,
                ),
                "refined_qud": None,
                "final_qud": None,
                "meaning_similarity": None,
                "refined_edu1_similarity": None,
                "refined_edu2_similarity": None,
                "refinement_status": "pending",
                "validation_reason": "",
            })

    return tasks


# ============================================================
# MODEL LOADING
# ============================================================

def load_models():
    if not os.path.isdir(MODEL_PATH):
        raise FileNotFoundError(
            f"Qwen model folder not found:\n{MODEL_PATH}"
        )

    required_files = [
        "config.json",
        "tokenizer_config.json",
    ]

    for filename in required_files:
        file_path = os.path.join(MODEL_PATH, filename)

        if not os.path.isfile(file_path):
            raise FileNotFoundError(
                f"Required model file not found: {file_path}"
            )

    has_weights = any(
        name.endswith((".safetensors", ".bin"))
        and os.path.isfile(os.path.join(MODEL_PATH, name))
        for name in os.listdir(MODEL_PATH)
    )

    if not has_weights:
        raise FileNotFoundError(
            f"No model weight files found in: {MODEL_PATH}"
        )

    print("\nLoading Qwen tokenizer...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    print("Loading Qwen2.5-0.5B...", flush=True)

    dtype = (
        torch.float16
        if DEVICE.startswith("cuda")
        else torch.float32
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    model.to(DEVICE)
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build a clean deterministic generation configuration.
    # Setting sampling-only fields to None avoids the warning
    # about temperature/top_p/top_k with do_sample=False.
    generation_config = GenerationConfig.from_model_config(
        model.config
    )
    generation_config.do_sample = False
    generation_config.num_beams = 1
    generation_config.max_new_tokens = MAX_NEW_TOKENS
    generation_config.temperature = None
    generation_config.top_p = None
    generation_config.top_k = None
    generation_config.pad_token_id = tokenizer.pad_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id

    print("Qwen model loaded.", flush=True)
    print("Inference device:", DEVICE, flush=True)

    if DEVICE.startswith("cuda"):
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
            flush=True,
        )
        print(
            "Allocated VRAM:",
            round(
                torch.cuda.memory_allocated(0) / (1024 ** 3),
                2,
            ),
            "GB",
            flush=True,
        )

    print("Loading MiniLM embedding model...", flush=True)

    embedder = SentenceTransformer(
        EMBEDDING_MODEL,
        device="cpu",
    )

    print("MiniLM loaded.", flush=True)

    return tokenizer, model, embedder, generation_config


# ============================================================
# QUD REFINEMENT
# ============================================================

def refine_qud(
    original,
    edu1,
    edu2,
    tokenizer,
    model,
    generation_config,
):
    prompt = f"""You are an expert editor refining a Question Under Discussion (QUD).

TASK:
Improve the original question only if needed.

RULES:
1. Preserve the original question's meaning and intent.
2. Use natural, grammatically correct English.
3. Keep the question grounded in the supplied EDU text.
4. Do not invent facts, entities, events, dates, or numbers.
5. Do not reverse the meaning of either EDU.
6. Do not answer the question.
7. Do not add explanations or alternatives.
8. Return exactly one complete question ending with ?.
9. If the original question is already good, return it unchanged.

EDU 1:
{edu1}

EDU 2:
{edu2}

ORIGINAL QUESTION:
{original}

REFINED QUESTION:"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    )

    input_device = model.get_input_embeddings().weight.device

    inputs = {
        key: value.to(input_device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            generation_config=generation_config,
            repetition_penalty=1.05,
        )

    generated_tokens = output[0][
        inputs["input_ids"].shape[1]:
    ]

    generated_text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return clean_question(generated_text)


# ============================================================
# SEMANTIC VALIDATION
# ============================================================

def similarity(embedder, text_a, text_b):
    if not text_a or not text_b:
        return 0.0

    vectors = embedder.encode(
        [text_a, text_b],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    score = cosine_similarity(
        vectors[0].reshape(1, -1),
        vectors[1].reshape(1, -1),
    )[0][0]

    return float(score)


def validate_refinement(
    original,
    refined,
    edu1,
    edu2,
    embedder,
):
    if not refined:
        return False, "Empty generated question", None, None, None

    if not is_question(refined):
        return (
            False,
            "Invalid question format or length",
            None,
            None,
            None,
        )

    # Do not allow numerical facts to be changed or introduced.
    if Counter(extract_numbers(original)) != Counter(
        extract_numbers(refined)
    ):
        return (
            False,
            "Numbers changed during refinement",
            None,
            None,
            None,
        )

    meaning_score = similarity(
        embedder, original, refined
    )

    edu1_score = (
        similarity(embedder, refined, edu1)
        if edu1.strip()
        else None
    )

    edu2_score = (
        similarity(embedder, refined, edu2)
        if edu2.strip()
        else None
    )

    if meaning_score < MIN_MEANING_SIMILARITY:
        return (
            False,
            f"Meaning similarity too low: {meaning_score:.3f}",
            meaning_score,
            edu1_score,
            edu2_score,
        )

    if (
        edu1_score is not None
        and edu1_score < MIN_EDU_SIMILARITY
    ):
        return (
            False,
            f"EDU 1 similarity too low: {edu1_score:.3f}",
            meaning_score,
            edu1_score,
            edu2_score,
        )

    if (
        edu2_score is not None
        and edu2_score < MIN_EDU_SIMILARITY
    ):
        return (
            False,
            f"EDU 2 similarity too low: {edu2_score:.3f}",
            meaning_score,
            edu1_score,
            edu2_score,
        )

    return (
        True,
        "Passed heuristic validation",
        meaning_score,
        edu1_score,
        edu2_score,
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    print("=" * 65)
    print("STAGE 7 — QUD REFINEMENT")
    print("=" * 65)
    print("Article:", ARTICLE_NAME)
    print("Device:", DEVICE)
    print("Input:", SELECTED_QUD_FILE)
    print("Output:", REFINED_QUD_FILE)

    input_data = read_json(SELECTED_QUD_FILE)
    tasks = prepare_tasks(input_data)

    print("EDU pairs:", len(input_data.get("pairs", [])))
    print("Selected QUD candidates:", len(tasks))

    if not tasks:
        raise ValueError(
            "No selected QUD candidates found in Stage 6 output."
        )

    # Load earlier progress for resume support.
    existing_results = []

    if os.path.isfile(REFINED_QUD_FILE):
        try:
            loaded = read_json(REFINED_QUD_FILE)
            if isinstance(loaded, list):
                existing_results = loaded
        except (json.JSONDecodeError, OSError):
            print(
                "Existing output is unreadable; starting fresh.",
                flush=True,
            )

    existing_by_key = {}

    for record in existing_results:
        if not isinstance(record, dict):
            continue

        key = (
            str(record.get("pair_id")),
            record.get("candidate_index"),
        )
        existing_by_key[key] = record

    completed_statuses = {
        "accepted",
        "fallback_original",
        "missing_original",
    }

    for task in tasks:
        key = (
            str(task["pair_id"]),
            task["candidate_index"],
        )
        old = existing_by_key.get(key)

        # Resume only when the original candidate is unchanged.
        if (
            old
            and old.get("refinement_status") in completed_statuses
            and normalize_text(old.get("original_qud"))
            == normalize_text(task.get("original_qud"))
        ):
            task.update(old)

    pending = [
        task for task in tasks
        if task.get("refinement_status") == "pending"
    ]

    if pending:
        tokenizer, model, embedder, generation_config = load_models()
    else:
        print("All candidates already processed.", flush=True)
        tokenizer = model = embedder = generation_config = None

    newly_processed = 0
    errors = 0

    for index, task in enumerate(tasks, start=1):
        if task.get("refinement_status") != "pending":
            continue

        original = str(task.get("original_qud") or "").strip()
        edu1 = str(task.get("text1") or "").strip()
        edu2 = str(task.get("text2") or "").strip()

        print(
            f"\n[{index}/{len(tasks)}] "
            f"Pair {task['pair_id']} | "
            f"Candidate {task['candidate_index']}",
            flush=True,
        )

        if not original:
            task["final_qud"] = None
            task["refined_qud"] = None
            task["refinement_status"] = "missing_original"
            task["validation_reason"] = "Original question is missing"

            save_json_atomic(REFINED_QUD_FILE, tasks)
            newly_processed += 1
            continue

        try:
            refined = refine_qud(
                original,
                edu1,
                edu2,
                tokenizer,
                model,
                generation_config,
            )

            task["refined_qud"] = refined

            # An unchanged question is safe to retain, but still check format.
            if normalize_text(refined) == normalize_text(original):
                valid = is_question(original)
                reason = (
                    "Original retained; model made no change"
                    if valid
                    else "Original retained; original format is invalid"
                )
                meaning_score = 1.0 if valid else None
                edu1_score = (
                    similarity(embedder, original, edu1)
                    if edu1 else None
                )
                edu2_score = (
                    similarity(embedder, original, edu2)
                    if edu2 else None
                )
            else:
                (
                    valid,
                    reason,
                    meaning_score,
                    edu1_score,
                    edu2_score,
                ) = validate_refinement(
                    original,
                    refined,
                    edu1,
                    edu2,
                    embedder,
                )

            task["meaning_similarity"] = meaning_score
            task["refined_edu1_similarity"] = edu1_score
            task["refined_edu2_similarity"] = edu2_score
            task["validation_reason"] = reason

            if valid:
                task["final_qud"] = refined
                task["refinement_status"] = "accepted"
                print("Accepted:", refined, flush=True)
            else:
                task["final_qud"] = original
                task["refinement_status"] = "fallback_original"
                print("Original retained:", reason, flush=True)

        except Exception as exc:
            task["final_qud"] = original
            task["refinement_status"] = "fallback_original"
            task["validation_reason"] = (
                f"{type(exc).__name__}: {exc}"
            )
            errors += 1

            print(
                f"Candidate failed; original retained: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            traceback.print_exc()

        newly_processed += 1

        # Save after each candidate so progress is not lost.
        save_json_atomic(REFINED_QUD_FILE, tasks)

    save_json_atomic(REFINED_QUD_FILE, tasks)

    accepted = sum(
        task.get("refinement_status") == "accepted"
        for task in tasks
    )
    fallback = sum(
        task.get("refinement_status") == "fallback_original"
        for task in tasks
    )
    missing = sum(
        task.get("refinement_status") == "missing_original"
        for task in tasks
    )

    print("\n" + "=" * 65)
    print("STAGE 7 SUMMARY")
    print("=" * 65)
    print("Total candidates:", len(tasks))
    print("Newly processed:", newly_processed)
    print("Accepted refinements:", accepted)
    print("Originals retained:", fallback)
    print("Missing originals:", missing)
    print("Generation errors:", errors)
    print("Output:", REFINED_QUD_FILE)
    print("=" * 65)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nSTAGE 7 FAILED", flush=True)
        print(f"{type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        raise
