import os

# Remove the allocator override while diagnosing CUDA memory allocation.
os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import re
import shutil
import tempfile
import gc
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from config import (
    ARTICLE_NAME,
    EDU_FILE,
    PAIR_FILE,
    OUTPUT_DIR,
    QUD_CANDIDATE_FILE,
)


# ============================================================
# STEP 3: QUD CANDIDATE GENERATION
# Qwen2.5-0.5B-Instruct | RTX 2050 4 GB
# ============================================================

MODEL_PATH = Path(
    r"E:\English_QUD\models\Qwen2.5-0.5B-Instruct"
)

CANDIDATES_PER_PAIR = 10
MAX_ATTEMPTS_PER_PAIR = 30

CONTEXT_WINDOW = 1
MAX_INPUT_TOKENS = 768
MAX_NEW_TOKENS = 64

TEMPERATURE = 0.8
TOP_P = 0.9
REPETITION_PENALTY = 1.12


# ============================================================
# JSON UTILITIES
# ============================================================

def load_json(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json_atomically(path, data):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".tmp",
            dir=directory,
            delete=False,
        ) as file:
            temp_path = file.name
            json.dump(data, file, indent=2, ensure_ascii=False)

        os.replace(temp_path, path)

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# ============================================================
# LOAD EDUs AND PAIRS
# ============================================================

print("=" * 68, flush=True)
print("STEP 3: QUD CANDIDATE GENERATION", flush=True)
print("=" * 68, flush=True)

print("\nLoading EDU data...", flush=True)
edus = load_json(EDU_FILE)

print("Loading EDU pairs...", flush=True)
pairs = load_json(PAIR_FILE)

if not isinstance(edus, list) or not edus:
    raise ValueError("EDU_FILE must contain a non-empty JSON list.")

if not isinstance(pairs, list) or not pairs:
    raise ValueError("PAIR_FILE must contain a non-empty JSON list.")

# Support EDU strings and dictionaries containing a "text" field.
normalized_edus = []

for index, edu in enumerate(edus, start=1):
    if isinstance(edu, dict):
        text = edu.get("text", edu.get("edu_text", ""))
    else:
        text = edu

    text = str(text).strip()

    if not text:
        raise ValueError(f"EDU {index} has empty text.")

    normalized_edus.append(text)

edus = normalized_edus
num_edus = len(edus)

for index, pair in enumerate(pairs, start=1):
    if not isinstance(pair, dict):
        raise ValueError(f"Pair {index} is not a JSON object.")

    if "edu1" not in pair or "edu2" not in pair:
        raise ValueError(
            f"Pair {index} must contain edu1 and edu2."
        )

    edu1 = pair["edu1"]
    edu2 = pair["edu2"]

    if (
        not isinstance(edu1, int)
        or isinstance(edu1, bool)
        or not isinstance(edu2, int)
        or isinstance(edu2, bool)
    ):
        raise ValueError(f"Pair {index} has invalid EDU IDs.")

    if not (
        1 <= edu1 <= num_edus
        and 1 <= edu2 <= num_edus
    ):
        raise ValueError(
            f"Invalid EDU IDs in pair {index}: "
            f"{edu1}, {edu2}. Total EDUs: {num_edus}"
        )

    if edu1 == edu2:
        raise ValueError(
            f"Pair {index} references the same EDU twice."
        )

print(f"Article       : {ARTICLE_NAME}", flush=True)
print(f"Total EDUs    : {num_edus}", flush=True)
print(f"Total pairs   : {len(pairs)}", flush=True)
print(
    f"Candidates    : {CANDIDATES_PER_PAIR} per pair",
    flush=True,
)


# ============================================================
# CHECK CUDA
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is unavailable. Check the PyTorch installation."
    )

print(f"\nGPU: {torch.cuda.get_device_name(0)}", flush=True)

gpu_properties = torch.cuda.get_device_properties(0)

print(
    f"Total VRAM: "
    f"{gpu_properties.total_memory / (1024 ** 3):.2f} GiB",
    flush=True,
)

print(
    f"CUDA allocated before loading: "
    f"{torch.cuda.memory_allocated(0) / (1024 ** 3):.3f} GiB",
    flush=True,
)

torch.cuda.empty_cache()


# ============================================================
# VERIFY THE PREVIOUSLY WORKING LOCAL MODEL
# ============================================================

print("\nChecking local Qwen model...", flush=True)
print(f"Model path: {MODEL_PATH}", flush=True)

if not MODEL_PATH.is_dir():
    raise FileNotFoundError(
        f"Model directory not found:\n{MODEL_PATH}"
    )

if not (MODEL_PATH / "config.json").is_file():
    raise FileNotFoundError(
        f"config.json is missing from {MODEL_PATH}"
    )

if not (MODEL_PATH / "model.safetensors").is_file():
    raise FileNotFoundError(
        f"model.safetensors is missing from {MODEL_PATH}"
    )

print("Model configuration and weights found.", flush=True)


# ============================================================
# BACK UP PREVIOUS OUTPUT
# ============================================================

# Preserve the previous result before starting a fresh run.
if os.path.isfile(QUD_CANDIDATE_FILE):
    backup_path = QUD_CANDIDATE_FILE + ".backup_before_step3"

    if not os.path.exists(backup_path):
        shutil.copy2(QUD_CANDIDATE_FILE, backup_path)
        print(
            f"Previous output backed up to:\n{backup_path}",
            flush=True,
        )
    else:
        print(
            f"Existing backup preserved:\n{backup_path}",
            flush=True,
        )


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\nLoading tokenizer...", flush=True)

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True,
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Tokenizer loaded successfully.", flush=True)


# ============================================================
# LOAD MODEL WITH 4-BIT QUANTIZATION (FITS 4GB VRAM)
# ============================================================

print("\nLoading Qwen2.5-0.5B-Instruct in 4-bit...", flush=True)

gc.collect()
torch.cuda.empty_cache()

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True,
    quantization_config=quantization_config,
    device_map="auto",
)

model.eval()

print("\nMODEL LOADED SUCCESSFULLY", flush=True)
print(f"Device map: {model.hf_device_map}", flush=True)

print(
    f"GPU memory allocated: "
    f"{torch.cuda.memory_allocated(0) / (1024 ** 3):.3f} GiB",
    flush=True,
)


# Determine where input embeddings reside.
input_device = model.get_input_embeddings().weight.device

if input_device.type == "meta":
    raise RuntimeError(
        "Input embeddings remain on the meta device. "
        "Model placement did not complete correctly."
    )

print(f"Input embedding device: {input_device}", flush=True)


# ============================================================
# BUILD LOCAL ARTICLE CONTEXT
# ============================================================

def build_context(edu1, edu2):
    first = max(1, min(edu1, edu2) - CONTEXT_WINDOW)
    last = min(num_edus, max(edu1, edu2) + CONTEXT_WINDOW)

    lines = []

    for edu_id in range(first, last + 1):
        if edu_id == edu1:
            marker = " [TARGET-A]"
        elif edu_id == edu2:
            marker = " [TARGET-B]"
        else:
            marker = ""

        lines.append(
            f"EDU {edu_id}{marker}: {edus[edu_id - 1]}"
        )

    return "\n".join(lines)


# ============================================================
# BUILD AN UNBIASED QUD PROMPT
# ============================================================

def build_prompt(edu1, edu2, text1, text2, candidate_number):
    context = build_context(edu1, edu2)

    return f"""
Generate ONE Question Under Discussion (QUD) for an
English news article.

TARGET EDU-A:
{text1}

TARGET EDU-B:
{text2}

LOCAL ARTICLE CONTEXT:
{context}

Candidate number: {candidate_number}

Requirements:
1. Generate one meaningful question grounded in the article.
2. Explore a meaningful connection between the two target EDUs.
3. Use only information supported by the article context.
4. Do not invent facts, names, dates, numbers, causes, or outcomes.
5. Do not ask a yes/no question.
6. Do not mention EDU IDs or these instructions.
7. Do not provide the answer or an explanation.
8. Use natural, concise wording.
9. Choose the question wording that best fits the information.
10. Do not assume a causal relationship unless the article supports it.

Return ONLY one question ending with a question mark.
""".strip()


# ============================================================
# CLEAN AND VALIDATE QUESTIONS
# ============================================================

def clean_question(text):
    if not text:
        return ""

    text = text.strip()

    text = re.sub(
        r"```.*?```",
        " ",
        text,
        flags=re.DOTALL,
    )

    text = re.sub(r"^[\s*\-•#]+", "", text)

    text = re.sub(
        r"^\d+[\.\):\-]\s*",
        "",
        text,
    )

    text = re.sub(
        r"^(question|qud|output)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = text.strip(" \t\n\r\"'“”‘’")
    text = re.sub(r"\s+", " ", text).strip()

    bad_patterns = [
        r"^here are\b",
        r"^the question\b",
        r"^questions?\s*:",
        r"^sure\b",
        r"^candidate\b",
        r"^this question\b",
        r"\bTARGET[- ]?[AB]\b",
        r"\bEDU[- ]?\d+\b",
    ]

    for pattern in bad_patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return ""

    if "?" not in text:
        return ""

    # Retain only the first question.
    text = text.split("?", 1)[0].strip() + "?"

    words = text.split()

    if len(words) < 5 or len(words) > 40:
        return ""

    if not re.match(
        r"^(who|what|when|where|why|how|which|whose|whom)\b",
        text,
        flags=re.IGNORECASE,
    ):
        return ""

    return text


# ============================================================
# NORMALIZE QUESTIONS FOR DEDUPLICATION
# ============================================================

def normalize_question(question):
    return re.sub(
        r"[^a-z0-9]+",
        " ",
        question.lower(),
    ).strip()


# ============================================================
# GENERATE ONE QUESTION
# ============================================================

def generate_one_qud(
    edu1,
    edu2,
    text1,
    text2,
    candidate_number,
):
    prompt = build_prompt(
        edu1,
        edu2,
        text1,
        text2,
        candidate_number,
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful discourse-analysis assistant. "
                "Generate one grounded, natural question only."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    )

    inputs = {
        key: value.to(input_device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][prompt_length:]

    generated_text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return clean_question(generated_text)


# ============================================================
# GENERATE UNIQUE CANDIDATES FOR ONE PAIR
# ============================================================

def generate_candidates(edu1, edu2, text1, text2):
    candidates = []
    seen = set()

    for attempt in range(1, MAX_ATTEMPTS_PER_PAIR + 1):
        if len(candidates) >= CANDIDATES_PER_PAIR:
            break

        question = generate_one_qud(
            edu1,
            edu2,
            text1,
            text2,
            len(candidates) + 1,
        )

        if not question:
            continue

        normalized = normalize_question(question)

        if not normalized or normalized in seen:
            continue

        seen.add(normalized)
        candidates.append(question)

        print(
            f"  QUD {len(candidates)}: {question}",
            flush=True,
        )

    if len(candidates) < CANDIDATES_PER_PAIR:
        print(
            f"  Note: generated {len(candidates)}/"
            f"{CANDIDATES_PER_PAIR} valid unique candidates "
            f"after {MAX_ATTEMPTS_PER_PAIR} attempts.",
            flush=True,
        )

    return candidates


# ============================================================
# MAIN GENERATION LOOP
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

all_results = []
total_candidates = 0

try:
    for pair_index, pair in enumerate(pairs, start=1):
        edu1 = pair["edu1"]
        edu2 = pair["edu2"]

        text1 = edus[edu1 - 1]
        text2 = edus[edu2 - 1]

        print("\n" + "=" * 68, flush=True)
        print(f"PAIR {pair_index}/{len(pairs)}", flush=True)
        print(f"EDU {edu1} -> EDU {edu2}", flush=True)
        print(f"A: {text1}", flush=True)
        print(f"B: {text2}", flush=True)
        print("=" * 68, flush=True)

        candidates = generate_candidates(
            edu1,
            edu2,
            text1,
            text2,
        )

        result = {
            "pair_id": pair_index,
            "edu1": edu1,
            "edu2": edu2,
            "text1": text1,
            "text2": text2,
            "pair_type": pair.get("pair_type", "unknown"),
            "semantic_similarity": pair.get(
                "semantic_similarity",
                None,
            ),
            "num_generated": len(candidates),
            "num_unique": len(candidates),
            "candidates": candidates,
        }

        all_results.append(result)
        total_candidates += len(candidates)

        # Save after each completed pair to preserve progress.
        save_json_atomically(
            QUD_CANDIDATE_FILE,
            all_results,
        )

        print(
            f"\nSaved progress: {len(all_results)}/{len(pairs)} pairs",
            flush=True,
        )

        print(
            f"Candidates for this pair: "
            f"{len(candidates)}/{CANDIDATES_PER_PAIR}",
            flush=True,
        )

        del candidates
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

except KeyboardInterrupt:
    print(
        "\nGeneration interrupted. Saving completed pairs...",
        flush=True,
    )

    save_json_atomically(
        QUD_CANDIDATE_FILE,
        all_results,
    )

    print(
        f"Progress saved to: {QUD_CANDIDATE_FILE}",
        flush=True,
    )

    raise

finally:
    if "model" in globals():
        del model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# FINAL SUMMARY
# ============================================================

pairs_with_10 = sum(
    len(item["candidates"]) == CANDIDATES_PER_PAIR
    for item in all_results
)

print("\n" + "=" * 68, flush=True)
print("STEP 3: QUD GENERATION FINISHED", flush=True)
print("=" * 68, flush=True)

print(f"Article             : {ARTICLE_NAME}", flush=True)
print(f"Total EDUs          : {num_edus}", flush=True)
print(f"Total pairs         : {len(pairs)}", flush=True)
print(f"Pairs processed     : {len(all_results)}", flush=True)
print(f"Total candidates    : {total_candidates}", flush=True)
print(f"Pairs with 10 QUDs  : {pairs_with_10}", flush=True)
print(f"Output file         : {QUD_CANDIDATE_FILE}", flush=True)

print("=" * 68, flush=True)