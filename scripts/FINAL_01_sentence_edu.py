
# ============================================================
# STAGE 1: SENTENCE-LEVEL EDU SEGMENTATION
# Dynamic article support
# No spaCy / PyTorch dependency
# ============================================================

import json
import re
from pathlib import Path

from config import (
    ARTICLE_NAME,
    ARTICLE_FILE,
    OUTPUT_DIR,
    EDU_FILE,
)


# ============================================================
# DATE DETECTION
# ============================================================

def is_date_line(line: str) -> bool:
    patterns = [
        r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$",
        r"^\d{4}[/-]\d{1,2}[/-]\d{1,2}$",
        (
            r"^(January|February|March|April|May|June|July|"
            r"August|September|October|November|December)"
            r"\s+\d{1,2},?\s+\d{4}$"
        ),
        (
            r"^\d{1,2}\s+"
            r"(January|February|March|April|May|June|July|"
            r"August|September|October|November|December)"
            r"\s+\d{4}$"
        ),
        (
            r"^(Monday|Tuesday|Wednesday|Thursday|Friday|"
            r"Saturday|Sunday),"
        ),
    ]

    return any(
        re.match(pattern, line.strip(), re.IGNORECASE)
        for pattern in patterns
    )


# ============================================================
# SENTENCE SPLITTING
# ============================================================

def split_sentences(text: str) -> list[str]:
    """Split ordinary English article text into sentences."""

    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    # Protect decimal numbers such as 3.14.
    text = re.sub(
        r"(?<=\d)\.(?=\d)",
        "<QUD_DECIMAL>",
        text,
    )

    # Protect abbreviations only when they are separate tokens.
    # The word boundaries prevent matching "Ms." inside "systems."
    abbreviations = [
        "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.",
        "Sr.", "Jr.", "St.", "vs.", "etc.",
        "e.g.", "i.e.", "U.S.", "U.K.",
    ]

    protected = {}

    for index, abbreviation in enumerate(abbreviations):
        placeholder = f"QUDABBR{index}TOKEN"

        escaped = re.escape(abbreviation)

        pattern = (
            r"(?<![A-Za-z])"
            + escaped
            + r"(?=\s|$|[,;:!?])"
        )

        text = re.sub(
            pattern,
            lambda match, p=placeholder: p,
            text,
            flags=re.IGNORECASE,
        )

        # Restore the standard spelling without modifying other words.
        protected[placeholder] = abbreviation

    # Split after sentence-ending punctuation followed by whitespace
    # and the beginning of another sentence.
    parts = re.split(
        r'(?<=[.!?])\s+(?=["“‘(\[]?[A-Z0-9])',
        text,
    )

    sentences = []

    for part in parts:
        sentence = part.strip()

        sentence = sentence.replace("<QUD_DECIMAL>", ".")

        for placeholder, abbreviation in protected.items():
            sentence = sentence.replace(
                placeholder,
                abbreviation,
            )

        if sentence:
            sentences.append(sentence)

    return sentences


# ============================================================
# ARTICLE SEGMENTATION
# ============================================================

def segment_article(text: str) -> list[str]:
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if not lines:
        return []

    edus = []
    body_start = 0

    first_line = re.sub(r"^#+\s*", "", lines[0]).strip()

    # Recognize an explicit Markdown title or a short standalone title.
    has_title = (
        lines[0].startswith("#")
        or (
            len(lines) > 1
            and not re.search(r"[.!?]\s*$", first_line)
            and len(first_line.split()) <= 15
            and not is_date_line(first_line)
        )
    )

    if has_title:
        edus.append(first_line)
        body_start = 1

    # Optional date line after the title.
    if (
        body_start < len(lines)
        and is_date_line(lines[body_start])
    ):
        edus.append(lines[body_start])
        body_start += 1

    body = " ".join(lines[body_start:]).strip()

    if body:
        edus.extend(split_sentences(body))

    return edus


# ============================================================
# VALIDATE INPUT
# ============================================================

input_path = Path(ARTICLE_FILE)
output_path = Path(OUTPUT_DIR)
edu_path = Path(EDU_FILE)

if not input_path.is_file():
    raise FileNotFoundError(
        f"Article not found: {input_path}\n"
        "Check QUD_ARTICLE_PATH in your environment."
    )


# ============================================================
# READ ARTICLE
# ============================================================

article = input_path.read_text(encoding="utf-8").strip()

if not article:
    raise ValueError(f"The article file is empty: {input_path}")


# ============================================================
# SEGMENT
# ============================================================

edus = segment_article(article)

if not edus:
    raise ValueError(
        f"No EDUs generated for article: {ARTICLE_NAME}"
    )


# ============================================================
# SAVE OUTPUTS
# ============================================================

output_path.mkdir(parents=True, exist_ok=True)

with edu_path.open("w", encoding="utf-8") as file:
    json.dump(
        edus,
        file,
        indent=2,
        ensure_ascii=False,
    )

txt_path = output_path / "edus.txt"

with txt_path.open("w", encoding="utf-8") as file:
    for index, edu in enumerate(edus, start=1):
        file.write(f"EDU {index}: {edu}\n")


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("=" * 70)
print("STAGE 1: SENTENCE-LEVEL EDU SEGMENTATION")
print("=" * 70)

print(f"Article    : {ARTICLE_NAME}")
print(f"Input      : {input_path}")
print(f"Total EDUs : {len(edus)}")
print(f"JSON       : {edu_path}")
print(f"Text       : {txt_path}")

print("\nEDUs")
print("-" * 70)

for index, edu in enumerate(edus, start=1):
    print(f"EDU {index}: {edu}")

print("=" * 70)
print("STAGE 1 COMPLETED SUCCESSFULLY")
print("=" * 70)
