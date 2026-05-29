import pandas as pd
import nltk
import re
import os

# ---------------- CONFIG ----------------
INPUT_FILE = "simple_wikipedia_random_domain_dataset.xlsx"
OUTPUT_FILE = "simple_wikipedia_sentence_level_scriptio_continua.xlsx"
# ----------------------------------------

# NLTK setup (only for sentence splitting & stopwords list)
nltk.download("punkt")
nltk.download("stopwords")

from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words("english"))


# ---------- Text Processing ----------

def remove_brackets_and_fix_spaces(text):
    """Remove (), [], {} and normalize spaces"""
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"\[[^\]]*\]", "", text)
    text = re.sub(r"\{[^}]*\}", "", text)
    text = re.sub(r"\s+", " ", text)  # fix double spaces
    return text.strip()


def to_scriptio_continua(text):
    """Lowercase, remove spaces & punctuation, keep letters and numbers"""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]", "", text)
    return text


def compute_statistics(sentence):
    """
    ALL statistics computed from Sentence_without_brackets
    """

    # LETTER-ONLY words
    words = re.findall(r"[A-Za-z]+", sentence)

    stopword_count = sum(1 for w in words if w.lower() in STOPWORDS)
    word_count = len(words)
    space_count = sentence.count(" ")

    # Character counts (exclude spaces only)
    char_count = sum(1 for c in sentence if not c.isspace())
    number_count = sum(1 for c in sentence if c.isdigit())
    letter_count = sum(1 for c in sentence if c.isalpha())
    special_char_count = sum(
        1 for c in sentence if not c.isalnum() and not c.isspace()
    )

    return (
        stopword_count,
        space_count,
        word_count,
        char_count,
        number_count,
        letter_count,
        special_char_count,
    )


# ---------- Main Pipeline ----------

def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_excel(INPUT_FILE)

    rows = []
    sentence_counter = 1

    for _, row in df.iterrows():
        wiki_id = row["id"]
        content = str(row["content"])

        sentences = sent_tokenize(content)

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue

            sentence_no_brackets = remove_brackets_and_fix_spaces(sent)
            scriptio = to_scriptio_continua(sentence_no_brackets)

            (
                stop_cnt,
                space_cnt,
                word_cnt,
                char_cnt,
                num_cnt,
                letter_cnt,
                special_cnt,
            ) = compute_statistics(sentence_no_brackets)

            rows.append({
                "Sentence_Id": f"sent_{sentence_counter:06d}",
                "simple_wiki_id": wiki_id,
                "Sentence in original content": sent,
                "Sentence_without_brackets": sentence_no_brackets,
                "Scriptio_continua": scriptio,
                "stopwords": stop_cnt,
                "space_count": space_cnt,
                "wordcount": word_cnt,
                "charactercount": char_cnt,  # excludes spaces
                "numbers": num_cnt,
                "letters": letter_cnt,
                "special_characters": special_cnt,
            })

            sentence_counter += 1

    final_df = pd.DataFrame(rows)

    output_path = os.path.join(os.getcwd(), OUTPUT_FILE)
    final_df.to_excel(output_path, index=False)

    print("✅ Dataset created successfully")
    print("📁 Output file:", output_path)
    print("📊 Total sentences:", len(final_df))


if __name__ == "__main__":
    main()
