import math
import textwrap
from deep_translator import GoogleTranslator


RAW_FILEPATH = "./data/raw"
PROCESSED_FILEPATH = "./data/processed"


def save_in_chunks(df, name_function, chunk_size):
    total_blocos = math.ceil(len(df) / chunk_size)
    for i in range(total_blocos):
        bloco = df.iloc[i * chunk_size : (i + 1) * chunk_size]
        filepath = name_function(i + 1)
        bloco.to_parquet(filepath, index=False)


def translate(text, limit=4000):
    parts = textwrap.wrap(
        text, width=limit, break_long_words=False, break_on_hyphens=False
    )
    translated = ""
    for part in parts:
        translated += GoogleTranslator(source="auto", target="pt").translate(part) + " "
    return translated.strip()
