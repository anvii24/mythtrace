"""Text processing shared by documents and queries, so both end up as the same kind of terms.

    preprocess("Type 2 Diabetes and diabetic foot care")              # stemmed
    preprocess("Type 2 Diabetes and diabetic foot care", stem=False)  # unstemmed

Pipeline:
  1. tokenise      - split on anything that is not a letter or digit; numbers are kept
                     ("Type 2" -> "type", "2"), because they matter in health text (type 1 vs type 2)
  2. case folding  - lowercase everything, so "Diabetes" and "diabetes" are the same term
  3. stop words    - drop very common words ("and", "the", "of") from nltk's English list;
                     they appear in almost every chunk, so they carry little meaning (idf near 0)
  4. stemming      - optional Porter stemmer: "diabetes", "diabetic" -> "diabet", so different
                     forms of a word match each other. Toggleable so eval can compare on vs off.
"""
import re
from functools import lru_cache

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

STOP_WORDS = set(stopwords.words("english"))
_stemmer = PorterStemmer()
_TOKEN_RE = re.compile(r"[a-z0-9]+")


@lru_cache(maxsize=None)
def stem_word(word: str) -> str:
    """Porter stem, cached: the same words repeat constantly across the corpus."""
    return _stemmer.stem(word)


def preprocess(text: str, stem: bool = True) -> list[str]:
    """Text -> list of index terms (order kept, duplicates kept, for tf and positions)."""
    tokens = _TOKEN_RE.findall(text.lower())             # 1 + 2: tokenise after case folding
    tokens = [t for t in tokens if t not in STOP_WORDS]  # 3: stop-word removal
    if stem:
        tokens = [stem_word(t) for t in tokens]          # 4: Porter stemming
    return tokens

