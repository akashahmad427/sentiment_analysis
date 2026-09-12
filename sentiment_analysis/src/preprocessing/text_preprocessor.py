"""
Text Preprocessing Module
--------------------------
Cleans and normalizes raw text for both ML and DL pipelines.
Handles: emojis, URLs, mentions, special characters, HTML entities.
"""

import re
import logging
import unicodedata
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Modular text preprocessor supporting both ML (heavy cleaning) 
    and DL (light cleaning) pipelines.
    """

    # Regex patterns
    URL_PATTERN = re.compile(r"http\S+|www\.\S+")
    MENTION_PATTERN = re.compile(r"@\w+")
    HASHTAG_PATTERN = re.compile(r"#(\w+)")
    HTML_PATTERN = re.compile(r"<[^>]+>")
    SPECIAL_CHARS_PATTERN = re.compile(r"[^a-zA-Z0-9\s.,!?'\"()-]")
    WHITESPACE_PATTERN = re.compile(r"\s+")
    EMOJI_PATTERN = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FA6F"
        "\U0001FA70-\U0001FAFF"
        "\u2702-\u27B0"
        "]+",
        flags=re.UNICODE,
    )

    def __init__(self, mode: str = "ml"):
        """
        Args:
            mode: 'ml' for heavy cleaning (TF-IDF), 'dl' for light cleaning (transformer).
        """
        assert mode in ("ml", "dl"), "mode must be 'ml' or 'dl'"
        self.mode = mode
        logger.info(f"TextPreprocessor initialized in '{mode}' mode")

    def remove_urls(self, text: str) -> str:
        return self.URL_PATTERN.sub(" ", text)

    def remove_mentions(self, text: str) -> str:
        return self.MENTION_PATTERN.sub(" ", text)

    def expand_hashtags(self, text: str) -> str:
        """Converts #SomeTopic → SomeTopic (preserves content signal)."""
        return self.HASHTAG_PATTERN.sub(r"\1", text)

    def remove_html(self, text: str) -> str:
        return self.HTML_PATTERN.sub(" ", text)

    def remove_emojis(self, text: str) -> str:
        return self.EMOJI_PATTERN.sub(" ", text)

    def normalize_unicode(self, text: str) -> str:
        return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")

    def remove_special_chars(self, text: str) -> str:
        return self.SPECIAL_CHARS_PATTERN.sub(" ", text)

    def normalize_whitespace(self, text: str) -> str:
        return self.WHITESPACE_PATTERN.sub(" ", text).strip()

    def to_lowercase(self, text: str) -> str:
        return text.lower()

    def clean_ml(self, text: str) -> str:
        """
        Heavy cleaning for classical ML (TF-IDF).
        Pipeline: HTML → URLs → mentions → hashtag_expand → 
                  emojis → unicode → lowercase → special_chars → whitespace
        """
        text = self.remove_html(text)
        text = self.remove_urls(text)
        text = self.remove_mentions(text)
        text = self.expand_hashtags(text)
        text = self.remove_emojis(text)
        text = self.normalize_unicode(text)
        text = self.to_lowercase(text)
        text = self.remove_special_chars(text)
        text = self.normalize_whitespace(text)
        return text

    def clean_dl(self, text: str) -> str:
        """
        Light cleaning for transformers (preserve more context).
        Transformers benefit from punctuation, capitalization signals.
        """
        text = self.remove_html(text)
        text = self.remove_urls(text)
        text = self.remove_mentions(text)
        text = self.expand_hashtags(text)
        # Keep emojis partially: BERT tokenizer handles unknown tokens gracefully
        text = self.remove_emojis(text)
        text = self.normalize_whitespace(text)
        return text

    def clean(self, text: str) -> str:
        """Dispatch to correct cleaning mode."""
        if not isinstance(text, str) or not text.strip():
            return ""
        if self.mode == "ml":
            return self.clean_ml(text)
        return self.clean_dl(text)

    def process_dataframe(self, df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
        """Clean a full DataFrame column in-place (adds cleaned_text column)."""
        df = df.copy()
        col = f"cleaned_text_{self.mode}"
        df[col] = df[text_col].apply(self.clean)
        # Drop rows where cleaning resulted in empty text
        before = len(df)
        df = df[df[col].str.strip().str.len() > 0].reset_index(drop=True)
        dropped = before - len(df)
        if dropped:
            logger.warning(f"Dropped {dropped} empty rows after cleaning")
        logger.info(f"Preprocessing complete. {len(df)} rows retained.")
        return df


if __name__ == "__main__":
    samples = [
        "I absolutely LOVE this product!!! 🎉🎊 Check it out at https://example.com #Amazing @friend",
        "<p>Terrible experience. Never buying again.</p> 😤",
        "It was okay. Nothing special. #meh",
    ]
    for mode in ("ml", "dl"):
        p = TextPreprocessor(mode=mode)
        print(f"\n--- {mode.upper()} MODE ---")
        for s in samples:
            print(f"  IN : {s}")
            print(f"  OUT: {p.clean(s)}")
            print()
