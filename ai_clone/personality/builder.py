from __future__ import annotations
import re
from collections import Counter
from typing import Dict, List

from ..ingestion.models import ParsedMessage
from .models import PersonalityProfile

_EMOJI_RE = re.compile(
    "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0\U000024C2-\U0001F251]+",
    flags=re.UNICODE,
)
_INFORMAL = frozenset({
    "lol", "lmao", "omg", "brb", "tbh", "ngl", "idk",
    "imo", "rn", "gonna", "wanna", "gotta",
})
_PUNCT = "!?.,;:"


class PersonalityBuilder:
    def build(self, messages: List[ParsedMessage]) -> PersonalityProfile:
        if not messages:
            return PersonalityProfile()
        texts = [m.text for m in messages]
        return PersonalityProfile(
            avg_sentence_length=self._avg_words(texts),
            emoji_frequency=self._emoji_freq(texts),
            top_phrases=self._top_ngrams(texts, n=2, top_k=20),
            formality_score=self._formality(texts),
            punctuation_style=self._punct_style(texts),
        )

    def _avg_words(self, texts: List[str]) -> float:
        counts = [len(t.split()) for t in texts if t.strip()]
        return sum(counts) / len(counts) if counts else 0.0

    def _emoji_freq(self, texts: List[str]) -> float:
        total = sum(len(_EMOJI_RE.findall(t)) for t in texts)
        return total / len(texts) if texts else 0.0

    def _top_ngrams(self, texts: List[str], n: int, top_k: int) -> List[str]:
        counter: Counter = Counter()
        for text in texts:
            words = re.findall(r"\b\w+\b", text.lower())
            for i in range(len(words) - n + 1):
                counter[" ".join(words[i : i + n])] += 1
        return [phrase for phrase, _ in counter.most_common(top_k)]

    def _formality(self, texts: List[str]) -> float:
        total = informal = 0
        for text in texts:
            words = re.findall(r"\b\w+\b", text.lower())
            total += len(words)
            informal += sum(1 for w in words if w in _INFORMAL)
        if total == 0:
            return 0.5
        return max(0.0, min(1.0, 1.0 - (informal / total) * 10))

    def _punct_style(self, texts: List[str]) -> Dict[str, float]:
        counter: Counter = Counter()
        for text in texts:
            for ch in text:
                if ch in _PUNCT:
                    counter[ch] += 1
        total = sum(counter.values())
        if total == 0:
            return {p: 0.0 for p in _PUNCT}
        return {p: counter.get(p, 0) / total for p in _PUNCT}
