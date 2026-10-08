"""Sentiment classification baseline for Russian and Kazakh comments, and its evaluation.

The lexicon classifier is a transparent baseline: any stronger model (a
fine-tuned transformer or an LLM) is plugged in through the same ``predict``
interface and evaluated with the same :func:`evaluate`, so results stay
comparable across experiments.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

LABELS = ("negative", "neutral", "positive")

# Stems match by prefix, which covers most Russian and Kazakh inflections.
POSITIVE_STEMS = (
    # ru
    "отличн",
    "супер",
    "класс",
    "спасиб",
    "нрав",
    "понрав",
    "люблю",
    "обожа",
    "рекоменд",
    "довол",
    "хорош",
    "быстр",
    "вкусн",
    "лучш",
    "крут",
    "удобн",
    "приятн",
    "молодц",
    "восторг",
    "прекрасн",
    # kk
    "рахмет",
    "керемет",
    "жақсы",
    "ұнады",
    "ұнайды",
    "тамаша",
    "мықты",
    "жылдам",
    "дәмді",
)
NEGATIVE_STEMS = (
    # ru
    "плох",
    "ужас",
    "отврат",
    "долг",
    "медлен",
    "обман",
    "верните",
    "разочар",
    "хуж",
    "кошмар",
    "грязн",
    "хамств",
    "сломал",
    "не работ",
    "жалоб",
    "отказ",
    "проблем",
    # kk
    "нашар",
    "жаман",
    "баяу",
    "сапасыз",
    "ұнамады",
    "қайтарыңыз",
    "алдады",
)
# Russian negation precedes the word ("не нравится"); Kazakh follows it ("жақсы емес").
PRE_NEGATIONS = ("не", "нет", "ни", "без")
POST_NEGATIONS = ("емес", "жоқ")
POSITIVE_EMOJI = ("👍", "❤", "😍", "🔥", "👏", "😊", "🥰")
NEGATIVE_EMOJI = ("👎", "😡", "🤬", "😠", "💩")

_TOKEN = re.compile(r"[\w']+", re.UNICODE)


def _polarity(token: str) -> int:
    if token.startswith(POSITIVE_STEMS):
        return 1
    if token.startswith(NEGATIVE_STEMS):
        return -1
    return 0


@dataclass
class LexiconClassifier:
    """Rule-based classifier: sum of token polarities with negation handling."""

    threshold: int = 1

    def score(self, text: str) -> int:
        lowered = text.lower()
        score = sum(lowered.count(e) for e in POSITIVE_EMOJI)
        score -= sum(lowered.count(e) for e in NEGATIVE_EMOJI)
        # multi-word negative stems ("не работ") are matched on the raw text first
        for stem in NEGATIVE_STEMS:
            if " " in stem and stem in lowered:
                score -= 1
                lowered = lowered.replace(stem, " ")
        negate, last = False, 0
        for token in _TOKEN.findall(lowered):
            if token in PRE_NEGATIONS:
                negate = True
                continue
            if token in POST_NEGATIONS:
                score -= 2 * last  # flip the contribution of the preceding sentiment word
                last = 0
                continue
            polarity = _polarity(token)
            last = -polarity if negate else polarity
            score += last
            negate = False
        return score

    def predict_one(self, text: str) -> str:
        s = self.score(text)
        if s >= self.threshold:
            return "positive"
        if s <= -self.threshold:
            return "negative"
        return "neutral"

    def predict(self, texts) -> list:
        return [self.predict_one(t) for t in texts]


@dataclass
class EvaluationReport:
    accuracy: float
    macro_f1: float
    per_class: dict = field(default_factory=dict)  # label -> {precision, recall, f1, support}
    confusion: dict = field(default_factory=dict)  # true label -> {predicted label -> count}

    def as_rows(self) -> list:
        return [
            (label, m["precision"], m["recall"], m["f1"], m["support"]) for label, m in self.per_class.items()
        ]


def evaluate(y_true, y_pred, labels=LABELS) -> EvaluationReport:
    """Accuracy, per-class precision/recall/F1, macro-F1 and the confusion matrix."""
    y_true, y_pred = list(y_true), list(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        raise ValueError("cannot evaluate an empty sample")
    confusion = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(y_true, y_pred):
        if t not in confusion or p not in confusion:
            raise ValueError(f"unknown label in pair ({t!r}, {p!r})")
        confusion[t][p] += 1
    per_class = {}
    for label in labels:
        tp = confusion[label][label]
        predicted = sum(confusion[t][label] for t in labels)
        actual = sum(confusion[label].values())
        precision = tp / predicted if predicted else 0.0
        recall = tp / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": actual}
    accuracy = sum(confusion[label][label] for label in labels) / len(y_true)
    macro_f1 = sum(m["f1"] for m in per_class.values()) / len(labels)
    return EvaluationReport(accuracy, macro_f1, per_class, confusion)
