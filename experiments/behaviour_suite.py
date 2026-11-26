"""Behavioural test suite for sentiment classifiers on Russian and Kazakh comments.

The suite follows the CheckList methodology (Ribeiro et al., 2020). Every case is a
short comment together with the set of labels that a correct classifier may return,
and cases are grouped by linguistic capability. The vocabulary is restricted to words
that the lexicon baseline knows, so pass rates measure how a classifier composes
polarity (negation, its scope, code-switching, mixed opinions) and not how large its
lexicon is.

Conventions:

* a negated negative word ("не плохо", "жаман емес") must not be labelled negative;
  both ``positive`` and ``neutral`` are accepted, because annotators disagree on litotes;
* a comment with one positive and one negative part is ``neutral``, the convention of
  the labelled sample in ``data/sample``.

All sentences were written for this suite. None is taken from real users.
"""

from __future__ import annotations

from dataclasses import dataclass

POSITIVE = frozenset({"positive"})
NEGATIVE = frozenset({"negative"})
NEUTRAL = frozenset({"neutral"})
NOT_NEGATIVE = frozenset({"positive", "neutral"})


@dataclass(frozen=True)
class Capability:
    key: str
    language: str  # "ru", "kk", "ru+kk" or "any"
    description: str


@dataclass(frozen=True)
class Case:
    capability: str
    text: str
    expected: frozenset


CAPABILITIES = (
    Capability("V-RU", "ru", "Vocabulary: single polar word in a neutral frame"),
    Capability("V-KK", "kk", "Vocabulary: single polar word in a neutral frame"),
    Capability("N1-RU", "ru", "Negation directly before a positive word"),
    Capability("N2-RU", "ru", "Negation directly before a negative word (litotes)"),
    Capability("N3-RU", "ru", "'без' / 'нет' directly before a negative noun"),
    Capability("N4-RU", "ru", "Negation separated from the polar word by one word"),
    Capability("N5-RU", "ru", "Postposed 'нет' and negative pronouns"),
    Capability("N1-KK", "kk", "'емес' after a positive predicate"),
    Capability("N2-KK", "kk", "'емес' after a negative predicate (litotes)"),
    Capability("N3-KK", "kk", "Existential 'жоқ' after a negative noun"),
    Capability("N4-KK", "kk", "Verbal negation: suffix -ма/-ме, 'болмады', '-ған жоқ'"),
    Capability("CS", "ru+kk", "Russian and Kazakh words in one comment"),
    Capability("MIX", "any", "One positive and one negative part"),
    Capability("EMO", "any", "Polarity carried only by an emoji"),
)

RU_FRAMES = ("Получилось {}", "В целом {}", "Всё {}")
RU_POSITIVE = (
    "отлично",
    "хорошо",
    "быстро",
    "удобно",
    "прекрасно",
    "круто",
    "приятно",
    "вкусно",
    "супер",
    "классно",
)
RU_NEGATIVE = ("плохо", "ужасно", "отвратительно", "медленно", "долго", "грязно")
RU_GRADABLE = ("хорошо", "быстро", "удобно", "приятно", "вкусно", "круто")
RU_INTENSIFIERS = ("очень", "слишком", "особо", "совсем")
RU_NEGATIVE_NOUNS = ("проблем", "жалоб", "отказов", "обмана", "хамства")  # genitive forms

KK_POSITIVE = (
    "Қызмет жақсы",
    "Қызмет керемет",
    "Қызмет тамаша",
    "Қызмет жылдам",
    "Жұмыс жақсы",
    "Жұмыс керемет",
    "Жұмыс тамаша",
    "Команда мықты",
    "Жеткізу жылдам",
    "Тамақ дәмді",
)
KK_NEGATIVE = (
    "Қызмет нашар",
    "Қызмет жаман",
    "Қызмет баяу",
    "Қызмет сапасыз",
    "Жұмыс нашар",
    "Жұмыс жаман",
    "Жұмыс сапасыз",
    "Жеткізу баяу",
)


def _vocabulary() -> list:
    cases = [Case("V-RU", f.format(w), POSITIVE) for f in RU_FRAMES for w in RU_POSITIVE]
    cases += [Case("V-RU", f.format(w), NEGATIVE) for f in RU_FRAMES for w in RU_NEGATIVE]
    cases += [Case("V-KK", s, POSITIVE) for s in KK_POSITIVE]
    cases += [Case("V-KK", s, POSITIVE) for s in ("Маған ұнады", "Маған ұнайды", "Бәрі ұнады", "Көп рахмет")]
    cases += [Case("V-KK", s, NEGATIVE) for s in KK_NEGATIVE]
    return cases


def _russian_negation() -> list:
    cases = [Case("N1-RU", f.format(f"не {w}"), NEGATIVE) for f in RU_FRAMES for w in RU_POSITIVE]
    cases += [Case("N2-RU", f.format(f"не {w}"), NOT_NEGATIVE) for f in RU_FRAMES for w in RU_NEGATIVE]
    for frame in ("Всё прошло без {}", "Работают без {}", "Нет {}"):
        cases += [Case("N3-RU", frame.format(n), NOT_NEGATIVE) for n in RU_NEGATIVE_NOUNS]
    cases += [Case("N4-RU", f"Получилось не {i} {w}", NEGATIVE) for i in RU_INTENSIFIERS for w in RU_GRADABLE]
    cases += [Case("N4-RU", f"Не было {n}", NOT_NEGATIVE) for n in RU_NEGATIVE_NOUNS]
    cases += [Case("N5-RU", f"{n.capitalize()} нет", NOT_NEGATIVE) for n in RU_NEGATIVE_NOUNS]
    cases += [Case("N5-RU", f"Никаких {n}", NOT_NEGATIVE) for n in ("проблем", "жалоб", "отказов")]
    cases += [Case("N5-RU", f"Никакого {n}", NOT_NEGATIVE) for n in ("обмана", "хамства")]
    return cases


def _kazakh_negation() -> list:
    cases = [Case("N1-KK", f"{s} емес", NEGATIVE) for s in KK_POSITIVE]
    cases += [Case("N2-KK", f"{s} емес", NOT_NEGATIVE) for s in KK_NEGATIVE]
    cases += [
        Case("N3-KK", s, NOT_NEGATIVE)
        for s in (
            "Проблема жоқ",
            "Проблемалар жоқ",
            "Ешқандай проблема жоқ",
            "Жалоба жоқ",
            "Ешқандай жалоба жоқ",
            "Обман жоқ",
        )
    ]
    verbal = ("Маған ұнамады", "Бізге ұнамады", "Маған ұнамайды", "Бізге ұнамайды", "Маған ұнаған жоқ")
    did_not = (
        "Қызмет жақсы болмады",
        "Жұмыс жақсы болмады",
        "Жеткізу жылдам болмады",
        "Тамақ дәмді болмады",
        "Қызмет керемет болмады",
    )
    was_not = (
        "Қызмет жақсы болған жоқ",
        "Жұмыс жақсы болған жоқ",
        "Жеткізу жылдам болған жоқ",
        "Тамақ дәмді болған жоқ",
    )
    cases += [Case("N4-KK", s, NEGATIVE) for s in verbal + did_not + was_not]
    return cases


def _other() -> list:
    cases = [
        Case("CS", s, POSITIVE)
        for s in (
            "Сервис жақсы",
            "Сервис керемет",
            "Доставка жылдам",
            "Рахмет, всё отлично",
            "Рахмет, очень удобно",
            "Жақсы, спасибо",
            "Керемет, рекомендую",
        )
    ]
    cases += [
        Case("CS", s, NEGATIVE)
        for s in (
            "Сервис нашар",
            "Доставка баяу",
            "Жаман, верните деньги",
            "Сапасыз, очень медленно",
            "Сервис жақсы емес",
            "Доставка жылдам емес",
        )
    ]
    cases += [
        Case("MIX", s, NEUTRAL)
        for s in (
            "Хорошо, но долго",
            "Быстро, но плохо",
            "Вкусно, но медленно",
            "Удобно, но долго",
            "Круто, но грязно",
            "Жақсы, бірақ баяу",
            "Жылдам, бірақ сапасыз",
            "Дәмді, бірақ баяу",
        )
    ]
    cases += [
        Case("EMO", s, POSITIVE) for s in ("Заказ получили 👍", "Заказ получили 🔥", "Жеткізу келді 👍", "❤❤")
    ]
    cases += [
        Case("EMO", s, NEGATIVE) for s in ("Курьер приехал 😡", "Заказ получили 👎", "😡😡", "Опять 🤬")
    ]
    return cases


def build_cases() -> list:
    """All cases of the suite, in the order of :data:`CAPABILITIES`."""
    cases = _vocabulary() + _russian_negation() + _kazakh_negation() + _other()
    order = {c.key: i for i, c in enumerate(CAPABILITIES)}
    return sorted(cases, key=lambda c: order[c.capability])
