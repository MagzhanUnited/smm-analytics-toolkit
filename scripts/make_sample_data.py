"""Generate the synthetic sample dataset in data/sample/.

The data is SYNTHETIC: post metrics come from a seeded random model, and the
comments were written for testing. It reproduces the structure of real
Instagram/Telegram exports without containing anyone's personal data.

    python scripts/make_sample_data.py
"""

from __future__ import annotations

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

SEED = 20260907
OUT = Path(__file__).resolve().parent.parent / "data" / "sample"

# Baseline ER by format and a bonus for evening slots: the pattern the
# metrics module is expected to recover.
FORMAT_ER = {"reels": 0.062, "carousel": 0.048, "photo": 0.031, "text": 0.022}
CHANNEL_REACH = {"instagram": (1800, 600), "telegram": (900, 250)}
HOURS = (9, 12, 13, 15, 18, 19, 20, 21)

COMMENTS = [
    # (text, label) written by hand; includes hard cases and personal data to be scrubbed
    ("Отличный сервис, всё быстро сделали, спасибо!", "positive"),
    ("Рахмет, керемет жұмыс 👍", "positive"),
    ("Очень нравится новый формат постов", "positive"),
    ("Рекомендую всем, кто ищет надёжного подрядчика", "positive"),
    ("Жақсы, бәрі ұнады", "positive"),
    ("Супер, заказали второй раз 🔥", "positive"),
    ("Молодцы, приятно работать", "positive"),
    ("Тамаша! Мықты команда", "positive"),
    ("Без проблем установили за один день", "positive"),
    ("Обожаю ваши обзоры, лучшие в городе", "positive"),
    ("Класс, удобно что можно оплатить онлайн", "positive"),
    ("Спасибо @aliya_k за быструю консультацию", "positive"),
    ("Наконец-то дождались, всё как обещали", "positive"),
    ("Сервис на высоте, так держать", "positive"),
    ("Ужасно долго ждали мастера", "negative"),
    ("Не работает кнопка оплаты уже второй день", "negative"),
    ("Верните деньги, обман", "negative"),
    ("Нашар қызмет, баяу жауап береді", "negative"),
    ("Плохая связь, оператор не берёт трубку 👎", "negative"),
    ("Разочарован, ожидал большего", "negative"),
    ("Сапасыз жұмыс, ұнамады", "negative"),
    ("Пишите на ivan.petrov@example.com, жалоба по заказу", "negative"),
    ("Звонил на +7 701 000 00 00 никто не ответил, кошмар", "negative"),
    ("Совсем не понравилось обслуживание", "negative"),
    ("Опять сломался лифт в третьем подъезде", "negative"),
    ("Хамство менеджера, больше не обращусь 😡", "negative"),
    ("Жаман, қайтарыңыз ақшамды", "negative"),
    ("Ну да, конечно, очень быстро, всего три недели", "negative"),  # sarcasm: hard case
    ("Пятый день без ответа", "negative"),  # no lexicon words: hard case
    ("Цена изменилась после заказа, так нельзя", "negative"),  # hard case
    ("Во сколько вы работаете в субботу?", "neutral"),
    ("Есть ли доставка в Косшы?", "neutral"),
    ("Сколько стоит обслуживание за месяц?", "neutral"),
    ("Қашан ашыласыздар?", "neutral"),
    ("Подскажите адрес офиса", "neutral"),
    ("Это акция только для новых клиентов?", "neutral"),
    ("Скиньте прайс в директ", "neutral"),
    ("Жеткізу бар ма?", "neutral"),
    ("А в Алматы есть филиал? Сайт www.example.kz не открывается", "neutral"),
    ("Можно оплатить через Kaspi?", "neutral"),
    ("Видел ваш пост в телеграме", "neutral"),
    ("Уточните сроки, пожалуйста", "neutral"),
    ("Нормально, но цены высокие", "neutral"),  # mixed: hard case
    ("Хорошо, но долго", "neutral"),  # mixed: hard case
    ("Не плохо", "positive"),  # double negative: hard case
    ("Не очень хорошо получилось", "negative"),
]


def make_posts(rng: random.Random) -> list:
    start = datetime(2026, 9, 7)
    rows, n = [], 0
    for day in range(42):
        date = start + timedelta(days=day)
        for channel, (mu, sigma) in CHANNEL_REACH.items():
            if rng.random() < 0.55:
                continue
            n += 1
            fmt = rng.choice(list(FORMAT_ER))
            hour = rng.choice(HOURS)
            reach = max(80, int(rng.gauss(mu, sigma)))
            er = FORMAT_ER[fmt] * (1.35 if hour >= 18 else 1.0) * (1.15 if date.weekday() >= 5 else 1.0)
            er *= rng.uniform(0.7, 1.3)
            interactions = int(reach * er)
            likes = int(interactions * 0.72)
            comments = int(interactions * 0.12)
            shares = int(interactions * 0.09)
            saves = max(0, interactions - likes - comments - shares)
            rows.append(
                {
                    "post_id": f"p{n:03d}",
                    "channel": channel,
                    "published_at": date.replace(hour=hour, minute=rng.choice((0, 15, 30))).isoformat(
                        timespec="minutes"
                    ),
                    "format": fmt,
                    "reach": reach,
                    "likes": likes,
                    "comments": comments,
                    "shares": shares,
                    "saves": saves,
                }
            )
    return rows


def make_comments(rng: random.Random, post_ids: list) -> list:
    return [
        {
            "comment_id": f"c{i:03d}",
            "post_id": rng.choice(post_ids),
            "author": f"user_{rng.randint(1, 30):03d}",
            "text": text,
            "label": label,
        }
        for i, (text, label) in enumerate(COMMENTS, start=1)
    ]


def write(path: Path, rows: list) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rng = random.Random(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    posts = make_posts(rng)
    write(OUT / "posts.csv", posts)
    write(OUT / "comments_labelled.csv", make_comments(rng, [p["post_id"] for p in posts]))
    print(f"wrote {len(posts)} posts and {len(COMMENTS)} comments to {OUT}")


if __name__ == "__main__":
    main()
