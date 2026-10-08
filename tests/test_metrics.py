import pytest
from conftest import make_post

from smmkit.io import load_posts
from smmkit.metrics import best_slots, engagement_rate, summarise


def test_engagement_rate():
    assert engagement_rate(make_post(reach=1000)) == pytest.approx(0.05)


def test_zero_reach_gives_zero():
    assert engagement_rate(make_post(reach=0)) == 0.0


def test_summarise_groups_and_sorts():
    posts = [
        make_post("a", channel="instagram", likes=90),
        make_post("b", channel="instagram", likes=10),
        make_post("c", channel="telegram", likes=0, comments=0, shares=0, saves=0),
    ]
    stats = summarise(posts, by="channel")
    assert [s.key for s in stats] == ["instagram", "telegram"]
    assert stats[0].posts == 2 and stats[0].total_reach == 2000
    assert stats[0].mean_er == pytest.approx((0.1 + 0.02) / 2)


@pytest.mark.parametrize("by, key", [("weekday", "Mon"), ("hour", "18"), ("format", "reels")])
def test_summarise_dimensions(by, key):
    assert summarise([make_post()], by=by)[0].key == key


def test_unknown_grouping():
    with pytest.raises(ValueError, match="unknown grouping"):
        summarise([make_post()], by="colour")


def test_best_slots_ignore_single_posts():
    posts = [
        make_post("lucky", when="2026-09-15T09:00", likes=500),  # one post only
        make_post("a", when="2026-09-14T18:00", likes=60),
        make_post("b", when="2026-09-21T19:30", likes=40),
    ]
    slots = best_slots(posts, top=3, min_posts=2)
    assert len(slots) == 1
    assert (slots[0].weekday, slots[0].hours, slots[0].posts) == ("Mon", "18-21", 2)


def test_sample_recovers_evening_and_reels_pattern(sample_dir):
    """The synthetic data has higher ER for reels and for evening posts; metrics must find it."""
    posts = load_posts(sample_dir / "posts.csv")
    assert summarise(posts, by="format")[0].key == "reels"
    evening = [engagement_rate(p) for p in posts if p.published_at.hour >= 18]
    daytime = [engagement_rate(p) for p in posts if p.published_at.hour < 18]
    assert sum(evening) / len(evening) > sum(daytime) / len(daytime)
