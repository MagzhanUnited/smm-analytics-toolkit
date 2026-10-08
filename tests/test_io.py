import pytest

from smmkit.io import COMMENT_COLUMNS, POST_COLUMNS, DataError, load_comments, load_posts

ROW = ["p1", "Instagram", "2026-09-14T18:30", "Reels", 1000, 40, 5, 3, 2]


def test_load_posts_normalises_case(write_csv):
    posts = load_posts(write_csv("posts.csv", POST_COLUMNS, [ROW]))
    assert posts[0].channel == "instagram" and posts[0].format == "reels"
    assert posts[0].interactions == 50


def test_missing_column_is_reported(write_csv):
    path = write_csv("posts.csv", POST_COLUMNS[:-1], [ROW[:-1]])
    with pytest.raises(DataError, match="missing columns: saves"):
        load_posts(path)


def test_bad_integer_names_row_and_column(write_csv):
    path = write_csv("posts.csv", POST_COLUMNS, [ROW, ROW[:4] + ["n/a"] + ROW[5:]])
    with pytest.raises(DataError, match=r"posts.csv:3: column 'reach'"):
        load_posts(path)


def test_negative_numbers_are_rejected(write_csv):
    path = write_csv("posts.csv", POST_COLUMNS, [ROW[:5] + [-1] + ROW[6:]])
    with pytest.raises(DataError, match="non-negative"):
        load_posts(path)


def test_bad_date(write_csv):
    path = write_csv("posts.csv", POST_COLUMNS, [ROW[:2] + ["14.09.2026"] + ROW[3:]])
    with pytest.raises(DataError, match="ISO 8601"):
        load_posts(path)


def test_missing_file(tmp_path):
    with pytest.raises(DataError, match="not found"):
        load_posts(tmp_path / "nope.csv")


def test_comments_are_scrubbed_and_pseudonymised(write_csv):
    path = write_csv(
        "comments.csv",
        COMMENT_COLUMNS + ("label",),
        [["c1", "p1", "user_001", "пишите a@b.kz", "Neutral"]],
    )
    (c,) = load_comments(path, salt="s")
    assert c.text == "пишите <EMAIL>"
    assert c.author != "user_001"
    assert c.label == "neutral"


def test_unknown_label(write_csv):
    path = write_csv("comments.csv", COMMENT_COLUMNS + ("label",), [["c1", "p1", "u", "t", "angry"]])
    with pytest.raises(DataError, match="unknown label"):
        load_comments(path, salt="s")


def test_sample_data_loads(sample_dir):
    posts = load_posts(sample_dir / "posts.csv")
    comments = load_comments(sample_dir / "comments_labelled.csv", salt="s")
    assert len(posts) > 30
    assert all(c.label for c in comments)
    assert not any("@" in c.text for c in comments)
