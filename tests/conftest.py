from datetime import datetime
from pathlib import Path

import pytest

from smmkit.io import Post

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample"


def make_post(
    post_id="p1",
    channel="instagram",
    when="2026-09-14T18:30",
    fmt="reels",
    reach=1000,
    likes=40,
    comments=5,
    shares=3,
    saves=2,
):
    return Post(post_id, channel, datetime.fromisoformat(when), fmt, reach, likes, comments, shares, saves)


@pytest.fixture
def sample_dir():
    return SAMPLE


@pytest.fixture
def write_csv(tmp_path):
    def _write(name, header, rows):
        path = tmp_path / name
        lines = [",".join(header)] + [",".join(str(v) for v in row) for row in rows]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    return _write
