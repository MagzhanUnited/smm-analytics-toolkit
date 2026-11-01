import json
import threading
import urllib.error
import urllib.request

import pytest

from smmkit.cli import main
from smmkit.io import load_posts
from smmkit.serve import make_server


@pytest.fixture
def server(sample_dir):
    srv = make_server(load_posts(sample_dir / "posts.csv"), "# report\n", host="127.0.0.1", port=0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()
    srv.server_close()


def get(url):
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return resp.status, resp.headers["Content-Type"], resp.read().decode("utf-8")
    except urllib.error.HTTPError as err:
        return err.code, err.headers["Content-Type"], err.read().decode("utf-8")


def test_healthz(server):
    status, ctype, body = get(server + "/healthz")
    assert status == 200
    assert ctype.startswith("application/json")
    payload = json.loads(body)
    assert payload["status"] == "ok"
    assert payload["posts"] > 0


def test_report_root(server):
    status, ctype, body = get(server + "/")
    assert status == 200
    assert ctype.startswith("text/markdown")
    assert body == "# report\n"


def test_metrics_by_format(server):
    status, _, body = get(server + "/api/metrics?by=format")
    assert status == 200
    rows = json.loads(body)
    assert {"key", "posts", "mean_er", "median_er", "total_reach"} <= set(rows[0])


def test_metrics_default_grouping_is_channel(server):
    _, _, body = get(server + "/api/metrics")
    keys = {row["key"] for row in json.loads(body)}
    assert "instagram" in keys


def test_metrics_rejects_unknown_grouping(server):
    status, _, body = get(server + "/api/metrics?by=moon")
    assert status == 400
    assert "allowed" in json.loads(body)


def test_unknown_route(server):
    status, _, _ = get(server + "/nope")
    assert status == 404


def test_cli_serve_once(sample_dir, capsys):
    # --port 0 picks a free port; --once handles one request, so the request is sent from a thread
    import re

    result = {}

    def call():
        for _ in range(100):
            out = capsys.readouterr().out
            match = re.search(r":(\d+)$", out.strip())
            if match:
                result["status"] = get(f"http://127.0.0.1:{match.group(1)}/healthz")[0]
                return
            threading.Event().wait(0.05)

    client = threading.Thread(target=call)
    client.start()
    code = main(
        [
            "serve",
            "--posts",
            str(sample_dir / "posts.csv"),
            "--comments",
            str(sample_dir / "comments_labelled.csv"),
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--once",
        ]
    )
    client.join(timeout=10)
    assert code == 0
    assert result["status"] == 200
