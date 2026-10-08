import pytest

from smmkit.cli import main


def test_report_writes_markdown(sample_dir, tmp_path, capsys):
    code = main(
        [
            "report",
            "--posts",
            str(sample_dir / "posts.csv"),
            "--comments",
            str(sample_dir / "comments_labelled.csv"),
            "--out",
            str(tmp_path),
        ]
    )
    assert code == 0
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Engagement rate by channel" in text
    assert "Recommended publication slots" in text
    assert "macro-F1" in text


def test_report_with_plots(sample_dir, tmp_path):
    pytest.importorskip("matplotlib")
    main(
        [
            "report",
            "--posts",
            str(sample_dir / "posts.csv"),
            "--comments",
            str(sample_dir / "comments_labelled.csv"),
            "--out",
            str(tmp_path),
            "--plots",
        ]
    )
    assert (tmp_path / "engagement_heatmap.png").stat().st_size > 0
    assert (tmp_path / "confusion_matrix.png").stat().st_size > 0


def test_metrics_command(sample_dir, capsys):
    assert main(["metrics", str(sample_dir / "posts.csv"), "--by", "format"]) == 0
    assert "reels" in capsys.readouterr().out


def test_evaluate_command(sample_dir, capsys):
    assert main(["evaluate", str(sample_dir / "comments_labelled.csv"), "--salt", "x"]) == 0
    assert "macro_f1=" in capsys.readouterr().out


def test_bad_input_exits_with_code_2(tmp_path, capsys):
    assert main(["metrics", str(tmp_path / "missing.csv")]) == 2
    assert "error:" in capsys.readouterr().err


def test_evaluate_without_labels(write_csv, capsys):
    path = write_csv("c.csv", ("comment_id", "post_id", "author", "text"), [["c1", "p1", "u", "t"]])
    assert main(["evaluate", str(path)]) == 2
    assert "no labelled comments" in capsys.readouterr().err
