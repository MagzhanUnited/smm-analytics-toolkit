import pytest

from smmkit.sentiment import LexiconClassifier, evaluate

clf = LexiconClassifier()


@pytest.mark.parametrize(
    "text, label",
    [
        ("Отличный сервис, спасибо!", "positive"),
        ("Рахмет, керемет 👍", "positive"),
        ("Ужасно долго ждали", "negative"),
        ("Нашар қызмет", "negative"),
        ("Не работает кнопка оплаты", "negative"),
        ("Во сколько вы открываетесь?", "neutral"),
        ("Не понравилось", "negative"),  # negation flips a positive stem
        ("Без проблем установили", "positive"),  # negation flips a negative stem
        ("жақсы емес", "negative"),  # Kazakh negation follows the word
        ("проблема жоқ", "positive"),
        ("😡😡", "negative"),
    ],
)
def test_lexicon_classifier(text, label):
    assert clf.predict_one(text) == label


def test_threshold_controls_neutral_band():
    assert LexiconClassifier(threshold=2).predict_one("хорошо") == "neutral"


def test_predict_is_batch_of_predict_one():
    texts = ["супер", "плохо", "адрес?"]
    assert clf.predict(texts) == [clf.predict_one(t) for t in texts]


def test_evaluate_perfect():
    y = ["positive", "neutral", "negative"]
    report = evaluate(y, y)
    assert report.accuracy == 1.0 and report.macro_f1 == 1.0


def test_evaluate_known_values():
    y_true = ["positive", "positive", "negative", "neutral"]
    y_pred = ["positive", "neutral", "negative", "neutral"]
    report = evaluate(y_true, y_pred)
    assert report.accuracy == pytest.approx(0.75)
    pos = report.per_class["positive"]
    assert (pos["precision"], pos["recall"], pos["support"]) == (1.0, 0.5, 2)
    assert report.per_class["neutral"]["precision"] == pytest.approx(0.5)
    assert report.confusion["positive"]["neutral"] == 1


@pytest.mark.parametrize(
    "y_true, y_pred, match",
    [(["positive"], [], "same length"), ([], [], "empty"), (["angry"], ["positive"], "unknown label")],
)
def test_evaluate_rejects_bad_input(y_true, y_pred, match):
    with pytest.raises(ValueError, match=match):
        evaluate(y_true, y_pred)
