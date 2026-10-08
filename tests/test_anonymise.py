import pytest

from smmkit.anonymise import pseudonymise, scrub_text


@pytest.mark.parametrize(
    "text, expected",
    [
        ("пишите на ivan.petrov@example.com", "пишите на <EMAIL>"),
        ("звоните +7 701 000 00 00 вечером", "звоните <PHONE> вечером"),
        ("тел 8(777)123-45-67", "тел <PHONE>"),
        ("спасибо @aliya_k!", "спасибо <USER>!"),
        ("см. https://example.kz/a?b=1 и www.site.kz", "см. <URL> и <URL>"),
        ("без персональных данных", "без персональных данных"),
    ],
)
def test_scrub_text(text, expected):
    assert scrub_text(text) == expected


def test_email_is_not_mistaken_for_handle():
    assert "<USER>" not in scrub_text("a.b@mail.kz")


def test_short_numbers_are_kept():
    assert scrub_text("заказ 12345 на 3 шт") == "заказ 12345 на 3 шт"


def test_pseudonym_is_stable_and_salted():
    a = pseudonymise("user_001", "salt-1")
    assert a == pseudonymise("user_001", "salt-1")
    assert a != pseudonymise("user_001", "salt-2")
    assert a != pseudonymise("user_002", "salt-1")
    assert len(a) == 12 and "user" not in a


def test_empty_salt_is_rejected():
    with pytest.raises(ValueError):
        pseudonymise("user_001", "")
