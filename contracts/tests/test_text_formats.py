import pytest

from robin_contracts.text_formats import InvalidText, load_json, load_yaml

SECRET = "hunter2"


def test_load_yaml_keeps_a_date_as_text():
    loaded = load_yaml("from: 2026-05-01\nat: 2026-05-01T05:00:00Z\n")

    assert loaded == {"from": "2026-05-01", "at": "2026-05-01T05:00:00Z"}


@pytest.mark.parametrize(
    ("text", "key", "line", "column"),
    [
        ("site: '04'\nsite: '05'\n", "site", 2, 1),
        ("date:\n  from: a\n  from: b\n", "from", 3, 3),
    ],
    ids=["top", "nested"],
)
def test_load_yaml_refuses_a_repeated_key_at_its_position(text, key, line, column):
    with pytest.raises(InvalidText) as raised:
        load_yaml(text)

    assert (raised.value.line, raised.value.column) == (line, column)
    assert f"{key!r}" in str(raised.value)
    assert "more than once" in str(raised.value)


@pytest.mark.parametrize(
    "text",
    [
        f'password: "{SECRET}\n',
        f"password: [{SECRET}\n",
        f"password: !!binary {SECRET}!\n",
        f"password: !!int {SECRET}\n",
        f"password: !!timestamp {SECRET}\n",
        "password: !!timestamp 2026-99-99\n",
        "password: !!python/object/apply:os.getcwd []\n",
    ],
    ids=["open-quote", "open-list", "bad-binary", "bad-int", "bad-timestamp", "no-such-date",
         "python-object"],
)
def test_a_yaml_problem_does_not_quote_the_text(text):
    with pytest.raises(InvalidText) as raised:
        load_yaml(text)

    assert SECRET not in str(raised.value)
    assert "2026-99-99" not in str(raised.value)


def test_a_yaml_syntax_problem_gives_its_position():
    with pytest.raises(InvalidText) as raised:
        load_yaml("site: [\n")

    assert raised.value.line is not None
    assert f"at line {raised.value.line}" in str(raised.value)


def test_load_json_refuses_a_repeated_key():
    with pytest.raises(InvalidText, match="'site' more than once"):
        load_json('{"site": "04", "date": {}, "site": "05"}')


@pytest.mark.parametrize("number", ["NaN", "Infinity", "-Infinity", "1e400"])
def test_load_json_refuses_a_number_that_is_not_finite(number):
    with pytest.raises(InvalidText):
        load_json(f'{{"score": {number}}}')


@pytest.mark.parametrize(
    "text",
    [f'{{"password": "{SECRET}"', "1" * 5000, "[" * 100000],
    ids=["open-object", "long-integer", "deep"],
)
def test_a_json_problem_does_not_quote_the_text(text):
    with pytest.raises(InvalidText) as raised:
        load_json(text)

    assert SECRET not in str(raised.value)
    assert "1" * 50 not in str(raised.value)


def test_a_json_syntax_problem_gives_its_position():
    with pytest.raises(InvalidText) as raised:
        load_json('{\n  "site": "04"\n  "date": 1}')

    assert (raised.value.line, raised.value.column) == (3, 3)
