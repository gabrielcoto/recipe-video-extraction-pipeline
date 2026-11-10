# Тесты метрики сравнения с эталоном (scripts/compare_gold.py).
# Метрика закреплена рубрикой (docs/evaluation.md): раскрытие в JSON Pointer,
# объединение путей, порог 80% без округления вверх. Ошибка в реализации
# метрики означала бы неверные оценочные числа, поэтому поведение проверяется.
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "compare_gold.py"

_spec = importlib.util.spec_from_file_location("compare_gold", SCRIPT)
compare_gold = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compare_gold)

flatten = compare_gold.flatten
compare = compare_gold.compare


def test_flatten_expands_lists_and_nested_fields():
    doc = {
        "title": "Омлет",
        "ingredients": [{"name": "Яйца", "quantity": {"value": 2, "unit": "pcs"}}],
        "note": None,
    }
    paths = flatten(doc)
    assert paths == {
        "/title": "Омлет",
        "/ingredients/0/name": "Яйца",
        "/ingredients/0/quantity/value": 2,
        "/ingredients/0/quantity/unit": "pcs",
        "/note": None,
    }


def test_flatten_empty_containers_are_terminal():
    # Пустой контейнер — конечное значение по рубрике, не раскрывается.
    assert flatten({"tags": [], "meta": {}}) == {"/tags": [], "/meta": {}}


def test_flatten_escapes_pointer_tokens():
    # RFC 6901: ~ -> ~0, / -> ~1 — иначе пути «a/b» и «a», «b» неразличимы.
    assert flatten({"a/b": 1, "x~y": 2}) == {"/a~1b": 1, "/x~0y": 2}


def test_compare_identical_documents():
    doc = {"title": "Омлет", "servings": 2}
    report = compare(doc, doc)
    assert report["matches"] == report["total"] == 2
    assert report["ratio"] == 1.0
    assert report["diffs"] == []


def test_missing_extra_and_wrong_lower_the_score():
    # Рубрика: пропущенное и лишнее поле увеличивают знаменатель и не
    # увеличивают числитель; неверное значение тоже не даёт совпадения.
    gold = {"title": "Омлет", "servings": 2, "extra": "x"}
    result = {"title": "Омлет", "servings": 3, "another": "y"}
    report = compare(gold, result)
    assert report["total"] == 4  # title, servings, extra, another
    assert report["matches"] == 1  # только title
    statuses = {d["pointer"]: d["status"] for d in report["diffs"]}
    assert statuses == {
        "/servings": "unequal",
        "/extra": "missing_in_result",
        "/another": "extra_in_result",
    }


def test_null_and_missing_are_distinguished_by_default():
    # Рубрика: отсутствующее поле и поле со значением null различаются.
    gold = {"title": "Омлет", "note": None}
    result = {"title": "Омлет"}
    report = compare(gold, result)
    assert report["matches"] == 1 and report["total"] == 2
    assert report["diffs"][0]["status"] == "missing_in_result"
    # Согласованная нормализация: null и отсутствие считаются совпадением.
    relaxed = compare(gold, result, absent_equals_null=True)
    assert relaxed["matches"] == 2 and relaxed["diffs"] == []


def test_exclude_removes_subtree():
    gold = {"title": "А", "source": {"type": "url", "url": "https://a"}}
    result = {"title": "А", "source": {"type": "upload"}}
    strict = compare(gold, result)
    # Объединение: /title, /source/type, /source/url; совпадает только /title.
    assert strict["matches"] == 1 and strict["total"] == 3
    without_source = compare(gold, result, excludes=("/source",))
    assert without_source["matches"] == without_source["total"] == 1


def _real_gold(name: str) -> Path:
    return ROOT / "data" / "expected_outputs" / f"{name}.json"


def test_cli_validate_only_accepts_contract_examples():
    result = compare_gold.main([
        "--validate-only",
        str(_real_gold("pasta")), str(_real_gold("omelette")), str(_real_gold("salad")),
    ])
    assert result == 0


def test_cli_validate_only_rejects_invalid_file(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema_version": "9.9.9"}), encoding="utf-8")
    assert compare_gold.main(["--validate-only", str(bad)]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_cli_compare_self_scores_full_match():
    # Эталон против самого себя: 100%, код 0.
    assert compare_gold.main([str(_real_gold("pasta")), str(_real_gold("pasta"))]) == 0


def test_cli_compare_different_recipes_fails_threshold(capsys):
    # Разные рецепты дают низкий процент — код 1 и отчёт с расхождениями.
    assert compare_gold.main([str(_real_gold("pasta")), str(_real_gold("omelette"))]) == 1
    output = capsys.readouterr().out
    assert "FAIL" in output and "Расхождения:" in output


def test_cli_compare_requires_two_files():
    with pytest.raises(SystemExit) as excinfo:
        compare_gold.main([str(_real_gold("pasta"))])
    assert excinfo.value.code == 2


def test_cli_excludes_source_subtree():
    # /source ставит сервер из фактического входа; если нормализация согласована,
    # поддерево исключается и не влияет на процент.
    gold, result = _real_gold("pasta"), _real_gold("pasta")
    assert compare_gold.main(["--exclude", "/source", str(gold), str(result)]) == 0
