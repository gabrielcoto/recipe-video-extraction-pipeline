#!/usr/bin/env python3
# Оценка извлечённого рецепта против ручного эталона по метрике рубрики
# (docs/evaluation.md): оба JSON раскрываются в пары «JSON Pointer → значение»,
# берётся объединение путей, совпадение = пути, присутствующие в обоих
# документах с равными значениями / размер объединения × 100%.
#
# Правила раскрытия из рубрики:
#   - скаляр, null и пустой контейнер — конечные значения;
#   - каждый элемент непустого списка и каждое вложенное поле раскрываются,
#     индексы списков входят в путь;
#   - отсутствующее поле и поле со значением null различаются
#     (если иная нормализация не согласована — см. --absent-equals-null).
#
# Примеры:
#   # Проверить файлы эталонов на соответствие контракту Recipe 1.0.0:
#   python scripts/compare_gold.py --validate-only data/expected_outputs/*.json
#
#   # Оценить один прогон против эталона:
#   python scripts/compare_gold.py data/expected_outputs/pasta.json run_pasta.json
#
#   # Исключить поддерево из оценки и считать null равным отсутствующему полю:
#   python scripts/compare_gold.py --exclude /source --absent-equals-null gold.json result.json
#
# Коды выхода: 0 — порог достигнут (или все файлы валидны);
#              1 — порог не достигнут / есть ошибки валидации;
#              2 — ошибка использования или чтения файлов.
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

DEFAULT_THRESHOLD = 0.8  # рубрика: не менее 80% для каждого видео


def escape_token(token: str) -> str:
    # Экранирование RFC 6901: сначала ~ -> ~0, затем / -> ~1.
    return str(token).replace("~", "~0").replace("/", "~1")


def flatten(value, pointer: str = "", out: dict | None = None) -> dict:
    """Раскрывает JSON до пар «указатель → конечное значение»."""
    if out is None:
        out = {}
    if isinstance(value, dict) and value:
        for key, child in value.items():
            flatten(child, f"{pointer}/{escape_token(key)}", out)
    elif isinstance(value, list) and value:
        for index, child in enumerate(value):
            flatten(child, f"{pointer}/{index}", out)
    else:
        out[pointer] = value  # скаляр, null, [] и {} — конечные значения
    return out


def _excluded(pointer: str, excludes: tuple[str, ...]) -> bool:
    # Точное совпадение или вложенность под исключённым поддеревом.
    return any(pointer == root or pointer.startswith(f"{root}/") for root in excludes)


def _equal(left, right) -> bool:
    # True/1 и False/0 в Python равны — для JSON-значений различаем булевы и числа.
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    return left == right


def compare(gold: dict, result: dict, excludes: tuple[str, ...] = (), absent_equals_null: bool = False) -> dict:
    gold_paths = {p: v for p, v in flatten(gold).items() if not _excluded(p, excludes)}
    result_paths = {p: v for p, v in flatten(result).items() if not _excluded(p, excludes)}
    union = sorted(set(gold_paths) | set(result_paths))
    if not union:
        raise ValueError("Объединение путей пусто — сравнивать нечего")
    matches = 0
    diffs: list[dict] = []
    for pointer in union:
        in_gold, in_result = pointer in gold_paths, pointer in result_paths
        if in_gold and in_result:
            if _equal(gold_paths[pointer], result_paths[pointer]):
                matches += 1
            else:
                diffs.append({"pointer": pointer, "status": "unequal", "gold": gold_paths[pointer], "result": result_paths[pointer]})
        elif in_gold:
            # Отсутствующее поле и null различаются; согласованная нормализация
            # (--absent-equals-null) считает null равным отсутствующему полю.
            if absent_equals_null and gold_paths[pointer] is None:
                matches += 1
            else:
                diffs.append({"pointer": pointer, "status": "missing_in_result", "gold": gold_paths[pointer]})
        else:
            if absent_equals_null and result_paths[pointer] is None:
                matches += 1
            else:
                diffs.append({"pointer": pointer, "status": "extra_in_result", "result": result_paths[pointer]})
    return {
        "matches": matches,
        "total": len(union),
        "ratio": matches / len(union),
        "diffs": diffs,
    }


def _short(value, limit: int = 60) -> str:
    text = json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def load_json(path: Path):
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except OSError as exc:
        raise SystemExit(f"Не удалось прочитать {path}: {exc}")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path} — не корректный JSON: {exc}")


def validate_json_text(text: str) -> str | None:
    """Возвращает сообщение об ошибке контракта или None, если документ валиден."""
    try:
        from fropti_core_contracts import validate_recipe_json
    except ImportError:
        return "пакет fropti_core_contracts не установлен (make install)"
    try:
        validate_recipe_json(text)
    except Exception as exc:  # noqa: BLE001 - любая ошибка валидации = невалидный документ
        return _short(str(exc), 200)
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Оценка извлечённого рецепта против ручного эталона (метрика рубрики) "
        "и проверка эталонов контракту Recipe 1.0.0.",
    )
    parser.add_argument("paths", nargs="+", help="эталон и результат (оценка) или файлы эталонов (--validate-only)")
    parser.add_argument("--validate-only", action="store_true", help="только проверить файлы контракту, без оценки")
    parser.add_argument("--exclude", action="append", default=[], metavar="POINTER",
                        help="исключить поддерево из оценки, например /source (можно повторять)")
    parser.add_argument("--absent-equals-null", action="store_true",
                        help="считать null и отсутствующее поле совпадением (согласованная нормализация)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"порог совпадения, по умолчанию {DEFAULT_THRESHOLD}")
    parser.add_argument("--json", action="store_true", help="машинночитаемый вывод (один JSON-объект)")
    parser.add_argument("--no-validate", action="store_true", help="не проверять документы контракту перед оценкой")
    args = parser.parse_args(argv)

    if args.validate_only:
        failures = 0
        for raw in args.paths:
            path = Path(raw)
            error = validate_json_text(path.read_text(encoding="utf-8")) if path.exists() else "файл не найден"
            if error:
                failures += 1
                print(f"FAIL {path}: {error}")
            else:
                print(f"OK   {path}: соответствует Recipe 1.0.0")
        return 1 if failures else 0

    if len(args.paths) != 2:
        parser.error("для оценки нужны ровно два файла: эталон и результат (или --validate-only со списком файлов)")
    if not 0 < args.threshold <= 1:
        parser.error("--threshold должен быть в интервале (0; 1]")

    gold_path, result_path = (Path(p) for p in args.paths)
    gold_doc, result_doc = load_json(gold_path), load_json(result_path)

    if not args.no_validate:
        errors = []
        for name, doc in (("эталон", gold_doc), ("результат", result_doc)):
            error = validate_json_text(json.dumps(doc, ensure_ascii=False))
            if error:
                errors.append(f"{name}: {error}")
        if errors:
            # Невалидный эталон или результат — оценка бессмысленна, рубрика
            # требует сначала привести документы к контракту.
            for error in errors:
                print(f"FAIL {error}", file=sys.stderr)
            print("Подсказка: приведите документы к Recipe 1.0.0 или запустите с --no-validate.", file=sys.stderr)
            return 1

    try:
        report = compare(gold_doc, result_doc, tuple(args.exclude), args.absent_equals_null)
    except ValueError as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 2

    passed = report["ratio"] >= args.threshold
    # Процент для показа округляется вниз до 0.1%: рубрика запрещает округление
    # вверх, а решение принимается по точной дроби, не по отображаемому числу.
    shown = math.floor(report["ratio"] * 1000) / 10
    if args.json:
        print(json.dumps({
            "gold": str(gold_path), "result": str(result_path),
            "matches": report["matches"], "total": report["total"],
            "percent": shown, "ratio": report["ratio"],
            "threshold": args.threshold, "passed": passed,
            "diffs": report["diffs"],
        }, ensure_ascii=False, indent=2))
    else:
        print(f"Эталон:   {gold_path}")
        print(f"Результат: {result_path}")
        print(f"Пути в объединении: {report['total']}; совпадений: {report['matches']}; расхождений: {len(report['diffs'])}.")
        print(f"Совпадение: {report['matches']}/{report['total']} = {shown}% (порог {args.threshold * 100:g}%) — {'PASS' if passed else 'FAIL'}")
        if report["diffs"]:
            print("Расхождения:")
            for diff in report["diffs"]:
                pointer = diff["pointer"]
                if diff["status"] == "missing_in_result":
                    print(f"  {pointer}  [нет в результате]  эталон={_short(diff['gold'])}")
                elif diff["status"] == "extra_in_result":
                    print(f"  {pointer}  [лишнее в результате]  результат={_short(diff['result'])}")
                else:
                    print(f"  {pointer}  эталон={_short(diff['gold'])}  результат={_short(diff['result'])}")
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:
        # Вывод обрезан получателем (head, less) — это не ошибка оценки.
        # Перенаправляем stdout в devnull, чтобы Python не упал вторично при
        # финальной промывке буферов.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        sys.exit(1)
