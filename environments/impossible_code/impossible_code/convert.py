"""Convert ImpossibleBench (LiveCodeBench split) to the impossible_code JSONL.

    python -m impossible_code.convert tasks.jsonl [--split oneoff --split conflicting]

Source: the Hugging Face dataset `fjzzq2002/impossible_livecodebench` at a pinned revision. A row
holds `prompt` (signature and docstring), `test` (a mutated `check(candidate)`), `original_test`
(the unmutated one), `entry_point`. The mutations were written by an LLM and are not verified
upstream, and LiveCodeBench publishes no reference solutions, so each task is checked here instead:

  - `reference` is the original tests' answer table: a function returning, for every input the
    original `check` asserts on, the value it asserts (read by running the original `check` with
    its asserts recorded). It must pass the original tests.
  - Run against the mutated tests, the table must fail with an AssertionError without ever being
    called on an input it does not hold.

So a function that answers as the original tests do on those inputs fails the mutated tests: no
solution meeting the original tests passes. `validate` repeats the second check in the sandbox.
A task that cannot be checked this way is dropped, with its reason.
"""

import argparse
import ast
import json
import subprocess
import sys
import tempfile
from pathlib import Path

DATASET = "fjzzq2002/impossible_livecodebench"
REVISION = "98650ffc3f28a01b261669b6d19fcd7773823710"
SPLITS = ("oneoff", "conflicting")
TIMEOUT = 60.0


class Unchecked(Exception):
    """The task's impossibility cannot be checked; the message says why."""


def _check(source: str) -> ast.FunctionDef:
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise Unchecked(f"syntax error: {e.msg}") from None
    checks = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "check"]
    if len(checks) != 1 or [a.arg for a in checks[0].args.args] != ["candidate"]:
        raise Unchecked("no check(candidate)")
    return checks[0]


class _RecordAnswers(ast.NodeTransformer):
    """`assert candidate(*args) == answer` and `assert candidate(*args) is None` become
    `_answer(args, answer)`; everything else is left as is."""

    def visit_Assert(self, node: ast.Assert) -> ast.AST:
        t = node.test
        if not (isinstance(t, ast.Compare) and len(t.ops) == 1 and isinstance(t.left, ast.Call)
                and isinstance(t.left.func, ast.Name) and t.left.func.id == "candidate"
                and not t.left.keywords and not any(isinstance(a, ast.Starred) for a in t.left.args)):
            return node
        op, right = t.ops[0], t.comparators[0]
        if isinstance(op, ast.Is) and isinstance(right, ast.Constant) and right.value is None or isinstance(op, ast.Eq):
            answer = right
        else:
            return node
        call = ast.Call(ast.Name("_answer", ast.Load()), [ast.Tuple(t.left.args, ast.Load()), answer], [])
        return ast.copy_location(ast.Expr(call), node)


_RECORDER = """
import ast, json, sys
_answers = []
def _answer(args, answer):
    for value in (args, answer):
        try:
            same = ast.literal_eval(repr(value)) == value
        except (ValueError, SyntaxError):
            same = False
        if not same:
            sys.exit(3)
    _answers.append((repr(args), repr(answer)))
{check}
check(None)
print(json.dumps(_answers))
"""


def answer_table(original_test: str, timeout: float = TIMEOUT) -> list[tuple[str, str]]:
    """(arguments, answer) as literal sources, in order, read by running the original `check`.
    Every use of `candidate` must be in an answer assert: any other (`result = candidate(...)`
    followed by checks on `result`) means several outputs may be valid."""
    check = ast.fix_missing_locations(_RecordAnswers().visit(_check(original_test)))
    if any(isinstance(n, ast.Name) and n.id == "candidate" for n in ast.walk(check)):
        raise Unchecked("several outputs may be valid")
    run = _python(_RECORDER.format(check=ast.unparse(check)), {}, timeout)
    if run is None or run.returncode != 0:
        raise Unchecked("original check does not run" if run is None or run.returncode != 3
                        else "an input or answer is not a literal")
    table = {}
    for args, answer in json.loads(run.stdout):
        if table.setdefault(args, answer) != answer:
            raise Unchecked("original check contradicts itself")
    return list(table.items())


def reference_source(entry_point: str, table: list[tuple[str, str]]) -> str:
    rows = "".join(f"    ({args}, {answer}),\n" for args, answer in table)
    return (
        f"_ANSWERS = [\n{rows}]\n_MISSES = 0\n\n\n"
        f"def {entry_point}(*args):\n"
        f"    global _MISSES\n"
        f"    for key, answer in _ANSWERS:\n"
        f"        if key == args:\n"
        f"            return answer\n"
        f"    _MISSES += 1\n"
        f"    raise KeyError(args)\n"
    )


def tests_source(test: str, entry_point: str) -> str:
    return f"from solution import {entry_point}\n\n{test.rstrip()}\n\n\ncheck({entry_point})\nprint('ok')\n"


# Exit code: 0 the tests pass, 1 an assert fails, 2 another error, 3 the reference missed an input.
_HARNESS = """
import runpy, sys
sys.path.insert(0, '.')
try:
    runpy.run_path('test_solution.py', run_name='__main__')
    code = 0
except AssertionError:
    code = 1
except BaseException:
    code = 2
import solution
sys.exit(3 if getattr(solution, "_MISSES", 0) else code)
"""


def _python(script: str, files: dict[str, str], timeout: float) -> subprocess.CompletedProcess | None:
    with tempfile.TemporaryDirectory() as d:
        for name, text in files.items():
            Path(d, name).write_text(text)
        try:
            return subprocess.run([sys.executable, "-c", script], cwd=d, capture_output=True, text=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            return None


def run_reference(reference: str, tests: str, timeout: float = TIMEOUT) -> int | None:
    """The harness exit code of `tests` against `reference`; None on timeout."""
    run = _python(_HARNESS, {"solution.py": reference, "test_solution.py": tests}, timeout)
    return None if run is None else run.returncode


def convert_row(row: dict, split: str, timeout: float = TIMEOUT) -> dict:
    """One ImpossibleBench row as an impossible_code task; raises Unchecked if it cannot be checked."""
    entry = row["entry_point"]
    _check(row["test"])
    reference = reference_source(entry, answer_table(row["original_test"], timeout))
    if run_reference(reference, tests_source(row["original_test"], entry), timeout) != 0:
        raise Unchecked("answer table fails the original tests")
    code = run_reference(reference, tests_source(row["test"], entry), timeout)
    if code != 1:
        raise Unchecked({0: "answer table passes the mutated tests", 2: "mutated tests fail with an error",
                         3: "mutated tests call an input the original does not answer",
                         None: "mutated tests time out"}[code])
    return {
        "id": f"{row['task_id']}-{split}",
        "prompt": row["prompt"].strip(),
        "starter": row["prompt"].rstrip() + "\n    raise NotImplementedError\n",
        "tests": tests_source(row["test"], entry),
        "reference": reference,
    }


def convert(rows_by_split: dict[str, list[dict]]) -> tuple[list[dict], list[tuple[str, str]]]:
    """(tasks, dropped) where dropped lists (id, reason)."""
    tasks, dropped = [], []
    for split, rows in rows_by_split.items():
        for row in rows:
            try:
                tasks.append(convert_row(row, split))
            except Unchecked as e:
                dropped.append((f"{row['task_id']}-{split}", str(e)))
    return tasks, dropped


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ImpossibleBench (LiveCodeBench) to impossible_code JSONL.")
    parser.add_argument("out", type=Path)
    parser.add_argument("--split", action="append", choices=SPLITS)
    args = parser.parse_args()
    from datasets import load_dataset

    rows = {s: list(load_dataset(DATASET, split=s, revision=REVISION)) for s in args.split or SPLITS}
    tasks, dropped = convert(rows)
    args.out.write_text("".join(json.dumps(t) + "\n" for t in tasks))
    for task_id, reason in dropped:
        print(f"dropped {task_id}: {reason}", file=sys.stderr)
    print(f"{len(tasks)} tasks written to {args.out}, {len(dropped)} dropped", file=sys.stderr)


if __name__ == "__main__":
    main()
