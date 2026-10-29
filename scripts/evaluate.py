"""Evaluation script for VidSense AI query performance.

Reads test queries from tests/eval_questions.json, executes query.answer_query()
for non-TODO entries, verifies correctness against specified expectations,
and displays a structured results table with accuracy and mean latency metrics.
"""

import json
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from vidsense import db, query


def load_eval_questions(path: str) -> list[dict]:
    if not os.path.exists(path):
        print(f"Error: Evaluation file not found at '{path}'", file=sys.stderr)
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        print(f"Error: Expected JSON list in '{path}', got {type(data).__name__}", file=sys.stderr)
        sys.exit(1)
    return data


def is_todo_entry(entry: dict) -> bool:
    """Returns True if the entry contains 'TODO' in any string field."""
    serialized = json.dumps(entry)
    return "TODO" in serialized


def check_expectation(ans: dict, expect: dict) -> tuple[bool, str]:
    exp_type = expect.get("type")
    found = ans.get("found", False)
    results = ans.get("results", [])

    if exp_type == "count":
        expected_val = expect.get("value", 0)
        tolerance = expect.get("tolerance", 0)
        actual_val = len(results)
        passed = abs(actual_val - expected_val) <= tolerance
        detail = f"count={actual_val} (expected {expected_val} +/- {tolerance})"
        return passed, detail

    elif exp_type == "found_at":
        ts_start = float(expect.get("ts_start", 0.0))
        ts_end = float(expect.get("ts_end", 0.0))
        if not found or not results:
            return False, f"found=False (expected interval [{ts_start:.1f}, {ts_end:.1f}])"

        overlapping = []
        for r in results:
            r_start = float(r.get("ts_start", 0.0))
            r_end = float(r.get("ts_end", 0.0))
            # Two intervals [A, B] and [C, D] overlap if max(A, C) <= min(B, D)
            if max(r_start, ts_start) <= min(r_end, ts_end):
                overlapping.append(r)

        passed = len(overlapping) > 0
        if passed:
            detail = f"overlapping={len(overlapping)} interval(s) in [{ts_start:.1f}, {ts_end:.1f}]"
        else:
            intervals_str = ", ".join(f"[{r.get('ts_start', 0):.1f}, {r.get('ts_end', 0):.1f}]" for r in results[:3])
            detail = f"no overlap with [{ts_start:.1f}, {ts_end:.1f}] (got: {intervals_str})"
        return passed, detail

    elif exp_type == "not_found":
        passed = (found is False)
        detail = f"found={found} (expected False)"
        return passed, detail

    else:
        return False, f"unknown expect type '{exp_type}'"


def run_evaluation(questions_path: str) -> None:
    db.init_db()
    questions = load_eval_questions(questions_path)

    sources = db.list_sources()
    source_map = {s["name"]: s["source_id"] for s in sources}

    print(f"Loaded {len(questions)} evaluation question(s) from {questions_path}")
    print(f"Indexed database sources available: {len(sources)}")
    print("=" * 90)

    evaluated_records = []
    latencies = []
    passed_count = 0
    failed_count = 0
    skipped_count = 0

    for idx, entry in enumerate(questions, start=1):
        q_text = entry.get("question", "")
        s_name = entry.get("source_name", "")
        expect = entry.get("expect", {})
        exp_type = expect.get("type", "unknown")

        if is_todo_entry(entry):
            skipped_count += 1
            evaluated_records.append({
                "id": idx,
                "question": q_text,
                "source": s_name,
                "type": exp_type,
                "status": "SKIPPED",
                "latency_ms": "-",
                "details": "Contains TODO placeholder",
            })
            continue

        source_id = source_map.get(s_name)
        if not source_id:
            failed_count += 1
            evaluated_records.append({
                "id": idx,
                "question": q_text,
                "source": s_name,
                "type": exp_type,
                "status": "FAIL",
                "latency_ms": "-",
                "details": f"Source '{s_name}' not found in database",
            })
            continue

        t0 = time.perf_counter()
        try:
            ans = query.answer_query(q_text, source_id)
            latency = (time.perf_counter() - t0) * 1000.0
            latencies.append(latency)

            passed, details = check_expectation(ans, expect)
            if passed:
                passed_count += 1
                status = "PASS"
            else:
                failed_count += 1
                status = "FAIL"

            evaluated_records.append({
                "id": idx,
                "question": q_text,
                "source": s_name,
                "type": exp_type,
                "status": status,
                "latency_ms": f"{latency:.1f}",
                "details": details,
            })
        except Exception as e:
            failed_count += 1
            evaluated_records.append({
                "id": idx,
                "question": q_text,
                "source": s_name,
                "type": exp_type,
                "status": "FAIL",
                "latency_ms": "-",
                "details": f"Error: {e}",
            })

    # Print results table
    header_fmt = "{:<4} | {:<32} | {:<18} | {:<9} | {:<7} | {:<9} | {:<25}"
    row_fmt    = "{:<4} | {:<32} | {:<18} | {:<9} | {:<7} | {:<9} | {:<25}"

    print(header_fmt.format("#", "Question", "Source", "Type", "Status", "Latency", "Details"))
    print("-" * 115)
    for r in evaluated_records:
        q_short = (r["question"][:29] + "...") if len(r["question"]) > 32 else r["question"]
        s_short = (r["source"][:15] + "...") if len(r["source"]) > 18 else r["source"]
        det_short = (r["details"][:22] + "...") if len(r["details"]) > 25 else r["details"]
        lat_str = f"{r['latency_ms']} ms" if r["latency_ms"] != "-" else "-"
        print(row_fmt.format(
            r["id"],
            q_short,
            s_short,
            r["type"],
            r["status"],
            lat_str,
            det_short,
        ))
    print("-" * 115)

    total_evaluated = passed_count + failed_count
    if total_evaluated > 0:
        accuracy = (passed_count / total_evaluated) * 100.0
        mean_latency = sum(latencies) / len(latencies) if latencies else 0.0
        acc_str = f"{accuracy:.1f}% ({passed_count}/{total_evaluated})"
        lat_str = f"{mean_latency:.1f} ms"
    else:
        acc_str = "N/A (0 non-skipped entries)"
        lat_str = "N/A"

    print("\n=== Evaluation Summary ===")
    print(f"Total entries:       {len(questions)}")
    print(f"Skipped (TODO):      {skipped_count}")
    print(f"Evaluated:           {total_evaluated}")
    print(f"Passed:              {passed_count}")
    print(f"Failed:              {failed_count}")
    print(f"Accuracy:            {acc_str}")
    print(f"Mean Latency:        {lat_str}")
    print("==========================\n")


if __name__ == "__main__":
    eval_file = sys.argv[1] if len(sys.argv) > 1 else str(ROOT_DIR / "tests" / "eval_questions.json")
    run_evaluation(eval_file)
