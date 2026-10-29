import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vidsense import db
from vidsense import detect
from vidsense.query import parse_query, answer_query


def main():
    sources = db.list_sources()
    ready_sources = [s for s in sources if s.get("status") == "ready"]
    if not ready_sources:
        print("No ready source found. Please run scripts/test_ingest.py first.")
        sys.exit(1)

    source = ready_sources[0]
    source_id = source["source_id"]
    print(f"Testing queries on source: {source_id} ('{source.get('name')}')\n")

    test_questions = [
        "How many people are there?",
        "When does the first car appear?",
        "What is happening in the first 30 seconds?",
        "Is there a person walking?",
        "Is there a dinosaur?",
    ]

    known_classes = detect.class_names()

    for idx, q in enumerate(test_questions, start=1):
        print(f"--- Query {idx}: \"{q}\" ---")
        parsed = parse_query(q, known_classes)
        print(f"parse_query output: {parsed}")

        res = answer_query(q, source_id=source_id)
        print(f"Answer: {res['answer']}")
        print(f"Found: {res['found']}")
        print(f"Results count: {len(res['results'])}")
        print("First 3 results:")
        for r in res["results"][:3]:
            print(f"  [{r['kind']}] {r['label']} (t={r['ts_start']:.2f}s - {r['ts_end']:.2f}s, thumb={r['thumbnail']})")
        print()

    print("ALL QUERY TESTS COMPLETED")


if __name__ == "__main__":
    main()
