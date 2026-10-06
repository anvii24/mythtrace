"""Export LLM answers from the attack experiment for hand labelling.

    python -m eval.export_labels            # writes eval/results/answers_to_label.csv
    python -m eval.export_labels --force    # overwrite an existing file (this ERASES labels!)

    # only the rows of a labelled file whose answer text is no longer the current answer
    python -m eval.export_labels --stale eval/results/answers_to_label.csv --out eval/results/answers_to_relabel.csv

    # another answers file (e.g. the Haiku run from eval/model_compare.py)
    python -m eval.export_labels --answers eval/results/attack_answers_haiku.jsonl \
        --configs bm25_none --out eval/results/answers_to_label_haiku.csv

Takes the chosen configs' answers (both samples) from an answers .jsonl written by
eval/attack_experiment.py or eval/model_compare.py, and adds the poison page's false claim so the
labeller can compare. The label column is left empty; fill it with one of the labels in
eval/label_rates.py (see the labelling guide in the README's Evaluation section).

utf-8-sig encoding so Excel opens the file with the right characters. When saving from Excel,
use "CSV UTF-8 (Comma delimited)", not the default .xlsx.
"""
import argparse
import csv
import json
import os

from eval.attack_experiment import ANSWERS_PATH, MANIFEST_PATH, RESULTS_DIR

LABEL_PATH = os.path.join(RESULTS_DIR, "answers_to_label.csv")
CONFIGS = ("bm25_none", "bm25_all")
COLUMNS = ("poison_id", "attack_type", "config", "sample", "question", "false_claim", "answer", "label")


def stale_keys(label_path: str, answers: list[dict]) -> set:
    """(poison_id, config, sample) of rows in a labelled file whose answer text differs from `answers`."""
    current = {(a["poison_id"], a["config"], str(a["sample"])): a["answer"] for a in answers}
    with open(label_path, encoding="utf-8-sig", newline="") as f:
        return {(r["poison_id"], r["config"], r["sample"]) for r in csv.DictReader(f)
                if current.get((r["poison_id"], r["config"], r["sample"])) != r["answer"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export answers for hand labelling.")
    parser.add_argument("--answers", default=ANSWERS_PATH, help="answers .jsonl to export from")
    parser.add_argument("--configs", default=",".join(CONFIGS), help="comma-separated configs to export")
    parser.add_argument("--out", default=LABEL_PATH)
    parser.add_argument("--stale", metavar="LABELLED_CSV",
                        help="only rows of this labelled file whose answer text is out of date")
    parser.add_argument("--force", action="store_true", help="overwrite an existing (maybe labelled) file")
    args = parser.parse_args()
    if os.path.exists(args.out) and not args.force:
        raise SystemExit(f"{args.out} already exists; it may contain labels. Use --force to overwrite.")
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]

    with open(MANIFEST_PATH, encoding="utf-8") as f:
        pages = {m["id"]: m for m in json.load(f)}
    with open(args.answers, encoding="utf-8") as f:
        answers = [a for a in map(json.loads, f) if a["config"] in configs]
    if args.stale:
        keep = stale_keys(args.stale, answers)
        answers = [a for a in answers if (a["poison_id"], a["config"], str(a["sample"])) in keep]
    answers.sort(key=lambda a: (a["poison_id"], configs.index(a["config"]), a["sample"]))

    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for a in answers:
            m = pages[a["poison_id"]]
            w.writerow({"poison_id": a["poison_id"], "attack_type": m["attack_type"], "config": a["config"],
                        "sample": a["sample"], "question": a["question"], "false_claim": m["false_claim"],
                        "answer": a["answer"], "label": ""})
    print(f"wrote {args.out} ({len(answers)} answers to label)")


if __name__ == "__main__":
    main()
