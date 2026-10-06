"""Export LLM answers from the attack experiment for hand labelling.

    python -m eval.export_labels            # writes eval/results/answers_to_label.csv
    python -m eval.export_labels --force    # overwrite an existing file (this ERASES labels!)

    # only the rows of a labelled file whose answer text is no longer the current answer
    python -m eval.export_labels --stale eval/results/answers_to_label.csv --out eval/results/answers_to_relabel.csv
    # same, judging already-relabelled rows by their relabelled text (round 2)
    python -m eval.export_labels --stale eval/results/answers_to_label.csv eval/results/answers_to_relabel.csv         --out eval/results/answers_to_relabel_2.csv

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


def stale_keys(label_paths: list[str], answers: list[dict]) -> set:
    """(poison_id, config, sample) of labelled rows whose answer text differs from `answers`.

    Several files are read in order and a later file's row replaces an earlier one with the same key
    (main file, then relabel files), so a row that was already relabelled is judged by its new text.
    """
    current = {(a["poison_id"], a["config"], str(a["sample"])): _norm(a["answer"]) for a in answers}
    labelled = {}
    for path in label_paths:
        with open(path, encoding="utf-8-sig", newline="") as f:
            labelled.update({(r["poison_id"], r["config"], r["sample"]): _norm(r["answer"]) for r in csv.DictReader(f)})
    return {k for k, text in labelled.items() if current.get(k) != text}


def _norm(text: str) -> str:
    """Spreadsheet saves turn the answers' line breaks into \\r\\n; that is not a different answer."""
    return text.replace("\r\n", "\n").strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export answers for hand labelling.")
    parser.add_argument("--answers", default=ANSWERS_PATH, help="answers .jsonl to export from")
    parser.add_argument("--configs", default=",".join(CONFIGS), help="comma-separated configs to export")
    parser.add_argument("--out", default=LABEL_PATH)
    parser.add_argument("--stale", metavar="LABELLED_CSV", nargs="+",
                        help="only labelled rows whose answer text is out of date; several files are "
                             "merged in order (main file first, then relabel files)")
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
