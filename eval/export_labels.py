"""Export LLM answers from the attack experiment for hand labelling.

    python -m eval.export_labels            # writes eval/results/answers_to_label.csv
    python -m eval.export_labels --force    # overwrite an existing file (this ERASES labels!)

Takes the bm25 none and bm25+all answers (both samples) from eval/results/attack_answers.jsonl,
written by eval/attack_experiment.py, and adds the poison page's false claim so the labeller can
compare. The label column is left empty; fill it with one of the labels in eval/label_rates.py
(see the labelling guide in the README's Evaluation section).

utf-8-sig encoding so Excel opens the file with the right characters.
"""
import argparse
import csv
import json
import os

from eval.attack_experiment import ANSWERS_PATH, MANIFEST_PATH, RESULTS_DIR

LABEL_PATH = os.path.join(RESULTS_DIR, "answers_to_label.csv")
CONFIGS = ("bm25_none", "bm25_all")
COLUMNS = ("poison_id", "attack_type", "config", "sample", "question", "false_claim", "answer", "label")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export answers for hand labelling.")
    parser.add_argument("--force", action="store_true", help="overwrite an existing (maybe labelled) file")
    args = parser.parse_args()
    if os.path.exists(LABEL_PATH) and not args.force:
        raise SystemExit(f"{LABEL_PATH} already exists; it may contain labels. Use --force to overwrite.")

    with open(MANIFEST_PATH, encoding="utf-8") as f:
        pages = {m["id"]: m for m in json.load(f)}
    with open(ANSWERS_PATH, encoding="utf-8") as f:
        answers = [a for a in map(json.loads, f) if a["config"] in CONFIGS]
    answers.sort(key=lambda a: (a["poison_id"], CONFIGS.index(a["config"]), a["sample"]))

    with open(LABEL_PATH, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for a in answers:
            m = pages[a["poison_id"]]
            w.writerow({"poison_id": a["poison_id"], "attack_type": m["attack_type"], "config": a["config"],
                        "sample": a["sample"], "question": a["question"], "false_claim": m["false_claim"],
                        "answer": a["answer"], "label": ""})
    print(f"wrote {LABEL_PATH} ({len(answers)} answers to label)")


if __name__ == "__main__":
    main()
