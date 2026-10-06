"""Report-ready figures from the saved evaluation results (no searches, no API calls).

    python -m eval.make_figures

Reads eval/results/*.csv / *.json (written by attack_experiment, retrieval_eval, label_rates' inputs)
and saves 300 dpi PNGs to eval/figures/. Captions are in eval/figures/README.md.

  fig1_attack_retrieval.png   poison chunk in top 5, per configuration, external vs insider
  fig2_alpha_sweep.png        same rate for bm25+quality as the g(d) weight alpha goes 0 -> 1
  fig3_crowding_out.png       clean top-5 chunks pushed out by injection, before/after normalisation fix
  fig4_answer_labels.png      hand labels of the LLM answers (endorses / refutes / ...)
  fig5_retrieval_quality.png  P@5, P@10, Hit@5, MRR on the clean corpus + best possible P@k
  fig6_stem_and_model.png     stemming on vs off, and tf-idf vs BM25
"""
import csv
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # write files only, no window
import matplotlib.pyplot as plt

from eval.label_rates import LABELS, RELABEL_PATHS, load, merge_relabel

RESULTS_DIR = os.path.join("eval", "results")
FIG_DIR = os.path.join("eval", "figures")

# One colour per meaning, the same in every figure (validated categorical palette, light mode)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#e34948"
GREY = "#a3a29b"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3de"
ATTACK_COLOURS = {"external": BLUE, "insider": ORANGE}
LABEL_COLOURS = {"endorses": RED, "refutes": BLUE, "mentions_neutrally": YELLOW, "ignores": GREY}
MODEL_COLOURS = {"bm25": BLUE, "tfidf": ORANGE}

CONFIG_NAMES = {  # attack_experiment config -> axis label
    "tfidf_none": "tf-idf\nno defense", "bm25_none": "BM25\nno defense", "bm25_quality": "BM25\n+ quality g(d)",
    "bm25_jaccard": "BM25\n+ Jaccard", "bm25_all": "BM25\n+ all", "tfidf_all": "tf-idf\n+ all"}
RETRIEVAL_NAMES = {  # retrieval_eval config -> legend label
    "bm25_stem": "BM25, stemmed", "bm25_nostem": "BM25, unstemmed", "tfidf_stem": "tf-idf, stemmed",
    "tfidf_nostem": "tf-idf, unstemmed", "bm25_stem_defenses": "BM25, stemmed + all defenses"}
RETRIEVAL_COLOURS = dict(zip(RETRIEVAL_NAMES, (BLUE, ORANGE, AQUA, YELLOW, MAGENTA)))
METRICS = ("P@5", "P@10", "Hit@5", "MRR")

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelsize": 11,
    "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
    "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight", "font.family": "DejaVu Sans"})


def read_csv(name: str) -> list[dict]:
    with open(os.path.join(RESULTS_DIR, name), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save(fig, name: str) -> None:
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print("wrote", path)


def bar_labels(ax, bars, fmt: str = "{:.0f}%", fontsize: float = 9) -> None:
    """Value on top of each bar, in text ink (not the series colour)."""
    for b in bars:
        ax.annotate(fmt.format(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=fontsize, color=INK)


def by_config_type(rows: list[dict]) -> dict:
    out = defaultdict(list)
    for r in rows:
        out[(r["config"], r["attack_type"])].append(r)
    return out


def in_top5_rate(rows: list[dict]) -> float:
    return 100 * sum(r["in_top5"] == "True" for r in rows) / len(rows)


# --- Figure 1 -----------------------------------------------------------------------------------
def fig_attack_retrieval(attack: dict) -> None:
    configs = list(CONFIG_NAMES)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    w = 0.38
    for i, t in enumerate(("external", "insider")):
        vals = [in_top5_rate(attack[(c, t)]) for c in configs]
        n = len(attack[(configs[0], t)])
        bars = ax.bar([x + (i - 0.5) * w for x in range(len(configs))], vals, w - 0.03,
                      color=ATTACK_COLOURS[t], label=f"{t} poison pages (n={n})")
        bar_labels(ax, bars)
    ax.set_xticks(range(len(configs)), [CONFIG_NAMES[c] for c in configs])
    ax.set_ylim(0, 112)
    ax.set_ylabel("Attack success (% of poison pages\nwith a poison chunk in the top 5)")
    ax.set_title("Retrieval-level attack success per configuration")
    ax.legend(loc="upper right", ncol=2, bbox_to_anchor=(1, 1.02))
    save(fig, "fig1_attack_retrieval.png")


# --- Figure 2 -----------------------------------------------------------------------------------
def fig_alpha_sweep(attack: dict) -> None:
    alphas = sorted({float(c.split("_a")[1]) for c, _ in attack if c.startswith("bm25_quality_a")})
    fig, ax = plt.subplots(figsize=(7, 4.5))
    styles = {"insider": dict(lw=4, ls="-", marker="o", ms=11, mfc=ORANGE, mec="white", mew=2),
              "external": dict(lw=2, ls="--", marker="o", ms=7, mfc="white", mec=BLUE, mew=2)}
    for t in ("insider", "external"):  # external drawn last: it overlaps insider at 100% up to alpha 0.5
        vals = [in_top5_rate(attack[(f"bm25_quality_a{a}", t)]) for a in alphas]
        ax.plot(alphas, vals, color=ATTACK_COLOURS[t], label=f"{t} poison pages (n=8)", **styles[t])
        ax.annotate(t, (alphas[-1], vals[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", color=INK, fontsize=10)
    ax.set_xticks(alphas)
    ax.set_xlim(-0.05, 1.18)
    ax.set_ylim(-5, 110)
    ax.set_xlabel("alpha (weight of source quality g(d) in net score = relevance + alpha * g(d))")
    ax.set_ylabel("Attack success (% poison in top 5)")
    ax.set_title("Alpha sweep: BM25 + quality g(d)")
    ax.text(0.25, 92, "both lines at 100% for alpha <= 0.5", ha="center", va="top", fontsize=9, color=INK_2)
    ax.legend(loc="lower left")
    save(fig, "fig2_alpha_sweep.png")


# --- Figure 3 -----------------------------------------------------------------------------------
def fig_crowding_out(before: dict, after: dict) -> None:
    configs = ("bm25_none", "bm25_all")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    w = 0.38
    for ax, (title, data) in zip(axes, (("Before normalisation fix", before), ("After normalisation fix", after))):
        for i, t in enumerate(("external", "insider")):
            vals = [sum(int(r["clean_top5_pushed_out"]) for r in data[(c, t)]) / len(data[(c, t)]) for c in configs]
            bars = ax.bar([x + (i - 0.5) * w for x in range(len(configs))], vals, w - 0.03,
                          color=ATTACK_COLOURS[t], label=f"{t} poison pages")
            bar_labels(ax, bars, "{:.2f}")
        ax.set_xticks(range(len(configs)), [CONFIG_NAMES[c].replace("\n", " ") for c in configs])
        ax.set_title(title, fontsize=12)
    axes[0].set_ylabel("Mean clean top-5 chunks pushed out (0-5)")
    axes[0].set_ylim(0, 3.6)
    axes[1].legend(loc="upper right")
    fig.suptitle("Crowding out: clean chunks displaced from the top 5 by injection", fontweight="bold", fontsize=13)
    save(fig, "fig3_crowding_out.png")


# --- Figure 4 -----------------------------------------------------------------------------------
def final_labels(path: str, relabel_paths: list[str]) -> list[dict]:
    """Labelled answers with both relabel rounds merged (same rules as eval/label_rates.py)."""
    rows, _ = load(path)
    for p in relabel_paths:
        rows, _ = merge_relabel(rows, load(p, keep_unlabelled=True)[0])
    return rows


def fig_answer_labels() -> None:
    sonnet = final_labels(os.path.join(RESULTS_DIR, "answers_to_label.csv"),
                          [p for p in RELABEL_PATHS if os.path.exists(p)])
    haiku = final_labels(os.path.join(RESULTS_DIR, "answers_to_label_haiku.csv"), [])
    groups = [("Sonnet\nno defense", [r for r in sonnet if r["config"] == "bm25_none"]),
              ("Sonnet\nall defenses", [r for r in sonnet if r["config"] == "bm25_all"]),
              ("Haiku\nno defense", [r for r in haiku if r["config"] == "bm25_none"])]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharey=True)
    for ax, t in zip(axes, ("external", "insider")):
        bottoms = [0.0] * len(groups)
        for label in LABELS:
            vals = []
            for name, rows in groups:
                sel = [r for r in rows if r["attack_type"] == t]
                vals.append(100 * sum(r["label"] == label for r in sel) / len(sel))
            ax.bar(range(len(groups)), vals, 0.6, bottom=bottoms, color=LABEL_COLOURS[label],
                   edgecolor="white", linewidth=2, label=label.replace("_", " "))
            for x, (v, b) in enumerate(zip(vals, bottoms)):
                if v >= 8:
                    ax.text(x, b + v / 2, f"{v:.0f}%", ha="center", va="center", fontsize=9,
                            color="white" if label in ("refutes", "endorses") else INK)
            bottoms = [b + v for b, v in zip(bottoms, vals)]
        n = len([r for r in groups[0][1] if r["attack_type"] == t])
        ax.set_xticks(range(len(groups)), [g for g, _ in groups])
        ax.set_title(f"{t.capitalize()} poison pages ({n} answers per bar)", fontsize=12)
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("% of labelled answers")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("How the LLM treated the false claim (hand labels; endorses = attack succeeded)",
                 fontweight="bold", fontsize=13)
    n_end = sum(r["label"] == "endorses" for _, rows in groups for r in rows)
    n_all = sum(len(rows) for _, rows in groups)
    fig.text(0.5, -0.09, f"endorses = 0% in every bar: {n_end} of {n_all} labelled answers endorse the false claim",
             ha="center", fontsize=10, color=INK)
    fig.subplots_adjust(bottom=0.2)
    save(fig, "fig4_answer_labels.png")


# --- Figures 5 and 6 ----------------------------------------------------------------------------
def retrieval_means() -> dict:
    return {r["config"]: {m: float(r[m]) for m in METRICS}
            for r in read_csv("retrieval_results.csv") if r["qid"] == "MEAN"}


def p_ceiling() -> dict:
    """Best possible mean P@k: a perfect ranking puts every relevant chunk first, but a question's
    expected pages may have fewer than k chunks (same computation as eval/retrieval_eval.py)."""
    with open(os.path.join(RESULTS_DIR, "qrels_pages.json"), encoding="utf-8") as f:
        qrels = json.load(f)
    kept = [q for q in qrels.values() if q["relevant_doc_ids"]]  # Q10 skipped: its page isn't crawled
    n_rel = [sum({e["page"]["doc_id"]: e["page"]["n_chunks"] for e in q["expected"] if e["page"]}.values())
             for q in kept]
    return {f"P@{k}": sum(min(k, n) / k for n in n_rel) / len(kept) for k in (5, 10)}, len(kept)


def fig_retrieval_quality(means: dict) -> None:
    ceil, n_q = p_ceiling()
    configs = list(RETRIEVAL_NAMES)
    fig, ax = plt.subplots(figsize=(11, 5))
    w = 0.16
    for i, c in enumerate(configs):
        bars = ax.bar([m + (i - 2) * w for m in range(len(METRICS))], [means[c][m] for m in METRICS], w - 0.02,
                      color=RETRIEVAL_COLOURS[c], label=RETRIEVAL_NAMES[c])
        bar_labels(ax, bars, "{:.2f}")
    for m, metric in enumerate(METRICS):  # ceiling: dashed line across that metric's group
        top = ceil.get(metric, 1.0)
        ax.hlines(top, m - 2.6 * w, m + 2.6 * w, colors=INK, linestyles="--", lw=1.5,
                  label="best possible (perfect ranking)" if m == 0 else None)
        ax.annotate(f"max {top:.2f}", (m + 2.6 * w, top), xytext=(2, 3), textcoords="offset points",
                    fontsize=8.5, color=INK_2, ha="right", va="bottom")
    ax.set_xticks(range(len(METRICS)), METRICS)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel(f"Mean over {n_q} questions")
    ax.set_title(f"Retrieval quality on the clean corpus (page-level relevance, {n_q} questions, Q10 skipped)")
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.08))
    save(fig, "fig5_retrieval_quality.png")


def fig_stem_and_model(means: dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.2), sharey=True, gridspec_kw={"width_ratios": [1.7, 1]})
    # Left: stemming on (solid) vs off (hatched), for each ranking function
    w = 0.2
    order = [("bm25", True), ("bm25", False), ("tfidf", True), ("tfidf", False)]
    for i, (model, stem) in enumerate(order):
        c = f"{model}_{'stem' if stem else 'nostem'}"
        bars = ax1.bar([m + (i - 1.5) * w for m in range(len(METRICS))], [means[c][m] for m in METRICS], w - 0.02,
                       color=MODEL_COLOURS[model] if stem else "white", edgecolor=MODEL_COLOURS[model],
                       hatch=None if stem else "///", linewidth=1.5, label=RETRIEVAL_NAMES[c])
        bar_labels(ax1, bars, "{:.2f}", fontsize=8)
    ax1.set_title("Stemming on (solid) vs off (hatched)", fontsize=12)
    ax1.set_ylabel("Mean over 19 questions")
    ax1.legend(loc="upper left", ncol=2, fontsize=9.5)
    # Right: tf-idf vs BM25, both stemmed (the default text processing)
    w = 0.35
    for i, model in enumerate(("tfidf", "bm25")):
        c = f"{model}_stem"
        bars = ax2.bar([m + (i - 0.5) * w for m in range(len(METRICS))], [means[c][m] for m in METRICS], w - 0.03,
                       color=MODEL_COLOURS[model], label=RETRIEVAL_NAMES[c])
        bar_labels(ax2, bars, "{:.2f}")
    ax2.set_title("tf-idf (lnc.ltc cosine) vs BM25, stemmed", fontsize=12)
    ax2.legend(loc="upper left")
    for ax in (ax1, ax2):
        ax.set_xticks(range(len(METRICS)), METRICS)
        ax.set_ylim(0, 1.15)
    fig.suptitle("Text processing and ranking function on the clean corpus (no defenses)",
                 fontweight="bold", fontsize=13)
    save(fig, "fig6_stem_and_model.png")


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    attack = by_config_type(read_csv("attack_results.csv"))
    before = by_config_type(read_csv("attack_results_before_norm_fix.csv"))
    fig_attack_retrieval(attack)
    fig_alpha_sweep(attack)
    fig_crowding_out(before, attack)
    fig_answer_labels()
    means = retrieval_means()
    fig_retrieval_quality(means)
    fig_stem_and_model(means)


if __name__ == "__main__":
    main()
