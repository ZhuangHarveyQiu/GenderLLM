
from __future__ import annotations

import argparse
import itertools
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


CATEGORIES = ["RB", "LEB", "M", "NB"]
BIAS_CATEGORIES = ["RB", "LEB", "M"]
BIAS_SET = set(BIAS_CATEGORIES)
SEED = 2026
N_ITEM_BOOTSTRAPS = 5_000
N_PANEL_DRAWS = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path.cwd(),
        help="Folder containing RawHumanData and RawLLMData, or input.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output folder. Defaults to PROJECT_DIR/DataAnalysis/output_v5.",
    )
    return parser.parse_args()


def resolve_inputs(project_dir: Path) -> tuple[Path, Path, Path | None]:
    candidates = [
        (
            project_dir / "RawHumanData" / "HumanDataRaw.csv",
            project_dir / "RawLLMData" / "parsed_annotations.csv",
            project_dir / "RawHumanData" / "HumanAnnotation.csv",
        ),
        (
            project_dir / "input" / "HumanDataRaw.csv",
            project_dir / "input" / "parsed_annotations.csv",
            project_dir / "input" / "HumanAnnotation.csv",
        ),
    ]
    for human_path, model_path, aggregate_path in candidates:
        if human_path.exists() and model_path.exists():
            return human_path, model_path, aggregate_path if aggregate_path.exists() else None
    expected = "\n".join(f"- {h}\n- {m}" for h, m, _ in candidates)
    raise FileNotFoundError(f"Could not locate the required input files. Checked:\n{expected}")


def is_contradictory(labels: frozenset[str] | set[str] | None) -> bool:
    return labels is not None and "NB" in labels and bool(set(labels) & BIAS_SET)


def load_human_raw(path: Path):
    raw = pd.read_csv(path, encoding="utf-8-sig")
    if raw.shape[1] != 31:
        raise ValueError(f"Expected one sentence column and 30 annotator columns; found {raw.shape[1]}")
    raw.columns = ["sentence"] + [f"A{i}" for i in range(1, 31)]
    raw["number"] = (
        raw.sentence.astype(str)
        .str.extract(r"^\s*(\d+)\s*[.\uff0e\u3002]")[0]
        .astype(int)
    )
    article_labels = []
    article_number = 0
    for sentence_number in raw.number:
        if sentence_number == 1:
            article_number += 1
        article_labels.append(f"P{article_number}")
    raw["article"] = article_labels
    raw["sentence_key"] = raw.article + "_" + raw.number.astype(str)

    if raw.sentence_key.nunique() != 57:
        raise ValueError("Expected 57 unique sentence keys")
    if list(Counter(raw.article).values()) != [20, 19, 18]:
        raise ValueError("Expected article lengths of 20, 19, and 18 sentences")

    cells: dict[tuple[str, int], frozenset[str]] = {}
    for _, row in raw.iterrows():
        for annotator in range(1, 31):
            labels = frozenset(
                token
                for token in re.split(r"[^A-Za-z]+", str(row[f"A{annotator}"]))
                if token in CATEGORIES
            )
            cells[(row.sentence_key, annotator)] = labels
    return raw, cells, list(raw.sentence_key), list(range(1, 31))


def clean_human(cells):
    cleaned = {}
    deleted = []
    for key, labels in cells.items():
        if is_contradictory(labels):
            cleaned[key] = None
            deleted.append(key)
        else:
            cleaned[key] = labels
    return cleaned, deleted


def load_models(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {"model_key", "rep", "article", "sentence_id", "labels"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Model annotation file is missing columns: {sorted(missing)}")
    data["sentence_key"] = data.article.astype(str) + "_" + data.sentence_id.astype(str)
    return data


def build_repetition_sets(data: pd.DataFrame):
    repetition_sets = {}
    source_row_counts = {}
    keys = ["model_key", "rep", "sentence_key"]
    for key, group in data.groupby(keys, sort=True):
        labels = set()
        for value in group.labels:
            labels |= {x for x in str(value).split("+") if x in CATEGORIES}
        repetition_sets[(key[0], int(key[1]), key[2])] = frozenset(labels)
        source_row_counts[(key[0], int(key[1]), key[2])] = len(group)
    return repetition_sets, source_row_counts


def consolidate_models(data: pd.DataFrame, sentence_order: list[str]):
    repetitions, source_row_counts = build_repetition_sets(data)
    models = sorted(data.model_key.unique())
    consolidated = {}
    log = {
        "deleted_repetitions": [],
        "missing_cells": [],
        "survivor_counts": Counter(),
        "single_survivor_cells": [],
        "synthesised_cells": [],
        "duplicate_source_cells": [k for k, n in source_row_counts.items() if n > 1],
    }

    for model in models:
        for sentence in sentence_order:
            triplet = [repetitions[(model, rep, sentence)] for rep in (1, 2, 3)]
            for rep, labels in zip((1, 2, 3), triplet):
                if is_contradictory(labels):
                    log["deleted_repetitions"].append((model, rep, sentence, sorted(labels)))
            survivors = [labels for labels in triplet if not is_contradictory(labels)]
            log["survivor_counts"][len(survivors)] += 1

            if not survivors:
                result = None
                log["missing_cells"].append((model, sentence, "no surviving repetition"))
            elif len(survivors) == 1:
                result = survivors[0]
                log["single_survivor_cells"].append((model, sentence))
            else:
                counts = Counter(label for labels in survivors for label in labels)
                result = frozenset(label for label, count in counts.items() if count >= 2)
                if not result:
                    result = None
                    log["missing_cells"].append((model, sentence, "no label selected twice"))

            if result is not None and result not in survivors:
                log["synthesised_cells"].append((model, sentence, sorted(result)))
            consolidated[(model, sentence)] = result

    return consolidated, models, repetitions, log


def selected_counts_human(cells, order, annotators, category=None):
    selected, valid = [], []
    for sentence in order:
        values = [cells[(sentence, a)] for a in annotators if cells[(sentence, a)] is not None]
        valid.append(len(values))
        selected.append(
            sum(1 for labels in values if category in labels)
            if category
            else sum(1 for labels in values if bool(labels - {"NB"}))
        )
    return np.asarray(selected), np.asarray(valid)


def selected_counts_models(consolidated, models, order, category=None):
    selected, valid = [], []
    for sentence in order:
        values = [
            consolidated[(model, sentence)]
            for model in models
            if consolidated[(model, sentence)] is not None
        ]
        valid.append(len(values))
        selected.append(
            sum(1 for labels in values if category in labels)
            if category
            else sum(1 for labels in values if labels != frozenset({"NB"}))
        )
    return np.asarray(selected), np.asarray(valid)


def selected_counts_repetition(repetitions, models, order, repetition, category=None):
    selected, valid = [], []
    for sentence in order:
        values = [
            repetitions[(model, repetition, sentence)]
            for model in models
            if not is_contradictory(repetitions[(model, repetition, sentence)])
        ]
        valid.append(len(values))
        selected.append(
            sum(1 for labels in values if category in labels)
            if category
            else sum(1 for labels in values if labels != frozenset({"NB"}))
        )
    return np.asarray(selected), np.asarray(valid)


def bootstrap_mean_interval(values, draws=N_ITEM_BOOTSTRAPS, seed=SEED):
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def bootstrap_difference_interval(x, y, draws=N_ITEM_BOOTSTRAPS, seed=SEED):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) != len(y):
        raise ValueError("Paired bootstrap inputs must have equal length")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(x), size=(draws, len(x)))
    differences = (x[indices] - y[indices]).mean(axis=1)
    return np.percentile(differences, [2.5, 97.5])


def fleiss_kappa(selected, n_valid):
    selected = np.asarray(selected, float)
    n_valid = np.asarray(n_valid, float)
    keep = n_valid >= 2
    selected, n_valid = selected[keep], n_valid[keep]
    if len(selected) == 0:
        return np.nan
    item_agreement = (
        (selected**2 + (n_valid - selected) ** 2) - n_valid
    ) / (n_valid * (n_valid - 1))
    prevalence = selected.sum() / n_valid.sum()
    expected = prevalence**2 + (1 - prevalence) ** 2
    return (item_agreement.mean() - expected) / (1 - expected) if expected < 1 else np.nan


def observed_pairwise_agreement(selected, n_valid):
    selected = np.asarray(selected, float)
    n_valid = np.asarray(n_valid, float)
    keep = n_valid >= 2
    selected, n_valid = selected[keep], n_valid[keep]
    agreeing_pairs = (
        selected * (selected - 1) / 2
        + (n_valid - selected) * (n_valid - selected - 1) / 2
    )
    all_pairs = n_valid * (n_valid - 1) / 2
    return float((agreeing_pairs / all_pairs).mean())


def krippendorff_alpha_nominal(selected, n_valid):
    selected = np.asarray(selected, float)
    n_valid = np.asarray(n_valid, float)
    keep = n_valid >= 2
    selected, n_valid = selected[keep], n_valid[keep]
    total_ratings = n_valid.sum()
    total_selected = selected.sum()
    observed_disagreement = (
        2 * selected * (n_valid - selected) / (n_valid - 1)
    ).sum() / total_ratings
    expected_disagreement = (
        2 * total_selected * (total_ratings - total_selected)
        / (total_ratings * (total_ratings - 1))
    )
    return 1 - observed_disagreement / expected_disagreement if expected_disagreement > 0 else np.nan


def bootstrap_kappa_interval(selected, n_valid, draws=N_ITEM_BOOTSTRAPS, seed=SEED):
    selected, n_valid = np.asarray(selected), np.asarray(n_valid)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(draws):
        indices = rng.integers(0, len(selected), len(selected))
        estimate = fleiss_kappa(selected[indices], n_valid[indices])
        if np.isfinite(estimate):
            estimates.append(estimate)
    return np.percentile(estimates, [2.5, 97.5])


def concordance_probability(human, model):
    concordant = discordant = model_ties = 0
    for i, j in itertools.combinations(range(len(human)), 2):
        human_difference = human[i] - human[j]
        if human_difference == 0:
            continue
        model_difference = model[i] - model[j]
        if model_difference == 0:
            model_ties += 1
        elif np.sign(human_difference) == np.sign(model_difference):
            concordant += 1
        else:
            discordant += 1
    denominator = concordant + discordant + model_ties
    probability = (
        (concordant + 0.5 * model_ties) / denominator if denominator else np.nan
    )
    return probability, concordant, discordant, model_ties, denominator


def bootstrap_concordance_interval(human, model, draws=N_ITEM_BOOTSTRAPS, seed=SEED):
    human, model = np.asarray(human), np.asarray(model)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(draws):
        indices = rng.integers(0, len(human), len(human))
        estimate = concordance_probability(human[indices], model[indices])[0]
        if np.isfinite(estimate):
            estimates.append(estimate)
    return np.percentile(estimates, [2.5, 97.5])


def kendall_tau_b(x, y):
    """Kendall's tau-b, including ties in either variable."""
    concordant = discordant = ties_x = ties_y = 0
    for i, j in itertools.combinations(range(len(x)), 2):
        dx = np.sign(x[i] - x[j])
        dy = np.sign(y[i] - y[j])
        if dx == 0 and dy == 0:
            continue
        if dx == 0:
            ties_x += 1
        elif dy == 0:
            ties_y += 1
        elif dx == dy:
            concordant += 1
        else:
            discordant += 1
    denominator = np.sqrt(
        (concordant + discordant + ties_x)
        * (concordant + discordant + ties_y)
    )
    return (concordant - discordant) / denominator if denominator else np.nan


def spearman_rho(x, y):
    """Spearman correlation using average ranks for ties."""
    x_rank = pd.Series(x).rank(method="average").to_numpy()
    y_rank = pd.Series(y).rank(method="average").to_numpy()
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def ordering_metrics(human, model):
    tau_b = kendall_tau_b(human, model)
    concordance, concordant, discordant, ties, ordered = concordance_probability(human, model)
    return {
        "kendall_tau_b": float(tau_b),
        "concordance": float(concordance),
        "concordant_pairs": concordant,
        "discordant_pairs": discordant,
        "model_tied_pairs": ties,
        "human_ordered_pairs": ordered,
    }


def annotator_profiles(cells, order, annotators):
    rows = []
    for annotator in annotators:
        values = [cells[(sentence, annotator)] for sentence in order if cells[(sentence, annotator)] is not None]
        flagged = sum(bool(labels - {"NB"}) for labels in values)
        rows.append(
            {
                "annotator": annotator,
                "valid_sentences": len(values),
                "flagged_sentences": flagged,
                "flag_share": flagged / len(values),
                **{category: sum(category in labels for labels in values) for category in CATEGORIES},
            }
        )
    return pd.DataFrame(rows)


def model_profiles(consolidated, models, order):
    rows = []
    for model in models:
        values = [
            consolidated[(model, sentence)]
            for sentence in order
            if consolidated[(model, sentence)] is not None
        ]
        flagged = sum(labels != frozenset({"NB"}) for labels in values)
        rows.append(
            {
                "model": model,
                "valid_sentences": len(values),
                "flagged_sentences": flagged,
                "flag_share": flagged / len(values),
                **{category: sum(category in labels for labels in values) for category in CATEGORIES},
            }
        )
    return pd.DataFrame(rows)


def repetition_stability(repetitions, model, order):
    stable = comparable = 0
    for sentence in order:
        values = [
            repetitions[(model, rep, sentence)]
            for rep in (1, 2, 3)
            if not is_contradictory(repetitions[(model, rep, sentence)])
        ]
        if len(values) >= 2:
            comparable += 1
            stable += int(len(set(values)) == 1)
    return stable, comparable


def six_person_panel_distribution(cells, order, annotators, category, draws=N_PANEL_DRAWS):
    rng = np.random.default_rng(SEED)
    annotators = np.asarray(annotators)
    estimates = []
    for _ in range(draws):
        subset = rng.choice(annotators, 6, replace=False)
        selected, n_valid = selected_counts_human(cells, order, subset, category)
        estimate = fleiss_kappa(selected, n_valid)
        if np.isfinite(estimate):
            estimates.append(estimate)
    return np.asarray(estimates)


def primary_sentence_share_table(cells, consolidated, models, order, annotators):
    rows = []
    for category in [None] + BIAS_CATEGORIES:
        label = "Any bias" if category is None else category
        human_selected, human_valid = selected_counts_human(cells, order, annotators, category)
        model_selected, model_valid = selected_counts_models(consolidated, models, order, category)
        human_proportion = human_selected / human_valid
        model_proportion = model_selected / model_valid
        human_interval = bootstrap_mean_interval(human_proportion)
        model_interval = bootstrap_mean_interval(model_proportion)
        difference_interval = bootstrap_difference_interval(human_proportion, model_proportion)
        rows.append(
            {
                "category": label,
                "human_mean_sentence_share": human_proportion.mean(),
                "human_item_bootstrap_low": human_interval[0],
                "human_item_bootstrap_high": human_interval[1],
                "model_mean_sentence_share": model_proportion.mean(),
                "model_item_bootstrap_low": model_interval[0],
                "model_item_bootstrap_high": model_interval[1],
                "human_minus_model": (human_proportion - model_proportion).mean(),
                "difference_item_bootstrap_low": difference_interval[0],
                "difference_item_bootstrap_high": difference_interval[1],
            }
        )
    return pd.DataFrame(rows)


def agreement_tables(cells, consolidated, models, order, annotators):
    primary_rows = []
    diagnostic_rows = []
    panel_rows = []
    for category in BIAS_CATEGORIES + [None]:
        label = "Any bias / No Bias" if category is None else category
        human_selected, human_valid = selected_counts_human(cells, order, annotators, category)
        model_selected, model_valid = selected_counts_models(consolidated, models, order, category)

        primary_rows.append(
            {
                "category": label,
                "human_alpha": krippendorff_alpha_nominal(human_selected, human_valid),
                "human_observed_pairwise_agreement": observed_pairwise_agreement(human_selected, human_valid),
                "model_alpha": krippendorff_alpha_nominal(model_selected, model_valid),
                "model_observed_pairwise_agreement": observed_pairwise_agreement(model_selected, model_valid),
            }
        )

        human_kappa_interval = bootstrap_kappa_interval(human_selected, human_valid)
        model_kappa_interval = bootstrap_kappa_interval(model_selected, model_valid)
        human_kappa = fleiss_kappa(human_selected, human_valid)
        model_kappa = fleiss_kappa(model_selected, model_valid)
        diagnostic_rows.append(
            {
                "category": label,
                "human_fleiss_kappa": human_kappa,
                "human_kappa_item_bootstrap_low": human_kappa_interval[0],
                "human_kappa_item_bootstrap_high": human_kappa_interval[1],
                "model_fleiss_kappa": model_kappa,
                "model_kappa_item_bootstrap_low": model_kappa_interval[0],
                "model_kappa_item_bootstrap_high": model_kappa_interval[1],
            }
        )

        human_panel_distribution = six_person_panel_distribution(
            cells, order, annotators, category
        )
        panel_rows.append(
            {
                "category": label,
                "model_fleiss_kappa": model_kappa,
                "six_human_panel_median_kappa": np.median(human_panel_distribution),
                "six_human_panel_low": np.percentile(human_panel_distribution, 2.5),
                "six_human_panel_high": np.percentile(human_panel_distribution, 97.5),
                "model_percentile_in_six_human_distribution": (
                    human_panel_distribution < model_kappa
                ).mean() * 100,
            }
        )
    return pd.DataFrame(primary_rows), pd.DataFrame(diagnostic_rows), pd.DataFrame(panel_rows)


def ordering_and_robustness(cells, consolidated, models, order, annotators, raw):
    human_selected, human_valid = selected_counts_human(cells, order, annotators)
    model_selected, model_valid = selected_counts_models(consolidated, models, order)
    human_proportion = human_selected / human_valid
    model_proportion = model_selected / model_valid

    overall_metrics = ordering_metrics(human_proportion, model_proportion)
    concordance_interval = bootstrap_concordance_interval(human_proportion, model_proportion)
    overall = pd.DataFrame(
        [
            {
                "human_mean_sentence_share": human_proportion.mean(),
                "model_mean_sentence_share": model_proportion.mean(),
                "human_minus_model": (human_proportion - model_proportion).mean(),
                **overall_metrics,
                "concordance_item_bootstrap_low": concordance_interval[0],
                "concordance_item_bootstrap_high": concordance_interval[1],
                "model_distinct_endorsement_values": len(set(model_proportion)),
            }
        ]
    )

    by_model_rows = []
    for model in models:
        selected, valid = selected_counts_models(consolidated, [model], order)
        proportion = selected / valid
        by_model_rows.append(
            {
                "model": model,
                "mean_flag_share": proportion.mean(),
                **ordering_metrics(human_proportion, proportion),
            }
        )
    by_model = pd.DataFrame(by_model_rows)

    article_indices = {
        article: np.asarray([i for i, sentence in enumerate(order) if sentence.startswith(article + "_")])
        for article in ("P1", "P2", "P3")
    }
    by_article_rows = []
    leave_article_rows = []
    for article, indices in article_indices.items():
        by_article_rows.append(
            {
                "article": article,
                "n_sentences": len(indices),
                "human_mean_sentence_share": human_proportion[indices].mean(),
                "model_mean_sentence_share": model_proportion[indices].mean(),
                "human_minus_model": (
                    human_proportion[indices] - model_proportion[indices]
                ).mean(),
                **ordering_metrics(human_proportion[indices], model_proportion[indices]),
            }
        )
        retained = np.asarray([i for i in range(len(order)) if i not in set(indices)])
        leave_article_rows.append(
            {
                "article_omitted": article,
                "n_sentences": len(retained),
                "human_mean_sentence_share": human_proportion[retained].mean(),
                "model_mean_sentence_share": model_proportion[retained].mean(),
                "human_minus_model": (
                    human_proportion[retained] - model_proportion[retained]
                ).mean(),
                **ordering_metrics(human_proportion[retained], model_proportion[retained]),
            }
        )

    leave_model_rows = []
    for omitted in models:
        retained_models = [model for model in models if model != omitted]
        selected, valid = selected_counts_models(consolidated, retained_models, order)
        proportion = selected / valid
        leave_model_rows.append(
            {
                "model_omitted": omitted,
                "n_models": len(retained_models),
                "model_mean_sentence_share": proportion.mean(),
                "human_minus_model": (human_proportion - proportion).mean(),
                **ordering_metrics(human_proportion, proportion),
            }
        )

    leave_annotator_rows = []
    for omitted in annotators:
        retained_annotators = [a for a in annotators if a != omitted]
        selected, valid = selected_counts_human(cells, order, retained_annotators)
        proportion = selected / valid
        leave_annotator_rows.append(
            {
                "annotator_omitted": omitted,
                "n_annotators": len(retained_annotators),
                "human_mean_sentence_share": proportion.mean(),
                "human_minus_model": (proportion - model_proportion).mean(),
                **ordering_metrics(proportion, model_proportion),
            }
        )
    leave_annotator = pd.DataFrame(leave_annotator_rows)

    profiles = annotator_profiles(cells, order, annotators)
    retained_annotators = profiles.loc[
        profiles.flag_share.between(0.10, 0.90, inclusive="both"), "annotator"
    ].astype(int).tolist()
    excluded_annotators = sorted(set(annotators) - set(retained_annotators))
    selected, valid = selected_counts_human(cells, order, retained_annotators)
    sensitivity_proportion = selected / valid
    extreme_sensitivity = pd.DataFrame(
        [
            {
                "rule": "retain annotators with flag share from 0.10 through 0.90",
                "post_hoc": True,
                "n_annotators_retained": len(retained_annotators),
                "excluded_annotators": ",".join(map(str, excluded_annotators)),
                "human_mean_sentence_share": sensitivity_proportion.mean(),
                "model_mean_sentence_share": model_proportion.mean(),
                "human_minus_model": (
                    sensitivity_proportion - model_proportion
                ).mean(),
                **ordering_metrics(sensitivity_proportion, model_proportion),
            }
        ]
    )

    annotator_summary = pd.DataFrame(
        [
            {
                "analysis": "full human panel",
                "n_annotators": len(annotators),
                "minimum_leave_one_out_human_share": leave_annotator.human_mean_sentence_share.min(),
                "maximum_leave_one_out_human_share": leave_annotator.human_mean_sentence_share.max(),
                "minimum_leave_one_out_difference": leave_annotator.human_minus_model.min(),
                "maximum_leave_one_out_difference": leave_annotator.human_minus_model.max(),
            },
            {
                "analysis": "post hoc 10%-90% sensitivity",
                "n_annotators": len(retained_annotators),
                "minimum_leave_one_out_human_share": sensitivity_proportion.mean(),
                "maximum_leave_one_out_human_share": sensitivity_proportion.mean(),
                "minimum_leave_one_out_difference": (
                    sensitivity_proportion - model_proportion
                ).mean(),
                "maximum_leave_one_out_difference": (
                    sensitivity_proportion - model_proportion
                ).mean(),
            },
        ]
    )

    return {
        "human_proportion": human_proportion,
        "model_proportion": model_proportion,
        "overall": overall,
        "by_model": by_model,
        "by_article": pd.DataFrame(by_article_rows),
        "leave_one_article_out": pd.DataFrame(leave_article_rows),
        "leave_one_model_out": pd.DataFrame(leave_model_rows),
        "leave_one_annotator_out": leave_annotator,
        "annotator_robustness_summary": annotator_summary,
        "extreme_response_sensitivity": extreme_sensitivity,
    }


def supplementary_tables(cells, consolidated, models, repetitions, order, annotators):
    human_label_counts = {
        category: int(selected_counts_human(cells, order, annotators, category)[0].sum())
        for category in CATEGORIES
    }
    model_label_counts = {
        category: int(selected_counts_models(consolidated, models, order, category)[0].sum())
        for category in CATEGORIES
    }
    label_rows = []
    for track, counts in (("human", human_label_counts), ("model", model_label_counts)):
        total = sum(counts.values())
        bias_total = sum(counts[category] for category in BIAS_CATEGORIES)
        for category in CATEGORIES:
            label_rows.append(
                {
                    "track": track,
                    "category": category,
                    "assigned_label_count": counts[category],
                    "all_label_share": counts[category] / total,
                    "within_bias_label_share": (
                        counts[category] / bias_total if category in BIAS_CATEGORIES else np.nan
                    ),
                }
            )

    repetition_stability_rows = []
    for model in models:
        stable, comparable = repetition_stability(repetitions, model, order)
        repetition_stability_rows.append(
            {
                "model": model,
                "stable_cells": stable,
                "comparable_cells": comparable,
                "exact_stability_share": stable / comparable,
            }
        )

    human_selected, human_valid = selected_counts_human(cells, order, annotators)
    human_proportion = human_selected / human_valid
    repetition_ordering_rows = []
    for repetition in (1, 2, 3):
        selected, valid = selected_counts_repetition(
            repetitions, models, order, repetition
        )
        proportion = selected / valid
        repetition_ordering_rows.append(
            {
                "repetition": repetition,
                "valid_model_sentence_cells": int(valid.sum()),
                "mean_flag_share": proportion.mean(),
                **ordering_metrics(human_proportion, proportion),
                "spearman_rho": spearman_rho(human_proportion, proportion),
            }
        )

    category_ordering_rows = []
    for category in BIAS_CATEGORIES:
        human_selected, human_valid = selected_counts_human(
            cells, order, annotators, category
        )
        model_selected, model_valid = selected_counts_models(
            consolidated, models, order, category
        )
        human_proportion = human_selected / human_valid
        model_proportion = model_selected / model_valid
        category_ordering_rows.append(
            {
                "category": category,
                "human_mean_sentence_share": human_proportion.mean(),
                "model_mean_sentence_share": model_proportion.mean(),
                **ordering_metrics(human_proportion, model_proportion),
                "spearman_rho": spearman_rho(human_proportion, model_proportion),
            }
        )

    return {
        "label_composition": pd.DataFrame(label_rows),
        "repetition_stability": pd.DataFrame(repetition_stability_rows),
        "repetition_ordering": pd.DataFrame(repetition_ordering_rows),
        "category_ordering": pd.DataFrame(category_ordering_rows),
    }


def crosscheck_aggregate(raw, raw_cells, aggregate_path: Path | None):
    if aggregate_path is None:
        return {"available": False}
    aggregate = pd.read_csv(aggregate_path, encoding="utf-8-sig")
    aggregate.columns = [
        "sentence_id", "english", "chinese", "RB", "LEB", "M", "NB", "total", "n_annotators"
    ]
    aggregate["index"] = aggregate.sentence_id.str.extract(r"(\d+)")[0].astype(int)
    offsets = {"P1": 0, "P2": 20, "P3": 39}
    lookup = aggregate.set_index("index")
    mismatches = {}
    for category in CATEGORIES:
        raw_counts = []
        aggregate_counts = []
        for _, row in raw.iterrows():
            key = row.sentence_key
            raw_counts.append(sum(category in raw_cells[(key, a)] for a in range(1, 31)))
            aggregate_index = int(row.number + offsets[row.article])
            aggregate_counts.append(int(lookup.loc[aggregate_index, category]))
        mismatches[category] = int(np.count_nonzero(np.asarray(raw_counts) != np.asarray(aggregate_counts)))
    return {"available": True, "mismatches": mismatches, "total_mismatches": sum(mismatches.values())}


def write_table(table: pd.DataFrame, path: Path):
    table.to_csv(path, index=False, float_format="%.6f")


def manuscript_tables(primary_shares, agreement_primary, by_model):
    """Return compact, publication-facing versions of the three Results tables."""

    def estimate_interval(row, estimate, low, high):
        return f"{row[estimate]:.3f} [{row[low]:.3f}, {row[high]:.3f}]"

    table_2_rows = []
    for _, row in primary_shares.iterrows():
        table_2_rows.append(
            {
                "Category": row.category,
                "Human mean sentence share [95% item-bootstrap CI]": estimate_interval(
                    row,
                    "human_mean_sentence_share",
                    "human_item_bootstrap_low",
                    "human_item_bootstrap_high",
                ),
                "Model mean sentence share [95% item-bootstrap CI]": estimate_interval(
                    row,
                    "model_mean_sentence_share",
                    "model_item_bootstrap_low",
                    "model_item_bootstrap_high",
                ),
                "Human - model [95% item-bootstrap CI]": estimate_interval(
                    row,
                    "human_minus_model",
                    "difference_item_bootstrap_low",
                    "difference_item_bootstrap_high",
                ),
            }
        )

    table_3 = agreement_primary.rename(
        columns={
            "category": "Category",
            "human_alpha": "Human alpha",
            "human_observed_pairwise_agreement": "Human observed pairwise agreement",
            "model_alpha": "Model alpha",
            "model_observed_pairwise_agreement": "Model observed pairwise agreement",
        }
    ).copy()
    table_3 = table_3[
        [
            "Category",
            "Human alpha",
            "Human observed pairwise agreement",
            "Model alpha",
            "Model observed pairwise agreement",
        ]
    ]

    table_4 = by_model.rename(
        columns={
            "model": "Model",
            "mean_flag_share": "Mean flag share",
            "kendall_tau_b": "Kendall tau-b",
            "concordance": "Concordance",
        }
    ).copy()
    table_4 = table_4[
        ["Model", "Mean flag share", "Kendall tau-b", "Concordance"]
    ]
    table_4["Model"] = table_4["Model"].str.title()

    return {
        "Table2_sentence_shares": pd.DataFrame(table_2_rows),
        "Table3_agreement": table_3,
        "Table4_by_model": table_4,
    }


def create_classification_figure(
    human_sentence_share,
    model_sentence_share,
    human_profiles,
    model_profiles_table,
    output_path,
):
    """Create the two-panel Results figure with Pillow (no plotting dependency)."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required to create Figure 1. Install pillow and rerun pipeline v5."
        ) from exc

    width, height = 3600, 1600
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    regular_candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    bold_candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ]

    def load_font(size, bold=False):
        candidates = bold_candidates if bold else regular_candidates
        for candidate in candidates:
            if candidate.exists():
                return ImageFont.truetype(str(candidate), size=size)
        return ImageFont.load_default()

    title_font = load_font(66, bold=True)
    panel_font = load_font(52, bold=True)
    label_font = load_font(42)
    tick_font = load_font(35)
    note_font = load_font(32)
    blue = "#0072B2"
    orange = "#D55E00"
    charcoal = "#222222"
    mid_gray = "#808080"
    light_gray = "#D6D6D6"

    def dashed_line(points, fill, width_px=5, dash=24, gap=18):
        (x1, y1), (x2, y2) = points
        distance = float(np.hypot(x2 - x1, y2 - y1))
        if distance == 0:
            return
        step_x = (x2 - x1) / distance
        step_y = (y2 - y1) / distance
        cursor = 0.0
        while cursor < distance:
            end = min(cursor + dash, distance)
            draw.line(
                [
                    (x1 + step_x * cursor, y1 + step_y * cursor),
                    (x1 + step_x * end, y1 + step_y * end),
                ],
                fill=fill,
                width=width_px,
            )
            cursor += dash + gap

    def vertical_label(x, y, label):
        bounds = draw.textbbox((0, 0), label, font=label_font)
        text_width = bounds[2] - bounds[0]
        text_height = bounds[3] - bounds[1]
        layer = Image.new("RGBA", (text_width + 30, text_height + 30), (255, 255, 255, 0))
        layer_draw = ImageDraw.Draw(layer)
        layer_draw.text((15, 15 - bounds[1]), label, fill=charcoal, font=label_font)
        rotated = layer.rotate(90, expand=True)
        image.paste(
            rotated,
            (int(x - rotated.width / 2), int(y - rotated.height / 2)),
            rotated,
        )

    draw.text(
        (width / 2, 58),
        "Human-model classification patterns",
        fill=charcoal,
        font=title_font,
        anchor="ma",
    )

    # Panel A: sentence-level relationship.
    ax1 = (240, 300, 1680, 1370)
    x0, y0, x1, y1 = ax1
    draw.text((x0, 205), "A  Sentence-level endorsement", fill=charcoal, font=panel_font)
    draw.line([(x0, y1), (x1, y1)], fill=charcoal, width=6)
    draw.line([(x0, y0), (x0, y1)], fill=charcoal, width=6)
    for tick in np.linspace(0, 1, 6):
        px = x0 + tick * (x1 - x0)
        py = y1 - tick * (y1 - y0)
        draw.line([(px, y1), (px, y1 + 18)], fill=charcoal, width=4)
        draw.line([(x0 - 18, py), (x0, py)], fill=charcoal, width=4)
        draw.text((px, y1 + 30), f"{tick:.1f}", fill=charcoal, font=tick_font, anchor="ma")
        draw.text((x0 - 34, py), f"{tick:.1f}", fill=charcoal, font=tick_font, anchor="rm")
        if tick not in (0, 1):
            draw.line([(x0, py), (x1, py)], fill=light_gray, width=2)
            draw.line([(px, y0), (px, y1)], fill=light_gray, width=2)
    dashed_line([(x0, y1), (x1, y0)], mid_gray, width_px=4)
    human_sentence_share = np.asarray(human_sentence_share, float)
    model_sentence_share = np.asarray(model_sentence_share, float)
    for human_value, model_value in zip(human_sentence_share, model_sentence_share):
        px = x0 + human_value * (x1 - x0)
        py = y1 - model_value * (y1 - y0)
        radius = 14
        draw.ellipse(
            [(px - radius, py - radius), (px + radius, py + radius)],
            fill=orange,
            outline="white",
            width=3,
        )
    human_mean = human_sentence_share.mean()
    model_mean = model_sentence_share.mean()
    dashed_line(
        [(x0 + human_mean * (x1 - x0), y0), (x0 + human_mean * (x1 - x0), y1)],
        blue,
        width_px=5,
    )
    dashed_line(
        [(x0, y1 - model_mean * (y1 - y0)), (x1, y1 - model_mean * (y1 - y0))],
        orange,
        width_px=5,
    )
    draw.text(
        ((x0 + x1) / 2, 1510),
        "Human sentence share",
        fill=charcoal,
        font=label_font,
        anchor="ma",
    )
    vertical_label(72, (y0 + y1) / 2, "Model sentence share")

    # Panel B: distributions of individual panel-member flag shares.
    ax2 = (2150, 300, 3400, 1370)
    x0b, y0b, x1b, y1b = ax2
    draw.text((x0b, 205), "B  Panel-member flag shares", fill=charcoal, font=panel_font)
    draw.line([(x0b, y1b), (x1b, y1b)], fill=charcoal, width=6)
    draw.line([(x0b, y0b), (x0b, y1b)], fill=charcoal, width=6)
    for tick in np.linspace(0, 1, 6):
        py = y1b - tick * (y1b - y0b)
        draw.line([(x0b - 18, py), (x0b, py)], fill=charcoal, width=4)
        draw.text((x0b - 34, py), f"{tick:.1f}", fill=charcoal, font=tick_font, anchor="rm")
        if tick not in (0, 1):
            draw.line([(x0b, py), (x1b, py)], fill=light_gray, width=2)
    group_x = {"Human annotators": 2560, "Models": 3130}
    distributions = {
        "Human annotators": np.asarray(human_profiles.flag_share, float),
        "Models": np.asarray(model_profiles_table.flag_share, float),
    }
    colors = {"Human annotators": blue, "Models": orange}
    rng = np.random.default_rng(SEED)
    for label, values in distributions.items():
        center = group_x[label]
        jitter = rng.uniform(-125, 125, size=len(values))
        for value, offset in zip(values, jitter):
            py = y1b - value * (y1b - y0b)
            radius = 16 if label == "Human annotators" else 22
            draw.ellipse(
                [
                    (center + offset - radius, py - radius),
                    (center + offset + radius, py + radius),
                ],
                fill=colors[label],
                outline="white",
                width=3,
            )
        median_y = y1b - np.median(values) * (y1b - y0b)
        draw.line(
            [(center - 145, median_y), (center + 145, median_y)],
            fill=charcoal,
            width=9,
        )
        draw.text((center, y1b + 34), label, fill=charcoal, font=label_font, anchor="ma")
    vertical_label(2020, (y0b + y1b) / 2, "Flag share")
    draw.text(
        (x0, 1450),
        "Dashed colored lines in A mark panel means; the gray diagonal marks equality. Black bars in B mark medians.",
        fill=mid_gray,
        font=note_font,
    )
    image.save(output_path, dpi=(300, 300))


def main() -> None:
    args = parse_args()
    project_dir = args.project_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else project_dir / "DataAnalysis" / "output_v5"
    )
    main_dir = output_dir / "main"
    supplementary_dir = output_dir / "supplementary"
    main_dir.mkdir(parents=True, exist_ok=True)
    supplementary_dir.mkdir(parents=True, exist_ok=True)

    human_path, model_path, aggregate_path = resolve_inputs(project_dir)
    raw, human_raw_cells, order, annotators = load_human_raw(human_path)
    human_cells, human_deleted = clean_human(human_raw_cells)
    model_data = load_models(model_path)
    consolidated, models, repetitions, consolidation_log = consolidate_models(
        model_data, order
    )

    if len(models) != 6:
        raise ValueError(f"Expected six models; found {len(models)}")
    if any(is_contradictory(value) for value in human_cells.values() if value is not None):
        raise AssertionError("Contradictory human judgment survived cleaning")
    if any(is_contradictory(value) for value in consolidated.values() if value is not None):
        raise AssertionError("Contradictory model judgment survived consolidation")

    primary_shares = primary_sentence_share_table(
        human_cells, consolidated, models, order, annotators
    )
    human_profiles = annotator_profiles(human_cells, order, annotators)
    models_profiles = model_profiles(consolidated, models, order)
    agreement_primary, agreement_diagnostics, six_person_panel = agreement_tables(
        human_cells, consolidated, models, order, annotators
    )
    robustness = ordering_and_robustness(
        human_cells, consolidated, models, order, annotators, raw
    )
    supplementary = supplementary_tables(
        human_cells, consolidated, models, repetitions, order, annotators
    )

    main_tables = {
        "M1_primary_sentence_shares": primary_shares,
        "M2_human_annotator_profiles": human_profiles,
        "M3_model_profiles": models_profiles,
        "M4_agreement_primary": agreement_primary,
        "M5_ordering_primary": robustness["overall"],
        "M6_by_model": robustness["by_model"],
        "M7_by_article": robustness["by_article"],
        "M8_leave_one_model_and_article": pd.concat(
            [
                robustness["leave_one_model_out"].assign(analysis_type="leave one model out"),
                robustness["leave_one_article_out"].assign(analysis_type="leave one article out"),
            ],
            ignore_index=True,
            sort=False,
        ),
        "M9_annotator_robustness_summary": robustness["annotator_robustness_summary"],
    }
    supplementary_tables_map = {
        "S1_label_composition": supplementary["label_composition"],
        "S2_agreement_diagnostics": agreement_diagnostics,
        "S3_six_person_panel_comparison": six_person_panel,
        "S4_model_repetition_stability": supplementary["repetition_stability"],
        "S5_repetition_ordering": supplementary["repetition_ordering"],
        "S6_category_ordering": supplementary["category_ordering"],
        "S7_leave_one_annotator_out": robustness["leave_one_annotator_out"],
        "S8_extreme_response_sensitivity": robustness["extreme_response_sensitivity"],
        "S9_spearman_overall": pd.DataFrame(
            [
                {
                    "analysis": "overall",
                    "spearman_rho": spearman_rho(
                        robustness["human_proportion"], robustness["model_proportion"]
                    ),
                }
            ]
        ),
    }

    for name, table in main_tables.items():
        write_table(table, main_dir / f"{name}.csv")
    for name, table in supplementary_tables_map.items():
        write_table(table, supplementary_dir / f"{name}.csv")

    for name, table in manuscript_tables(
        primary_shares, agreement_primary, robustness["by_model"]
    ).items():
        write_table(table, main_dir / f"{name}.csv")

    figure_sentence_data = pd.DataFrame(
        {
            "sentence_key": order,
            "human_sentence_share": robustness["human_proportion"],
            "model_sentence_share": robustness["model_proportion"],
        }
    )
    figure_profile_data = pd.concat(
        [
            human_profiles[["annotator", "flag_share"]]
            .rename(columns={"annotator": "member"})
            .assign(panel="Human annotators"),
            models_profiles[["model", "flag_share"]]
            .rename(columns={"model": "member"})
            .assign(panel="Models"),
        ],
        ignore_index=True,
    )
    write_table(figure_sentence_data, main_dir / "Figure1_sentence_data.csv")
    write_table(figure_profile_data, main_dir / "Figure1_profile_data.csv")
    create_classification_figure(
        robustness["human_proportion"],
        robustness["model_proportion"],
        human_profiles,
        models_profiles,
        main_dir / "Figure1_classification_patterns.png",
    )

    report = {
        "analysis_version": "v5",
        "seed": SEED,
        "input_files": {
            "human": str(human_path),
            "model": str(model_path),
            "aggregate_crosscheck": str(aggregate_path) if aggregate_path else None,
        },
        "data_quality": {
            "human_raw_judgments": len(human_raw_cells),
            "human_deleted_contradictions": len(human_deleted),
            "human_valid_judgments": len(human_raw_cells) - len(human_deleted),
            "model_parsed_rows": len(model_data),
            "model_repetition_cells": len(repetitions),
            "model_deleted_contradictions": len(consolidation_log["deleted_repetitions"]),
            "model_missing_consolidated_cells": len(consolidation_log["missing_cells"]),
            "model_synthesised_cells": len(consolidation_log["synthesised_cells"]),
            "aggregate_crosscheck": crosscheck_aggregate(
                raw, human_raw_cells, aggregate_path
            ),
        },
        "primary_sentence_shares": primary_shares.to_dict("records"),
        "agreement_primary": agreement_primary.to_dict("records"),
        "ordering_primary": robustness["overall"].to_dict("records")[0],
        "annotator_robustness": {
            "summary": robustness["annotator_robustness_summary"].to_dict("records"),
            "extreme_response_sensitivity": robustness[
                "extreme_response_sensitivity"
            ].to_dict("records")[0],
        },
        "outputs": {
            "main": sorted(path.name for path in main_dir.iterdir()),
            "supplementary": sorted(path.name for path in supplementary_dir.iterdir()),
        },
    }
    with (output_dir / "analysis_report_v5.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, default=str)

    print(f"Analysis complete. Main outputs: {main_dir}")
    print(f"Supplementary outputs: {supplementary_dir}")
    print(primary_shares.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\nAnnotator robustness:")
    print(
        robustness["annotator_robustness_summary"].to_string(
            index=False, float_format=lambda value: f"{value:.3f}"
        )
    )


if __name__ == "__main__":
    main()
