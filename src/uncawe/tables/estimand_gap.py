"""Estimand-gap check of BayCAP on stored ``main`` results (work-plan card WP-2).

BayCAP's posterior describes the population table $C\\operatorname{diag}(\\pi^Q)$, while ``main`` scores the realised
table of each target bag (``true_cts``). This module estimates, without new runs, how much a realised-table posterior
predictive would change BayCAP's vanilla-accuracy intervals.

``main`` stores BayCAP's accuracy samples but not its table draws, so the predictive is approximated. Of the $n^t_i$
target instances predicted as $\\omega_i$, the number correctly classified is $\\operatorname{Bin}(n^t_i,\\rho_{ii})$,
with $\\rho_{ii}$ the precision of class $\\omega_i$; the realised accuracy therefore fluctuates around the population
accuracy with variance

$$\\sigma_g^2=\\frac{1}{(N^Q)^2}\\sum_i n^t_i\\,\\rho_{ii}(1-\\rho_{ii}).$$

Per bag, $\\bar\\rho$ is the stored mean table ``estim_cts`` normalised per predicted class, and $n^t$ are the
predicted counts (the predicted-class margins of ``true_cts``, observable without labels). Each approximate
predictive sample is a stored accuracy sample plus independent $\\mathcal N(0,\\hat\\sigma_g^2)$ noise. The
approximation ignores the variation of $\\rho$ across posterior draws and the non-normality of the binomial sum, and
applies to vanilla accuracy only. Labels are used only for scoring.

Outputs, under ``<OUT_DIR>/uncawe/tables/estimand_gap/``: a per-(dataset, classifier) CSV, a LaTeX tabular fragment
and a standalone document. Tables follow the ``cap-ml`` layout (true classes on rows, predicted classes on columns).
"""

import os
import subprocess
import zlib
from argparse import ArgumentParser
from glob import glob

import numpy as np
import pandas as pd

from uncawe import env
from uncawe.config import SAMPLE_SIZE, acc_from_ct
from uncawe.main import EXPERIMENT
from uncawe.results import Results
from uncawe.tables import main as main_tables
from uncawe.util import decorate_dataset, pretrain_basedir

TABLE_NAME = "estimand_gap"
DOMAIN = "classic"
ACC_NAME = "vanilla_accuracy"
METHOD = "BayesCAP"
CLASSIFIER_ORDER = ("LR", "kNN", "SVM", "MLP", "RF")
SEED = 0
INTERVALS = ("population", "predictive")
INTERVAL_LABELS = {"population": "BayCAP", "predictive": "approx. predictive"}
SCORES = ("coverage", "amplitude", "winkler")
UNDER_COVERAGE = 0.90
NOMINAL_BAND = (0.93, 0.97)
SMALL_RATIO = 0.3


def pred_counts(true_ct: np.ndarray, sample_size: int = SAMPLE_SIZE) -> np.ndarray:
    """Return the number of target instances predicted as each class (column sums of the ``cap-ml`` table)."""
    return np.rint(np.asarray(true_ct).sum(axis=0) * sample_size).astype(int)


def precisions(estim_ct: np.ndarray) -> np.ndarray:
    """Return $\\bar\\rho_{ii}$, the precision of each predicted class under a ``cap-ml`` table (0 if never predicted)."""
    estim_ct = np.asarray(estim_ct, dtype=float)
    col_mass = estim_ct.sum(axis=0)
    return np.divide(np.diag(estim_ct), col_mass, out=np.zeros_like(col_mass), where=col_mass > 0)


def gap_std(n_pred: np.ndarray, rho: np.ndarray) -> float:
    """Return $\\hat\\sigma_g$, the std of the realised accuracy given the predicted counts and precisions."""
    n_pred = np.asarray(n_pred, dtype=float)
    rho = np.clip(np.asarray(rho, dtype=float), 0.0, 1.0)
    return float(np.sqrt(np.sum(n_pred * rho * (1 - rho))) / n_pred.sum())


def approx_predictive(samples: np.ndarray, sigma_g: float, rng: np.random.Generator) -> np.ndarray:
    """Add independent $\\mathcal N(0,\\sigma_g^2)$ noise to each stored accuracy sample."""
    samples = np.asarray(samples, dtype=float)
    return samples + rng.normal(0.0, sigma_g, size=samples.shape)


def pair_rng(dataset: str, classifier: str, seed: int = SEED) -> np.random.Generator:
    """Return a generator that depends only on the seed, the dataset and the classifier."""
    return np.random.default_rng([seed, zlib.crc32(f"{dataset}/{classifier}".encode())])


def validation_size(dataset: str, classifier: str, n_classes: int) -> int:
    """Return $|V|$, read from the stored validation posteriors of the pretrained classifier."""
    path = os.path.join(pretrain_basedir(), DOMAIN, f"{classifier}_{dataset}_{n_classes}_post.npz")
    if not os.path.exists(path):
        matches = glob(os.path.join(pretrain_basedir(), DOMAIN, f"*_{dataset}_{n_classes}_post.npz"))
        if not matches:
            raise FileNotFoundError(f"No pretrain posteriors found for {dataset} ({n_classes} classes)")
        path = matches[0]
    with np.load(path) as npz:
        return int(npz["V_posteriors"].shape[0])


def full_prevalence(prev, n_classes: int) -> np.ndarray:
    """Complete a stored prevalence (``util.get_plain_prev`` drops the first class)."""
    prev = np.atleast_1d(np.asarray(prev, dtype=float))
    if len(prev) == n_classes - 1:
        prev = np.concatenate([[1.0 - prev.sum()], prev])
    return prev


def bag_scores(row, rng: np.random.Generator) -> dict:
    true_acc = acc_from_ct(ACC_NAME, row.true_cts)
    samples = np.asarray(row.estim_acc_ranges, dtype=float)
    sigma_g = gap_std(pred_counts(row.true_cts), precisions(row.estim_cts))
    sigma_p = float(np.std(samples))
    out = {"sigma_g": sigma_g, "sigma_p": sigma_p, "ratio": sigma_g / sigma_p if sigma_p > 0 else np.inf}
    for interval, accs in zip(INTERVALS, (samples, approx_predictive(samples, sigma_g, rng))):
        out[(interval, "coverage")] = main_tables.get_acc_coverage(METHOD, true_acc, accs)
        out[(interval, "amplitude")] = main_tables.get_acc_amplitude(METHOD, accs)
        out[(interval, "winkler")] = main_tables.get_acc_winkler(METHOD, true_acc, accs)
    return out


def pair_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (dataset, classifier) with the means over bags and the regime descriptors."""
    rows = []
    for (dataset, classifier), _df in df.groupby(["dataset", "classifier"], sort=False):
        rng = pair_rng(dataset, classifier)
        bags = pd.DataFrame([bag_scores(row, rng) for row in _df.itertuples(index=False)])
        n_classes = int(_df["n_classes"].iloc[0])
        n_val = validation_size(dataset, classifier, n_classes)
        val_prev = full_prevalence(_df["val_prev"].iloc[0], n_classes)
        record = {
            "dataset": dataset,
            "classifier": classifier,
            "n_classes": n_classes,
            "n_val": n_val,
            "median_val_per_class": float(np.median(n_val * val_prev)),
            "ratio": float(bags["ratio"].median()),
            "sigma_g": float(bags["sigma_g"].mean()),
            "sigma_p": float(bags["sigma_p"].mean()),
        }
        for interval in INTERVALS:
            for score in SCORES:
                record[f"{interval}_{score}"] = float(bags[(interval, score)].mean())
        rows.append(record)
    return pd.DataFrame(rows)


def gate_summary(table: pd.DataFrame) -> dict:
    """Return the quantities used by the provisional decision rule of the Q0 gate."""
    under = table["population_coverage"] < UNDER_COVERAGE
    fixed = (
        under
        & table["predictive_coverage"].between(*NOMINAL_BAND)
        & (table["predictive_winkler"] < table["population_winkler"])
    )
    return {
        "pairs": len(table),
        "under_covering": int(under.sum()),
        "fixed_by_predictive": int(fixed.sum()),
        "ratio_below_small": int((table["ratio"] < SMALL_RATIO).sum()),
        "winkler_improved": int((table["predictive_winkler"] < table["population_winkler"]).sum()),
    }


def pilot_datasets(table: pd.DataFrame, classifier: str = "LR") -> dict:
    """Suggest the datasets of the exact pilot (WP-3): largest ratio, smallest ratio, one more multiclass."""
    _t = table.loc[table["classifier"] == classifier].sort_values("ratio")
    largest, smallest = _t["dataset"].iloc[-1], _t["dataset"].iloc[0]
    chosen = {"largest_ratio": largest, "smallest_ratio": smallest}
    if (_t.loc[_t["dataset"].isin(chosen.values()), "n_classes"] <= 2).all():
        multiclass = _t.loc[(_t["n_classes"] > 2) & ~_t["dataset"].isin(chosen.values())]
        if not multiclass.empty:
            chosen["multiclass"] = multiclass["dataset"].iloc[-1]
    return chosen


def _load_results() -> pd.DataFrame:
    base_dir = os.path.join(env.root_dir, EXPERIMENT)
    results = Results.load(
        base_dir=base_dir,
        domain=DOMAIN,
        acc_name=ACC_NAME,
        filter_classifiers=list(CLASSIFIER_ORDER),
        filter_methods=[METHOD],
    )
    return results.df


def _ordered(table: pd.DataFrame) -> pd.DataFrame:
    """Order binary before multiclass datasets, each by decreasing $|V|$ (a proxy for dataset size)."""
    table = table.copy()
    keys = table.assign(multiclass=table["n_classes"] > 2).drop_duplicates("dataset")
    order = keys.sort_values(["multiclass", "n_val"], ascending=[True, False])["dataset"].tolist()
    table["dataset"] = pd.Categorical(table["dataset"], categories=order, ordered=True)
    table["classifier"] = pd.Categorical(table["classifier"], categories=CLASSIFIER_ORDER, ordered=True)
    return table.sort_values(["dataset", "classifier"]).reset_index(drop=True)


CAPTION = (
    "Estimand-gap check of BayCAP on \\texttt{main} (vanilla accuracy, $|U|=1000$). "
    "$\\tilde n_V$: median validation instances per class; $\\hat\\sigma_g/\\hat\\sigma_p$: median over bags; "
    "coverage $\\mathcal{C}$, amplitude $\\mathcal{A}$ and Winkler score $\\mathcal{W}$ against the realised table, "
    "means over the 100 bags."
)
LABEL = "tab:estimand-gap"


def _styled_table(table: pd.DataFrame):
    """Return the Styler of the per-pair table; dataset-level descriptors are index levels, shown once per dataset."""
    cols = {("", "$\\hat\\sigma_g/\\hat\\sigma_p$"): table["ratio"].to_numpy()}
    for interval in INTERVALS:
        for score in SCORES:
            cols[(INTERVAL_LABELS[interval], main_tables.metrics_map[score])] = table[f"{interval}_{score}"].to_numpy()
    out = pd.DataFrame(cols)
    out.index = pd.MultiIndex.from_arrays(
        [
            table["dataset"].astype(str).map(lambda d: decorate_dataset(main_tables.dataset_map.get(d, d))),
            table["n_classes"].astype(int),
            table["median_val_per_class"].round().astype(int),
            table["classifier"].astype(str),
        ],
        names=["dataset", "$K$", "$\\tilde n_V$", "classifier"],
    )
    return out.style.format(precision=3)


def _booktabs_clines(latex: str) -> str:
    """Use booktabs rules between datasets: ``\\cline`` adds no vertical space and touches the next row."""
    return latex.replace("\\cline{", "\\cmidrule(lr){")


def _latex_table(table: pd.DataFrame) -> str:
    """Return a ``tabular`` fragment for inclusion in other documents."""
    return _booktabs_clines(_styled_table(table).to_latex(hrules=True, multicol_align="c", clines="skip-last;data"))


def _latex_longtable(table: pd.DataFrame) -> str:
    """Return a ``longtable`` that breaks across pages, used by the standalone document."""
    latex = _styled_table(table).to_latex(
        environment="longtable",
        hrules=True,
        multicol_align="c",
        multirow_align="naive",
        clines="skip-last;data",
        caption=CAPTION,
        label=LABEL,
    )
    return _booktabs_clines(latex)


def _latex_document(longtable_path: str) -> str:
    return "\n".join(
        [
            "\\documentclass{article}",
            "\\usepackage[a4paper,hmargin=1.5cm,vmargin=0.8cm]{geometry}",
            "\\usepackage{booktabs,longtable,multirow}",
            "\\begin{document}",
            "\\pagestyle{empty}",
            "\\scriptsize",
            "\\setlength{\\LTcapwidth}{\\textwidth}",
            f"\\input{{{longtable_path}}}",
            "\\end{document}",
            "",
        ]
    )


def _table_dir(pargs) -> str:
    return pargs.table_dir if pargs.table_dir is not None else os.path.join(env.root_dir, "tables", TABLE_NAME)


def gen_tables(pargs):
    df = _load_results()
    if df.empty:
        raise RuntimeError(f"No {METHOD} results found for {EXPERIMENT}. Check that experiment results exist.")

    table = _ordered(pair_table(df))
    table_dir = _table_dir(pargs)
    tabular_dir = os.path.join(table_dir, "tabular")
    os.makedirs(tabular_dir, exist_ok=True)

    table.to_csv(os.path.join(table_dir, f"{TABLE_NAME}.csv"), index=False)
    with open(os.path.join(tabular_dir, f"{TABLE_NAME}.tex"), "w") as f:
        f.write(_latex_table(table))
    with open(os.path.join(tabular_dir, f"{TABLE_NAME}_long.tex"), "w") as f:
        f.write(_latex_longtable(table))
    with open(os.path.join(table_dir, f"{TABLE_NAME}_doc.tex"), "w") as f:
        f.write(_latex_document(f"tabular/{TABLE_NAME}_long.tex"))

    with pd.option_context("display.width", 200, "display.max_rows", 200, "display.precision", 3):
        print(table.drop(columns=["sigma_g", "sigma_p"]).to_string(index=False))
    print("gate summary:", gate_summary(table))
    print("suggested pilot datasets (LR):", pilot_datasets(table))
    print(f"tables written to {table_dir}")


def gen_pdf(pargs):
    table_dir = _table_dir(pargs)
    tex_filename = f"{TABLE_NAME}_doc.tex"
    if not os.path.exists(os.path.join(table_dir, tex_filename)):
        raise FileNotFoundError(f"{tex_filename} does not exist in {table_dir}. Generate tables first.")

    # longtable needs a second pass to settle the column widths
    for _ in range(2):
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_filename],
            cwd=table_dir,
            check=True,
            stdout=subprocess.DEVNULL,
        )


if __name__ == "__main__":
    parser = ArgumentParser(description="Estimand-gap check of BayCAP on stored main results.")
    parser.add_argument("-t", "--tables", action="store_true", help="Generate tables")
    parser.add_argument("-p", "--pdf", action="store_true", help="Generate PDF from tables")
    parser.add_argument("-a", "--all", action="store_true", help="Generate both tables and PDF")
    parser.add_argument(
        "--table-dir",
        "--output-dir",
        dest="table_dir",
        default=None,
        help="Directory where the CSV, table documents and tabular fragments are saved",
    )
    pargs = parser.parse_args()

    if pargs.tables or pargs.all:
        gen_tables(pargs)
    if pargs.pdf or pargs.all:
        gen_pdf(pargs)
