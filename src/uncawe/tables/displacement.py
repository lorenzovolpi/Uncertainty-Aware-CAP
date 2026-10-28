"""Causes of BayCAP's displacement on stored ``main`` results (work-plan card WP-3, E2.2 of ``docs/ext2_baycap.md``).

BayCAP, as run in ``main`` (``BayesCAP``), targets the population table $C\\operatorname{diag}(\\pi^Q)$; the realised
pushforward ``BayesCAP-real`` (work-plan card WP-4, E2.3) targets the realised table of the target sample. The method is
an argument (default ``BayesCAP``). Every score here is computed against the realised table of each target bag
(``true_cts``). The analysis uses the bags' true class proportions and
recalls, so it is an analysis of the error, not a method. Vanilla accuracy, $|U|=1000$, all (dataset, classifier) pairs.

Per bag, the shift of BayCAP's point estimate from the realised accuracy splits exactly into a recall term and a
prevalence term,

$$\\hat A-A=\\sum_j\\pi_j(\\hat r_j-r_j)+\\sum_j(\\hat\\pi_j-\\pi_j)\\,\\hat r_j,$$

with $\\hat\\pi,\\hat r$ read from the stored mean table ``estim_cts`` and $\\pi,r$ from ``true_cts``. The prior part of
the recall term, $\\sum_j\\pi_j(\\tilde r_j-f_{jj}/n^s_j)$, is computed on the validation counts $f_{ij}$ rebuilt from
the stored validation posteriors and labels, as ``BayCAP.fit`` builds them, with $\\tilde r_j$ the posterior mean of the
recall given $V$ under the method's prior on the columns of $C$: $\\frac{f_{jj}+1}{n^s_j+K}$ under
$\\operatorname{Dir}(\\mathbf 1)$, $\\frac{f_{jj}+\\hat\\nu\\hat m}{n^s_j+\\hat\\nu}$ under the empirical-Bayes prior of
``BayesCAP-EB`` and ``BayesCAP-EB-real`` (work-plan card WP-6, E2.4).

Per pair, the predicted coverage tells how much of the under-coverage the shift alone explains: it is the coverage the
interval would have if its width were right for the spread of the error across bags and every bag were shifted by the
mean shift (normal approximation).

Per pair, the conditioning of $\\hat C$ is measured on the vectors summing to zero, through the singular values of
$Q^\\top\\hat CQ$ ($Q$ an orthonormal basis of those vectors): $\\kappa_0=\\sigma_{\\max}/\\sigma_{\\min}$ and the
absolute amplification $1/\\sigma_{\\min}$. With $K=2$, $Q^\\top\\hat CQ=\\mathrm{TPR}-\\mathrm{FPR}$, so $\\kappa_0=1$
and only $1/\\sigma_{\\min}$ is informative. The difficulty of the problem is the accuracy of $h$ on $V$.

Outputs, under ``<OUT_DIR>/uncawe/tables/displacement/`` for ``BayesCAP`` and under
``<OUT_DIR>/uncawe/tables/displacement/<method>/`` for any other method: a per-pair CSV and one table (diagnostics, interval scores
and error terms side by side) as a LaTeX tabular fragment, a longtable and a standalone landscape document. Tables follow the ``cap-ml``
layout (true classes on rows, predicted classes on columns); $\\hat C$ is returned in the paper's layout (predicted
classes on rows).
"""

import os
import subprocess
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from scipy.linalg import null_space
from scipy.stats import norm
from sklearn.metrics import confusion_matrix

from uncawe import env
from uncawe.config import CONFIDENCE_LEVEL, acc_from_ct
from uncawe.experiments.methods import BAYCAP_VARIANTS
from uncawe.main import EXPERIMENT
from uncawe.methods.baycap import fit_beta_binomial
from uncawe.pretrain.data import PretrainInfo, load_info_paths
from uncawe.tables import main as main_tables
from uncawe.tables.estimand_gap import ACC_NAME, CLASSIFIER_ORDER, DOMAIN, METHOD, booktabs_clines, load_results
from uncawe.util import decorate_dataset

TABLE_NAME = "displacement"
SCORES = ("coverage", "amplitude", "winkler")
TERMS = ("shift", "recall_term", "prior_part", "prevalence_term", "abs_prevalence_term", "prev_l1")
SINGULAR_RTOL = 1e-12


def class_props_and_recalls(ct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the class proportions $\\pi_j$ (row sums) and recalls $r_j$ of a ``cap-ml`` table (0 if $\\pi_j=0$)."""
    ct = np.asarray(ct, dtype=float)
    props = ct.sum(axis=1)
    recalls = np.divide(np.diag(ct), props, out=np.zeros_like(props), where=props > 0)
    return props, recalls


def error_terms(estim_ct: np.ndarray, true_ct: np.ndarray) -> dict:
    """Split the shift $\\hat A-A$ of a mean table from a realised table into its recall and prevalence terms.

    The recall term is written as $\\sum_j(\\pi_j\\hat r_j-T_{jj})$, equal to $\\sum_j\\pi_j(\\hat r_j-r_j)$ and defined
    also for classes absent from the bag; the two terms add up to the shift exactly.
    """
    true_ct = np.asarray(true_ct, dtype=float)
    est_props, est_recalls = class_props_and_recalls(estim_ct)
    props = true_ct.sum(axis=1)
    return {
        "shift": float(np.trace(np.asarray(estim_ct, dtype=float)) - np.trace(true_ct)),
        "recall_term": float(np.sum(props * est_recalls - np.diag(true_ct))),
        "prevalence_term": float(np.sum((est_props - props) * est_recalls)),
        "prev_l1": float(np.abs(est_props - props).sum()),
    }


def method_c_prior(method: str) -> str:
    """Return the prior on the columns of $C$ of a BayCAP method of ``main`` (``"uniform"`` for any other name)."""
    return BAYCAP_VARIANTS.get(method, ("population", "uniform"))[1]


def plain_recalls(val_counts: np.ndarray) -> np.ndarray:
    """Return the recalls $f_{jj}/n^s_j$ of a ``cap-ml`` validation table (NaN for a class absent from $V$)."""
    val_counts = np.asarray(val_counts, dtype=float)
    n_s = val_counts.sum(axis=1)
    return np.divide(np.diag(val_counts), n_s, out=np.full_like(n_s, np.nan), where=n_s > 0)


def smoothed_recalls(val_counts: np.ndarray, c_prior: str = "uniform") -> np.ndarray:
    """Return the posterior means of the recalls given $V$ under the prior ``c_prior`` on the columns of $C$.

    ``"uniform"``: $\\frac{f_{jj}+1}{n^s_j+K}$; ``"empirical-bayes"``: $\\frac{f_{jj}+\\hat\\nu\\hat m}{n^s_j+\\hat\\nu}$,
    with $(\\hat m,\\hat\\nu)$ fitted on the same counts as ``BayCAP.fit`` does (``fit_beta_binomial``).
    """
    val_counts = np.asarray(val_counts, dtype=float)
    n_s = val_counts.sum(axis=1)
    f_diag = np.diag(val_counts)
    if c_prior == "uniform":
        return (f_diag + 1) / (n_s + val_counts.shape[0])
    if c_prior == "empirical-bayes":
        fit = fit_beta_binomial(f_diag, n_s)
        return (f_diag + fit.nu * fit.m) / (n_s + fit.nu)
    raise ValueError(f"Unknown c_prior {c_prior!r}")


def prior_part(val_counts: np.ndarray, true_props: np.ndarray, smoothed: np.ndarray | None = None) -> float:
    """Return $\\sum_j\\pi_j(\\tilde r_j-f_{jj}/n^s_j)$, the shift that the prior on the columns of $C$ adds to the
    recalls estimated on $V$.

    ``val_counts`` is the validation confusion table in the ``cap-ml`` layout (true classes on rows); ``smoothed`` are
    the recalls $\\tilde r_j$ of :func:`smoothed_recalls`, by default those of the $\\operatorname{Dir}(\\mathbf 1)$
    prior, $\\frac{f_{jj}+1}{n^s_j+K}$.
    """
    if smoothed is None:
        smoothed = smoothed_recalls(val_counts)
    return float(np.sum(np.asarray(true_props, dtype=float) * (smoothed - plain_recalls(val_counts))))


def predicted_coverage(shift: float, amplitude: float, confidence_level: float = CONFIDENCE_LEVEL) -> float:
    """Return the coverage of an interval of the given amplitude displaced by ``shift``, under a normal error.

    The interval is read as $\hat A\pm z\sigma$, with $z$ the normal quantile of ``confidence_level`` and
    $\sigma=\text{amplitude}/(2z)$, and the error $\hat A-A$ as normal with mean ``shift`` and standard deviation
    $\sigma$. With $s=\text{shift}/(\text{amplitude}/2)$ the probability that $A$ falls in the interval is
    $\Phi(z(1-s))-\Phi(-z(1+s))$; it equals ``confidence_level`` when the shift is zero. NaN if the amplitude is zero.
    """
    if amplitude <= 0:
        return np.nan
    z = norm.ppf(0.5 + confidence_level / 2)
    s = shift / (amplitude / 2)
    return float(norm.cdf(z * (1 - s)) - norm.cdf(-z * (1 + s)))


def rate_matrix(val_counts: np.ndarray) -> np.ndarray:
    """Return $\\hat C$ in the paper's layout, $\\hat C_{ij}=P(\\hat Y=\\omega_i\\mid Y=\\omega_j)$ (columns sum to one)."""
    val_counts = np.asarray(val_counts, dtype=float)
    n_s = val_counts.sum(axis=1)
    return np.divide(val_counts, n_s[:, None], out=np.zeros_like(val_counts), where=n_s[:, None] > 0).T


def sum_zero_basis(n_classes: int) -> np.ndarray:
    """Return an orthonormal basis, of shape ``(K, K-1)``, of the vectors whose entries sum to zero."""
    return null_space(np.ones((1, n_classes)))


def conditioning(rates: np.ndarray) -> tuple[float, float]:
    """Return $(\\kappa_0, 1/\\sigma_{\\min})$ of a paper-layout rate matrix on the vectors summing to zero.

    Both are infinite when $Q^\\top CQ$ is singular (relative tolerance ``SINGULAR_RTOL``).
    """
    rates = np.asarray(rates, dtype=float)
    basis = sum_zero_basis(rates.shape[0])
    sv = np.linalg.svd(basis.T @ rates @ basis, compute_uv=False)
    s_max, s_min = float(sv.max()), float(sv.min())
    if s_min <= SINGULAR_RTOL * max(s_max, 1.0):
        return np.inf, np.inf
    return s_max / s_min, 1.0 / s_min


def validation_counts(info: PretrainInfo) -> np.ndarray:
    """Rebuild the validation confusion table of ``BayCAP.fit`` (``cap-ml`` layout) from the stored posteriors."""
    val = info.load_dataset_bundle().V
    val_posteriors, _ = info.load_posteriors()
    return confusion_matrix(val.y, np.argmax(val_posteriors, axis=-1), labels=val.classes)


def load_validation_counts(pairs, domain: str = DOMAIN) -> dict:
    """Return the validation confusion tables of the given (dataset, classifier) pairs of ``domain`` (default
    ``classic``)."""
    infos = {}
    for path in load_info_paths(domain=domain):
        p = PretrainInfo.load(path, fast=True)
        infos[(p.d_info.name, p.h_info.name)] = p
    missing = [pair for pair in pairs if pair not in infos]
    if missing:
        raise FileNotFoundError(f"No pretrain info found for {missing}")
    return {pair: validation_counts(infos[pair]) for pair in pairs}


def bag_scores(row, val_counts: np.ndarray, method: str = METHOD, smoothed: np.ndarray | None = None) -> dict:
    true_ct = np.asarray(row.true_cts, dtype=float)
    estim_ct = np.asarray(row.estim_cts, dtype=float)
    samples = np.asarray(row.estim_acc_ranges, dtype=float)
    true_acc = acc_from_ct(ACC_NAME, true_ct)
    out = error_terms(estim_ct, true_ct)
    out["prior_part"] = prior_part(val_counts, true_ct.sum(axis=1), smoothed)
    out["abs_prevalence_term"] = abs(out["prevalence_term"])
    out["coverage"] = main_tables.get_acc_coverage(method, true_acc, samples)
    out["amplitude"] = main_tables.get_acc_amplitude(method, samples)
    out["winkler"] = main_tables.get_acc_winkler(method, true_acc, samples)
    # accuracy is linear in the table: both gaps are zero up to floating point
    out["point_gap"] = abs(float(np.mean(samples)) - float(np.trace(estim_ct)))
    out["truth_gap"] = abs(true_acc - float(np.trace(true_ct)))
    return out


def pair_table(df: pd.DataFrame, val_counts: dict, method: str = METHOD) -> pd.DataFrame:
    """Return one row per (dataset, classifier): means over bags, then the per-pair diagnostics of $\\hat C$ and $V$.

    The prior part uses the prior on the columns of $C$ of ``method`` (:func:`method_c_prior`).
    """
    rows = []
    c_prior = method_c_prior(method)
    for (dataset, classifier), _df in df.groupby(["dataset", "classifier"], sort=False):
        counts = val_counts[(dataset, classifier)]
        smoothed = smoothed_recalls(counts, c_prior)
        bags = pd.DataFrame([bag_scores(row, counts, method, smoothed) for row in _df.itertuples(index=False)])
        n_s = counts.sum(axis=1)
        kappa0, inv_sigma_min = conditioning(rate_matrix(counts))
        record = {
            "dataset": dataset,
            "classifier": classifier,
            "n_classes": int(_df["n_classes"].iloc[0]),
            "n_bags": len(bags),
            "n_val": int(n_s.sum()),
            "min_val_per_class": int(n_s.min()),
            "median_val_per_class": float(np.median(n_s)),
            "kappa0": kappa0,
            "inv_sigma_min": inv_sigma_min,
            "val_accuracy": float(np.trace(counts) / n_s.sum()),
        }
        for col in SCORES + TERMS:
            record[col] = float(bags[col].mean())
        record["predicted_coverage"] = predicted_coverage(record["shift"], record["amplitude"])
        record["max_point_gap"] = float(bags["point_gap"].max())
        record["max_truth_gap"] = float(bags["truth_gap"].max())
        rows.append(record)
    return pd.DataFrame(rows)


def ordered(table: pd.DataFrame) -> pd.DataFrame:
    """Order datasets by $K$, then by decreasing $|V|$; classifiers as in ``CLASSIFIER_ORDER``."""
    table = table.copy()
    keys = table.drop_duplicates("dataset").sort_values(["n_classes", "n_val"], ascending=[True, False])
    table["dataset"] = pd.Categorical(table["dataset"], categories=keys["dataset"].tolist(), ordered=True)
    table["classifier"] = pd.Categorical(table["classifier"], categories=CLASSIFIER_ORDER, ordered=True)
    return table.sort_values(["dataset", "classifier"]).reset_index(drop=True)


def compute_pair_table(method: str = METHOD) -> pd.DataFrame:
    """Load the stored ``main`` results of ``method`` and the validation counts, and return the ordered per-pair table."""
    df = load_results(method)
    if df.empty:
        raise RuntimeError(f"No {method} results found for {EXPERIMENT}. Check that experiment results exist.")
    pairs = list(df.groupby(["dataset", "classifier"], sort=False).groups)
    return ordered(pair_table(df, load_validation_counts(pairs), method))


def _fmt_ratio(v: float) -> str:
    return "$\\infty$" if np.isinf(v) else f"{v:.2f}"


METHOD_LABELS = {
    "BayesCAP": "BayCAP",
    "BayesCAP-real": "BayCAP-real",
    "BayesCAP-EB": "BayCAP-EB",
    "BayesCAP-EB-real": "BayCAP-EB-real",
}
CAPTION = (
    "Causes of {method}'s displacement on \\texttt{main} (vanilla accuracy, $|U|=1000$). "
    "$n^s_j$: validation instances of class $\\omega_j$. Diagnostics: $\\kappa_0(\\hat C)$ and $1/\\sigma_{\\min}$, "
    "condition number and absolute amplification of $\\hat C$ on vectors summing to zero; $\\mathrm{acc}_V$, accuracy "
    "of $h$ on $V$. Interval and error: means over the 100 bags, against the realised table; $\\mathcal{C}_s$, coverage "
    "predicted from the mean shift and amplitude alone (normal approximation); $\\hat A-A$, shift; $R$, "
    "recall term; $R_0$, its prior part (under the method's prior on $C$); $P$, $|P|$, prevalence term and its absolute value; "
    "$\\lVert\\hat\\pi-\\pi\\rVert_1$, prevalence error."
)
LABEL = "tab:displacement"


def caption(method: str = METHOD) -> str:
    return CAPTION.replace("{method}", METHOD_LABELS.get(method, method))


def label(method: str = METHOD) -> str:
    return LABEL if method == METHOD else f"{LABEL}-{method}"


COLUMNS = {
    "kappa0": ("diagnostics", "$\\kappa_0(\\hat C)$"),
    "inv_sigma_min": ("diagnostics", "$1/\\sigma_{\\min}$"),
    "val_accuracy": ("diagnostics", "$\\mathrm{acc}_V$"),
    "coverage": ("interval", main_tables.metrics_map["coverage"]),
    "predicted_coverage": ("interval", "$\\mathcal{C}_s$"),
    "amplitude": ("interval", main_tables.metrics_map["amplitude"]),
    "winkler": ("interval", main_tables.metrics_map["winkler"]),
    "shift": ("error", "$\\hat A-A$"),
    "recall_term": ("error", "$R$"),
    "prior_part": ("error", "$R_0$"),
    "prevalence_term": ("error", "$P$"),
    "abs_prevalence_term": ("error", "$|P|$"),
    "prev_l1": ("error", "$\\lVert\\hat\\pi-\\pi\\rVert_1$"),
}


def _styled_table(table: pd.DataFrame):
    """Return the Styler of the per-pair table; dataset-level descriptors are index levels, shown once per dataset."""
    out = pd.DataFrame({label: table[col].to_numpy() for col, label in COLUMNS.items()})
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    out.index = pd.MultiIndex.from_arrays(
        [
            table["dataset"].astype(str).map(lambda d: decorate_dataset(main_tables.dataset_map.get(d, d))),
            table["n_classes"].astype(int),
            table["min_val_per_class"].astype(int),
            table["median_val_per_class"].round().astype(int),
            table["classifier"].astype(str),
        ],
        names=["dataset", "$K$", "$\\min n^s_j$", "$\\tilde n^s_j$", "classifier"],
    )
    ratio_cols = [COLUMNS["kappa0"], COLUMNS["inv_sigma_min"]]
    return out.style.format(precision=3).format(_fmt_ratio, subset=ratio_cols)


def _latex_table(table: pd.DataFrame) -> str:
    return booktabs_clines(_styled_table(table).to_latex(hrules=True, multicol_align="c", clines="skip-last;data"))


def _latex_longtable(table: pd.DataFrame, method: str = METHOD) -> str:
    latex = _styled_table(table).to_latex(
        environment="longtable",
        hrules=True,
        multicol_align="c",
        multirow_align="naive",
        clines="skip-last;data",
        caption=caption(method),
        label=label(method),
    )
    return booktabs_clines(latex)


def _latex_document(longtable_path: str) -> str:
    return "\n".join(
        [
            "\\documentclass{article}",
            "\\usepackage[a4paper,landscape,hmargin=1.5cm,vmargin=0.8cm]{geometry}",
            "\\usepackage{amsmath,booktabs,longtable,multirow}",
            "\\begin{document}",
            "\\pagestyle{empty}",
            "\\scriptsize",
            "\\setlength{\\LTcapwidth}{\\textwidth}",
            f"\\input{{{longtable_path}}}",
            "\\end{document}",
            "",
        ]
    )


def table_dir(method: str = METHOD) -> str:
    """Return the output folder of ``method``: the stored E2.2 folder for ``BayesCAP``, a subfolder for the others."""
    base = os.path.join(env.root_dir, "tables", TABLE_NAME)
    return base if method == METHOD else os.path.join(base, method)


def _table_dir(pargs) -> str:
    return pargs.table_dir if pargs.table_dir is not None else table_dir(pargs.method)


def gen_tables(pargs):
    table = compute_pair_table(pargs.method)
    out_dir = _table_dir(pargs)
    tabular_dir = os.path.join(out_dir, "tabular")
    os.makedirs(tabular_dir, exist_ok=True)

    table.to_csv(os.path.join(out_dir, f"{TABLE_NAME}.csv"), index=False)
    with open(os.path.join(tabular_dir, f"{TABLE_NAME}.tex"), "w") as f:
        f.write(_latex_table(table))
    with open(os.path.join(tabular_dir, f"{TABLE_NAME}_long.tex"), "w") as f:
        f.write(_latex_longtable(table, pargs.method))
    with open(os.path.join(out_dir, f"{TABLE_NAME}_doc.tex"), "w") as f:
        f.write(_latex_document(f"tabular/{TABLE_NAME}_long.tex"))

    with pd.option_context("display.width", 250, "display.max_rows", 200, "display.precision", 3):
        print(table.drop(columns=["max_point_gap", "max_truth_gap"]).to_string(index=False))
    print(
        f"correlation of coverage and predicted coverage: "
        f"{table['coverage'].corr(table['predicted_coverage']):.3f}; pairs with coverage more than 0.05 below the "
        f"prediction: {int((table['coverage'] < table['predicted_coverage'] - 0.05).sum())}"
    )
    print(
        f"pairs: {len(table)}; max |mean of samples - acc of mean table|: {table['max_point_gap'].max():.2e}; "
        f"max |true acc - trace of true table|: {table['max_truth_gap'].max():.2e}"
    )
    print(f"tables written to {out_dir}")


def gen_pdf(pargs):
    out_dir = _table_dir(pargs)
    tex_filename = f"{TABLE_NAME}_doc.tex"
    if not os.path.exists(os.path.join(out_dir, tex_filename)):
        raise FileNotFoundError(f"{tex_filename} does not exist in {out_dir}. Generate tables first.")

    # longtable needs a second pass to settle the column widths
    for _ in range(2):
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_filename],
            cwd=out_dir,
            check=True,
            stdout=subprocess.DEVNULL,
        )


if __name__ == "__main__":
    parser = ArgumentParser(description="Causes of BayCAP's displacement on stored main results (E2.2).")
    parser.add_argument("-t", "--tables", action="store_true", help="Generate tables")
    parser.add_argument("-p", "--pdf", action="store_true", help="Generate PDF from tables")
    parser.add_argument("-a", "--all", action="store_true", help="Generate both tables and PDF")
    parser.add_argument("--method", default=METHOD, help="Method of main to analyse (default: %(default)s)")
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
