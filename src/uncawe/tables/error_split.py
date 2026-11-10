"""The error split and the oracle $2\\times2$ of the main-table methods on stored ``main`` results (work-plan card WP-14,
E4.1 and E4.3 of ``docs/ext4_uncertainty_budget.md``).

Methods: the methods of the main table whose estimate comes from a contingency table, ``tables/main.py:MAIN_METHODS``
without ``DoC`` (which estimates accuracy directly): ``QuAcc_NxN``, ``O-LEAP``, ``SQS-H``, ``SQS-Hbs`` (SQS-H with a bootstrap of the prevalence), ``PQ`` (SQS-S),
``BayesCAP-real``, ``BayesCAP-Hf-real`` and ``BayCAP-KDEy-mlp-real``. Vanilla accuracy, $|U|=1000$, the pairs of
``main`` stored for all seven methods. Every method and every quantity is scored against the realised table of each bag
(``true_cts``).

Per bag and method (``uncawe.error_split``): the parts $e_1$ (prevalence), $e_2$ (recalls against the pool) and $e_3$
(luck of the bag) of the error $\\hat A-A(h,U)$ of the method's mean table ``estim_cts``, the cross term $x$ and the four
oracle estimates. The pool recalls $d^{\\text{pool}}$ are those of $h$ on the test pool from which the bags are drawn:
the argmax of the pretrain ``U_posteriors`` against the labels of the pool, the same predictions that build
``true_cts``. The split and the oracle cells use the true prevalence of the bag and the labels of the pool: they are an
analysis, and the three oracle cells are references, not methods.

Bags with a non-finite mean table (NUTS produced no draws, ``nuts_failed``) are dropped from that method and counted per
pair (``<method>_n_dropped``); bags with an empty estimated column are kept, with the convention of
``uncawe.error_split``, and counted (``<method>_n_empty``). The identity $e_1+e_2+e_3=\\texttt{estim\\_accs}-A(h,U)$ is
checked on every kept bag (``IDENTITY_TOL``); the largest gap is reported per pair. $e_3$ uses no estimate, so it is the
same for every method on the same bag: its single column is computed on all the bags of the pair; the part with the
largest RMS of each method compares its $e_1$ and $e_2$ with the $e_3$ of the bags it kept. The cell
$(\\pi^Q,d^{\\text{pool}})$ of E4.3 also uses no estimate and gets a single column for the same reason.

Tables (``--tables-set``, default both):

- ``e4.1``: table A, the RMS over the bags of $e_1$, $e_2$, $e_3$ and of the total error; table B, the signed means of
  $e_1$, $e_2$, $e_3$;
- ``e4.3``: the mean absolute error of the four cells against the realised accuracy, with the mean and the RMS of $x$;
  the same against the pool accuracy $\\sum_j\\pi^Q_jd^{\\text{pool}}_j$, which removes $e_3$.

Rows: one per pair, binary datasets first, then multiclass, each block closed by the mean over its pairs (and, in table
A, by the share of pairs in which each part has the largest RMS). The tables are descriptive: no $^\\dagger$ mark and no
test (deviation from Part V.4 of ``docs/work_plan.md``, stated in card WP-14).

Outputs, under ``<OUT_DIR>/uncawe/tables/error_split/``: a per-bag CSV (``error_split_bags.csv``), a per-pair CSV
(``error_split.csv``), and for each table a LaTeX tabular fragment, a longtable and a standalone landscape A3 document.
"""

import os
import subprocess
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from uncawe import env
from uncawe import error_split as es
from uncawe.config import acc_from_ct
from uncawe.main import EXPERIMENT
from uncawe.pretrain.data import PretrainInfo, load_info_paths
from uncawe.tables import eb_prior
from uncawe.tables import main as main_tables
from uncawe.tables.displacement import ordered
from uncawe.tables.estimand_gap import ACC_NAME, DOMAIN, booktabs_clines, load_results
from uncawe.util import decorate_dataset

TABLE_NAME = "error_split"
KEYS = ["dataset", "classifier"]
METHODS = tuple(m for m in main_tables.MAIN_METHODS if m != "DoC")
IDENTITY_TOL = 1e-9
TABLE_SETS = {"e4.1": ("rms", "signed"), "e4.3": ("oracle_realised", "oracle_pool")}
SHORT_LABELS = {
    "QuAcc_NxN": "QuAcc",
    "O-LEAP": "LEAP",
    "SQS-H": "SQS-H",
    "SQS-Hbs": "SQS-H\\textsubscript{bs}",
    "PQ": "SQS-S",
    "BayesCAP-real": "\\bcap{}",
    "BayesCAP-Hf-real": "Hf",
    "BayCAP-KDEy-mlp-real": "KDEy",
}
PART_LABELS = {"e1": "$e_1$", "e2": "$e_2$", "e3": "$e_3$", "total": "$\\hat A-A$"}
CELL_LABELS = {
    "est_est": "$(\\tilde\\pi,\\tilde d)$",
    "oracle_est": "$(\\pi^Q,\\tilde d)$",
    "est_oracle": "$(\\tilde\\pi,d^{\\mathrm{pool}})$",
    "oracle_oracle": "$(\\pi^Q,d^{\\mathrm{pool}})$",
}
TARGETS = ("realised", "pool")
BLOCKS = ("binary", "multiclass")


# --------------------------------------------------------------------------------------------------- loading


def load_all(methods=METHODS) -> dict[str, pd.DataFrame]:
    """The stored ``main`` results of ``methods`` (vanilla accuracy) on the pairs stored for all of them."""
    results = {}
    for method in methods:
        df = load_results(method)
        if df.empty:
            raise RuntimeError(f"No {method} results found for {EXPERIMENT} ({ACC_NAME}).")
        results[method] = df
    pairs = eb_prior.common_pairs(results)
    if not pairs:
        raise RuntimeError(f"No pair of {EXPERIMENT} has stored results for all of {list(methods)}.")
    index = pd.MultiIndex.from_tuples(pairs, names=KEYS)
    return {
        method: df.loc[pd.MultiIndex.from_frame(df[KEYS]).isin(index)].reset_index(drop=True)
        for method, df in results.items()
    }


def load_pool_info(pairs, domain: str = DOMAIN) -> dict:
    """Return, per (dataset, classifier) pair, the pool recalls $d^{\\text{pool}}$ and $|V|$, from the pretrain files.

    $d^{\\text{pool}}$ compares the argmax of the stored ``U_posteriors`` with the labels of the pool ``U`` of the
    dataset bundle, from which the bags of ``main`` are drawn.
    """
    infos = {}
    for path in load_info_paths(domain=domain):
        info = PretrainInfo.load(path, fast=True)
        infos[(info.d_info.name, info.h_info.name)] = info
    missing = [pair for pair in pairs if pair not in infos]
    if missing:
        raise FileNotFoundError(f"No pretrain info found for {missing}")
    out = {}
    for pair in pairs:
        info = infos[pair]
        bundle = info.load_dataset_bundle()
        _, u_posteriors = info.load_posteriors()
        if len(u_posteriors) != len(bundle.U.y):
            raise ValueError(f"{pair}: {len(u_posteriors)} pool posteriors for {len(bundle.U.y)} pool labels")
        n_classes = int(info.d_info.n_classes)
        d_pool = es.pool_recalls(bundle.U.y, np.argmax(u_posteriors, axis=-1), n_classes)
        out[pair] = {"d_pool": d_pool, "n_val": len(bundle.V.y), "n_pool": len(bundle.U.y)}
    return out


# --------------------------------------------------------------------------------------------------- computation


def _kept(df: pd.DataFrame) -> np.ndarray:
    """Mask of the bags with a finite mean table and no NUTS failure."""
    finite = np.array([np.all(np.isfinite(np.asarray(ct, dtype=float))) for ct in df["estim_cts"]])
    if "nuts_failed" in df.columns:
        finite &= pd.to_numeric(df["nuts_failed"]).fillna(0).to_numpy() != 1
    return finite


def method_bags(df: pd.DataFrame, d_pool: np.ndarray) -> pd.DataFrame:
    """The split of the kept bags of one method on one pair, one row per bag (``bag``: position in the stored order).

    ``identity_gap`` compares $e_1+e_2+e_3$ with the stored ``estim_accs`` minus the realised accuracy, so that it also
    checks that the stored point estimate is the trace of the stored mean table.
    """
    kept = _kept(df)
    sub = df.loc[kept]
    if sub.empty:
        return pd.DataFrame({column: [] for column in ("bag", *es.SPLIT_KEYS)})
    out = es.split(np.stack(sub["estim_cts"].to_numpy()), np.stack(sub["true_cts"].to_numpy()), d_pool)
    true_acc = np.array([acc_from_ct(ACC_NAME, np.asarray(ct, dtype=float)) for ct in sub["true_cts"]])
    stored_error = pd.to_numeric(sub["estim_accs"]).to_numpy() - true_acc
    out["identity_gap"] = np.abs(out["e1"] + out["e2"] + out["e3"] - stored_error)
    bags = pd.DataFrame(out)
    bags.insert(0, "bag", np.flatnonzero(kept))
    return bags


def _check_paired(results: dict[str, pd.DataFrame], key) -> np.ndarray:
    """Return the realised tables of the pair, after checking that every method has the same bags."""
    tables = {m: np.stack(df["true_cts"].to_numpy()) for m, df in results.items()}
    first_method, first = next(iter(tables.items()))
    for method, other in tables.items():
        if other.shape != first.shape or not np.allclose(other, first):
            raise ValueError(f"The bags of {key} differ between {first_method} and {method}")
    return first


def pair_summary(bags: dict[str, pd.DataFrame], e3_all: np.ndarray, n_stored: int) -> dict:
    """The per-pair columns of the tables, from the per-bag splits of each method (``bags``) and the $e_3$ of all bags.

    The oracle errors of each cell are the absolute errors of the cell's estimate, against the realised accuracy and
    against the pool accuracy.
    """
    row = {
        "n_bags": n_stored,
        "e3_rms": es.rms(e3_all),
        "e3_mean": float(np.mean(e3_all)),
    }
    for method, b in bags.items():
        row[f"{method}_n_dropped"] = n_stored - len(b)
        row[f"{method}_n_empty"] = int((b["n_empty"] > 0).sum()) if len(b) else 0
        row[f"{method}_max_identity_gap"] = float(b["identity_gap"].max()) if len(b) else np.nan
        for part in ("e1", "e2", "total"):
            row[f"{method}_{part}_rms"] = es.rms(b[part])
        for part in ("e1", "e2", "e3", "total"):
            row[f"{method}_{part}_mean"] = float(b[part].mean()) if len(b) else np.nan
        row[f"{method}_e3_rms_kept"] = es.rms(b["e3"])
        row[f"{method}_largest"] = es.largest_part(
            (row[f"{method}_e1_rms"], row[f"{method}_e2_rms"], row[f"{method}_e3_rms_kept"])
        )
        row[f"{method}_x_mean"] = float(b["x"].mean()) if len(b) else np.nan
        row[f"{method}_x_rms"] = es.rms(b["x"])
        for cell in es.CELLS:
            estimate = b[f"acc_{cell}"].to_numpy()
            row[f"{method}_{cell}_ae_realised"] = float(np.mean(np.abs(estimate - b["true_acc"]))) if len(b) else np.nan
            row[f"{method}_{cell}_ae_pool"] = float(np.mean(np.abs(estimate - b["pool_acc"]))) if len(b) else np.nan
    # the cell (pi^Q, d^pool) uses no estimate: on all bags its error against A is e3, against the pool accuracy 0
    row["oracle_oracle_ae_realised"] = float(np.mean(np.abs(e3_all)))
    row["oracle_oracle_ae_pool"] = 0.0
    return row


def compute(results: dict[str, pd.DataFrame] | None = None, pool: dict | None = None):
    """Return the per-bag table (every method, kept bags) and the ordered per-pair table."""
    results = load_all() if results is None else results
    groups = {method: dict(list(df.groupby(KEYS, sort=False))) for method, df in results.items()}
    first = next(iter(groups.values()))
    pairs = list(first)
    pool = load_pool_info(pairs) if pool is None else pool
    bag_frames, rows = [], []
    for key in pairs:
        pair_results = {method: groups[method][key].reset_index(drop=True) for method in results}
        true_cts = _check_paired(pair_results, key)
        d_pool = pool[key]["d_pool"]
        e3_all = es.split(true_cts, true_cts, d_pool)["e3"]
        bags = {method: method_bags(df, d_pool) for method, df in pair_results.items()}
        n_classes = int(first[key]["n_classes"].iloc[0])
        row = {"dataset": key[0], "classifier": key[1], "n_classes": n_classes, "n_val": pool[key]["n_val"]}
        row |= pair_summary(bags, e3_all, len(true_cts))
        rows.append(row)
        for method, b in bags.items():
            bag_frames.append(b.assign(dataset=key[0], classifier=key[1], n_classes=n_classes, method=method))
    bag_table = pd.concat(bag_frames, ignore_index=True)
    lead = ["dataset", "classifier", "n_classes", "method", "bag"]
    bag_table = bag_table[lead + [c for c in bag_table.columns if c not in lead]]
    return bag_table, ordered(pd.DataFrame(rows))


def block_of(n_classes: int) -> str:
    return "binary" if n_classes == 2 else "multiclass"


def largest_shares(table: pd.DataFrame, methods=METHODS) -> dict[str, np.ndarray]:
    """Per method, the share of the pairs of ``table`` in which part $1$, $2$ and $3$ has the largest RMS."""
    return {m: np.array([(table[f"{m}_largest"] == k).mean() for k in (1, 2, 3)]) for m in methods}


def analysis_summary(table: pd.DataFrame, methods=METHODS) -> pd.DataFrame:
    """Per block and method: the mean over the pairs of the RMS of each part and of the total, the shares of the
    largest part, the mean AE of the four cells against the realised accuracy, and the dropped and empty-column bags."""
    rows = []
    for block, part in table.groupby(table["n_classes"].map(block_of), sort=False):
        shares = largest_shares(part, methods)
        for m in methods:
            record = {"block": block, "method": m, "pairs": len(part)}
            for p in ("e1", "e2", "total"):
                record[f"{p}_rms"] = part[f"{m}_{p}_rms"].mean()
            record["e3_rms"] = part[f"{m}_e3_rms_kept"].mean()
            for k, share in zip((1, 2, 3), shares[m]):
                record[f"largest_{k}"] = share
            for cell in es.CELLS:
                record[f"{cell}_ae"] = part[f"{m}_{cell}_ae_realised"].mean()
            record["x_rms"] = part[f"{m}_x_rms"].mean()
            record["n_dropped"] = int(part[f"{m}_n_dropped"].sum())
            record["n_empty"] = int(part[f"{m}_n_empty"].sum())
            rows.append(record)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------------- tables


def _fmt(value) -> str:
    if isinstance(value, str):
        return value
    return "--" if value is None or np.isnan(value) else f"{value:.3f}"


def _layout_rows(table: pd.DataFrame, share_row: bool):
    """Yield ``(kind, label, frame)`` in table order: one ``pair`` per pair, then per block a ``mean`` row and, with
    ``share_row``, a ``share`` row; ``frame`` holds the pair's row or the pairs of the block."""
    blocks = table["n_classes"].map(block_of)
    for block in BLOCKS:
        part = table.loc[blocks == block]
        if part.empty:
            continue
        for _, row in part.iterrows():
            yield "pair", (row["dataset"], int(row["n_classes"]), row["classifier"]), row.to_frame().T
        yield "mean", (f"mean, {block}", "", ""), part
        if share_row:
            yield "share", (f"largest part, {block}", "", ""), part


def _index(labels) -> pd.MultiIndex:
    names = [
        decorate_dataset(main_tables.dataset_map.get(d, d)) if k != "" else f"\\textit{{{d}}}" for d, k, _ in labels
    ]
    return pd.MultiIndex.from_arrays(
        [names, [k for _, k, _ in labels], [c for _, _, c in labels]], names=["dataset", "$K$", "classifier"]
    )


def _group_styles(num: pd.DataFrame, groups: list[str], kinds: list[str]) -> pd.DataFrame:
    """For each ``pair`` and ``mean`` row and each group: a colour scale from the worst (red) to the best (green) value
    of the row in the group, lower is better, as ``tables/baycap_fixed.py:_score_cell_styles``; the best in bold."""
    styles = pd.DataFrame("", index=num.index, columns=num.columns)
    for group in groups:
        columns = [c for c in num.columns if c[0] == group]
        for i, kind in enumerate(kinds):
            if kind == "share":
                continue
            quality = -num[columns].iloc[i].to_numpy(dtype=float)
            if np.all(np.isnan(quality)):
                continue
            best, worst = np.nanmax(quality), np.nanmin(quality)
            for column, value in zip(columns, quality):
                if np.isnan(value):
                    continue
                color = main_tables._gradient_hex(value, worst, best).lstrip("#").upper()
                style = f"cellcolor:[HTML]{{{color}}};"
                if np.isclose(value, best):
                    style += " bfseries: ;"
                styles.iloc[i, num.columns.get_loc(column)] = style
    return styles


def _styler(num: pd.DataFrame, disp: pd.DataFrame, labels, kinds, coloured: list[str]):
    num.columns = disp.columns = pd.MultiIndex.from_tuples(num.columns)
    num.index = disp.index = _index(labels)
    styles = _group_styles(num, coloured, kinds) if coloured else pd.DataFrame("", index=num.index, columns=num.columns)
    return disp.style.format(_fmt).apply(lambda _: styles, axis=None)


def _mean(frame: pd.DataFrame, column: str) -> float:
    return float(pd.to_numeric(frame[column]).mean())


def rms_styler(table: pd.DataFrame, methods=METHODS):
    """E4.1 table A: RMS of $e_1$, $e_2$ (per method), $e_3$ (one column), total (per method, with the superscript of
    the part with the largest RMS)."""
    num, disp, labels, kinds = {}, {}, [], []
    for kind, label, frame in _layout_rows(table, share_row=True):
        labels.append(label)
        kinds.append(kind)
        shares = largest_shares(frame, methods) if kind == "share" else None
        for part in ("e1", "e2"):
            for m in methods:
                col = (PART_LABELS[part], SHORT_LABELS[m])
                value = np.nan if kind == "share" else _mean(frame, f"{m}_{part}_rms")
                num.setdefault(col, []).append(value)
                disp.setdefault(col, []).append(f"{100 * shares[m][int(part[1]) - 1]:.0f}\\%" if shares else value)
        col = (PART_LABELS["e3"], "")
        value = np.nan if kind == "share" else _mean(frame, "e3_rms")
        num.setdefault(col, []).append(value)
        disp.setdefault(col, []).append("" if kind == "share" else value)
        for m in methods:
            col = (PART_LABELS["total"], SHORT_LABELS[m])
            if kind == "share":
                num.setdefault(col, []).append(np.nan)
                disp.setdefault(col, []).append(f"{100 * shares[m][2]:.0f}\\%")
                continue
            value = _mean(frame, f"{m}_total_rms")
            if kind == "pair":
                largest = int(frame[f"{m}_largest"].iloc[0])
            else:
                largest = es.largest_part([_mean(frame, f"{m}_{p}") for p in ("e1_rms", "e2_rms", "e3_rms_kept")])
            num.setdefault(col, []).append(value)
            disp.setdefault(col, []).append(f"{_fmt(value)}$^{{{largest}}}$" if largest else _fmt(value))
    coloured = [PART_LABELS[p] for p in ("e1", "e2", "total")]
    return _styler(pd.DataFrame(num), pd.DataFrame(disp), labels, kinds, coloured)


def signed_styler(table: pd.DataFrame, methods=METHODS):
    """E4.1 table B: the signed means of $e_1$, $e_2$ (per method) and $e_3$ (one column), not coloured."""
    num, labels, kinds = {}, [], []
    for kind, label, frame in _layout_rows(table, share_row=False):
        labels.append(label)
        kinds.append(kind)
        for part in ("e1", "e2"):
            for m in methods:
                num.setdefault((PART_LABELS[part], SHORT_LABELS[m]), []).append(_mean(frame, f"{m}_{part}_mean"))
        num.setdefault((PART_LABELS["e3"], ""), []).append(_mean(frame, "e3_mean"))
    num = pd.DataFrame(num)
    return _styler(num, num.copy(), labels, kinds, [])


def oracle_styler(table: pd.DataFrame, target: str, methods=METHODS):
    """E4.3: mean AE of the four cells against ``target`` (``realised`` or ``pool``), per method; the cell
    $(\\pi^Q,d^{\\text{pool}})$ in one column; the mean and the RMS of $x$ per method, not coloured."""
    num, labels, kinds = {}, [], []
    for kind, label, frame in _layout_rows(table, share_row=False):
        labels.append(label)
        kinds.append(kind)
        for cell in es.CELLS[:3]:
            for m in methods:
                num.setdefault((CELL_LABELS[cell], SHORT_LABELS[m]), []).append(_mean(frame, f"{m}_{cell}_ae_{target}"))
        num.setdefault((CELL_LABELS["oracle_oracle"], ""), []).append(_mean(frame, f"oracle_oracle_ae_{target}"))
        for stat in ("mean", "rms"):
            for m in methods:
                header = f"{SHORT_LABELS[m]} {'mean' if stat == 'mean' else 'RMS'}"
                num.setdefault(("$x$", header), []).append(_mean(frame, f"{m}_x_{stat}"))
    num = pd.DataFrame(num)
    coloured = [CELL_LABELS[c] for c in es.CELLS[:3]]
    return _styler(num, num.copy(), labels, kinds, coloured)


COMMON_CAPTION = (
    "Error split of the main-table methods on \\texttt{main} (E4.1, E4.3; vanilla accuracy, $|U|=1000$), against the "
    "realised table of each bag. $e_1=\\sum_j(\\tilde\\pi_j-\\pi^Q_j)\\tilde d_j$ (prevalence), "
    "$e_2=\\sum_j\\pi^Q_j(\\tilde d_j-d^{\\mathrm{pool}}_j)$ (recalls against the test pool), "
    "$e_3=\\sum_j\\pi^Q_j(d^{\\mathrm{pool}}_j-c_j/n_j)$ (luck of the bag), with $e_1+e_2+e_3=\\hat A-A$; $e_3$ uses no "
    "estimate and is the same for every method. Methods: QuAcc, LEAP, SQS-H, SQS-S, \\bcap{} (\\bcap{}-real), Hf "
    "(\\bcap{}-Hf-real), KDEy (\\bcap{}-KDEy-mlp-real). Bags where NUTS failed are dropped. "
)
COLOUR_NOTE = (
    "In each row and coloured group, colours run from the worst (red) to the best (green) method and the best is in "
    "bold; lower is better. "
)
CAPTIONS = {
    "rms": COMMON_CAPTION
    + "Table A: root mean square over the bags of each part and of the total error $\\hat A-A$; the superscript of "
    "the total names the part with the largest RMS. Rows \\textit{largest part}: share of the pairs of the block in "
    "which $e_1$, $e_2$ (under their groups) and $e_3$ (under the total) has the largest RMS. "
    + COLOUR_NOTE
    + "The mean squares of the parts do not add up to that of the total: "
    "$\\overline{(e_1+e_2+e_3)^2}=\\overline{e_1^2}+\\overline{e_2^2}+\\overline{e_3^2}"
    "+2(\\overline{e_1e_2}+\\overline{e_1e_3}+\\overline{e_2e_3})$, and the cross products can be negative.",
    "signed": COMMON_CAPTION
    + "Table B: mean over the bags of each part (a mean far from $0$ is a systematic error); not coloured.",
    "oracle_realised": COMMON_CAPTION
    + "Oracle $2\\times2$ (E4.3): mean absolute error against the realised accuracy of $\\sum_j\\pi_jd_j$, with "
    "$\\pi\\in\\{\\tilde\\pi,\\pi^Q\\}$ and $d\\in\\{\\tilde d,d^{\\mathrm{pool}}\\}$; the three cells with $\\pi^Q$ or "
    "$d^{\\mathrm{pool}}$ are references, not methods; $(\\pi^Q,d^{\\mathrm{pool}})$ uses no estimate (one column). "
    "$x=\\sum_j(\\tilde\\pi_j-\\pi^Q_j)(\\tilde d_j-d^{\\mathrm{pool}}_j)$, the interaction: mean and RMS over the bags. "
    + COLOUR_NOTE,
    "oracle_pool": COMMON_CAPTION
    + "Oracle $2\\times2$ (E4.3) as in the previous table, with every cell scored against the pool accuracy "
    "$\\sum_j\\pi^Q_jd^{\\mathrm{pool}}_j$ instead of the realised accuracy, which removes $e_3$ (the cell "
    "$(\\pi^Q,d^{\\mathrm{pool}})$ is then $0$). " + COLOUR_NOTE,
}


def stylers(table: pd.DataFrame, names) -> dict:
    builders = {
        "rms": lambda: rms_styler(table),
        "signed": lambda: signed_styler(table),
        "oracle_realised": lambda: oracle_styler(table, "realised"),
        "oracle_pool": lambda: oracle_styler(table, "pool"),
    }
    return {name: builders[name]() for name in names}


def table_names(sets) -> list[str]:
    return [name for s in sets for name in TABLE_SETS[s]]


def table_dir(pargs=None) -> str:
    if pargs is not None and pargs.table_dir is not None:
        return pargs.table_dir
    return os.path.join(env.root_dir, "tables", TABLE_NAME)


def _stem(name: str) -> str:
    return f"{TABLE_NAME}_{name}"


def _latex_document(longtable_path: str) -> str:
    document = eb_prior._latex_document(longtable_path).replace(
        "\\begin{document}",
        "\\usepackage[table]{xcolor}\n\\newcommand{\\bcap}{BayCAP}\n\\begin{document}",
    )
    return document.replace("[a4paper,", "[a3paper,")


def check_summary(table: pd.DataFrame, methods=METHODS) -> pd.DataFrame:
    """Per method: the largest identity gap over the pairs, the pairs above ``IDENTITY_TOL``, and the dropped and
    empty-column bags in total and on how many pairs."""
    rows = []
    for m in methods:
        gaps = table[f"{m}_max_identity_gap"]
        rows.append(
            {
                "method": m,
                "max_identity_gap": float(gaps.max()),
                "pairs_above_tol": int((gaps > IDENTITY_TOL).sum()),
                "dropped_bags": int(table[f"{m}_n_dropped"].sum()),
                "pairs_with_dropped": int((table[f"{m}_n_dropped"] > 0).sum()),
                "empty_column_bags": int(table[f"{m}_n_empty"].sum()),
                "pairs_with_empty": int((table[f"{m}_n_empty"] > 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def gen_tables(pargs):
    bag_table, table = compute()
    out_dir = table_dir(pargs)
    tabular_dir = os.path.join(out_dir, "tabular")
    os.makedirs(tabular_dir, exist_ok=True)
    bag_table.to_csv(os.path.join(out_dir, f"{TABLE_NAME}_bags.csv"), index=False)
    table.to_csv(os.path.join(out_dir, f"{TABLE_NAME}.csv"), index=False)

    for name, styler in stylers(table, table_names(pargs.tables_set)).items():
        label = f"tab:{TABLE_NAME}-{name}"
        with open(os.path.join(tabular_dir, f"{_stem(name)}.tex"), "w") as f:
            f.write(booktabs_clines(styler.to_latex(hrules=True, multicol_align="c", clines="skip-last;data")))
        with open(os.path.join(tabular_dir, f"{_stem(name)}_long.tex"), "w") as f:
            f.write(eb_prior._latex_longtable(styler, CAPTIONS[name], label))
        with open(os.path.join(out_dir, f"{_stem(name)}_doc.tex"), "w") as f:
            f.write(_latex_document(f"tabular/{_stem(name)}_long.tex"))

    checks = check_summary(table)
    with pd.option_context("display.width", 250, "display.max_columns", None, "display.precision", 4):
        print(f"\nsummary per block and method:\n{analysis_summary(table).to_string(index=False)}")
        print(f"\nchecks:\n{checks.to_string(index=False)}")
    if (checks["pairs_above_tol"] > 0).any():
        print(f"WARNING: identity gap above {IDENTITY_TOL:g} on some pairs (see checks)")
    print(f"pairs: {len(table)}; tables written to {out_dir}")


def gen_pdf(pargs):
    out_dir = table_dir(pargs)
    for name in table_names(pargs.tables_set):
        tex_filename = f"{_stem(name)}_doc.tex"
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
    parser = ArgumentParser(description="The error split and the oracle 2x2 of the main-table methods (E4.1, E4.3).")
    parser.add_argument("-t", "--tables", action="store_true", help="Generate tables")
    parser.add_argument("-p", "--pdf", action="store_true", help="Generate PDF from tables")
    parser.add_argument("-a", "--all", action="store_true", help="Generate both tables and PDF")
    parser.add_argument(
        "--tables-set",
        nargs="+",
        choices=tuple(TABLE_SETS),
        default=list(TABLE_SETS),
        help="Tables to write: e4.1 (RMS and signed means), e4.3 (oracle 2x2); default: both",
    )
    parser.add_argument(
        "--table-dir",
        "--output-dir",
        dest="table_dir",
        default=None,
        help="Directory where the CSVs, table documents and tabular fragments are saved",
    )
    pargs = parser.parse_args()

    if pargs.tables or pargs.all:
        gen_tables(pargs)
    if pargs.pdf or pargs.all:
        gen_pdf(pargs)
