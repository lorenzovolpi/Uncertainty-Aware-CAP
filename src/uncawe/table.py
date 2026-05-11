import os
import pickle
from argparse import ArgumentParser

import numpy as np
import pandas as pd

import uncawe.env as env
from uncawe.config import acc_from_ct, get_acc_names, get_method_names
from uncawe.main import EXPERIMENT
from uncawe.results import Results
from uncawe.util import decorate_dataset

METRICS = ("acc_err", "coverage", "amplitude")
LOWER_IS_BETTER = {"acc_err": True, "coverage": False, "amplitude": True}
VALUE_FORMAT = "{:.3f}"

dataset_map = {
    "poker_hand": "poker-hand",
    "hand_digits": "hand-digits",
    "page_block": "page-block",
    "image_seg": "image-seg",
    "stanfordnlp__imdb": "imdb",
    "fancyzhx__yelp_polarity": "yelp-polarity",
    "stanfordnlp__sst2": "sst2",
    "fancyzhx__ag_news": "ag-news",
    "fancyzhx__dbpedia_14": "dbpedia-14",
    "community-datasets__yahoo_answers_topics": "yahoo-answers-topics",
    "yelp_reviews": "yelp-reviews",
    "ag_news-lt": "ag-news-lt",
    "dbpedia_14-lt": "dbpedia-14-lt",
    "ethz__food101": "food101",
}


def get_acc_coverage(true_acc: float, accs_range: np.ndarray) -> float:
    pass


def get_acc_amplitude(accs_range: np.ndarray) -> float:
    pass


def _ordered_columns(methods: list[str]) -> pd.MultiIndex:
    return pd.MultiIndex.from_product([methods, METRICS], names=["method", "metric"])


def _format_mean_std(mean: pd.DataFrame, std: pd.DataFrame) -> pd.DataFrame:
    formatted = pd.DataFrame("", index=mean.index, columns=mean.columns)

    for col in mean.columns:
        formatted[col] = [
            "" if pd.isna(m) else f"{VALUE_FORMAT.format(m)} $\\pm$ {('--' if pd.isna(s) else VALUE_FORMAT.format(s))}"
            for m, s in zip(mean[col], std[col])
        ]

    return formatted


def _rdylgn_hex(value: float, vmin: float, vmax: float) -> str:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

    from matplotlib import colormaps
    from matplotlib.colors import Normalize, rgb2hex

    norm = Normalize(vmin=vmin, vmax=vmax)
    return rgb2hex(colormaps["RdYlGn"](norm(value)))


def _text_color(background: str) -> str:
    red = int(background[1:3], 16) / 255
    green = int(background[3:5], 16) / 255
    blue = int(background[5:7], 16) / 255
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    return "#f1f1f1" if luminance < 0.45 else "#000000"


def _cell_styles(mean: pd.DataFrame) -> pd.DataFrame:
    styles = pd.DataFrame("", index=mean.index, columns=mean.columns)

    for col in mean.columns:
        _, metric = col
        values = mean[col]
        scores = -values if LOWER_IS_BETTER[metric] else values
        vmin = scores.min(skipna=True)
        vmax = scores.max(skipna=True)

        if pd.isna(vmin) or pd.isna(vmax):
            continue

        if np.isclose(vmin, vmax):
            colors = pd.Series("#ffffbf", index=scores.index)
        else:
            colors = scores.apply(lambda value: "" if pd.isna(value) else _rdylgn_hex(value, vmin, vmax))

        for dataset, color in colors.items():
            if color:
                styles.loc[dataset, col] = f"background-color: {color}; color: {_text_color(color)};"

    for metric in METRICS:
        metric_values = mean.xs(metric, axis=1, level="metric")
        best_values = metric_values.min(axis=1) if LOWER_IS_BETTER[metric] else metric_values.max(axis=1)

        for dataset, best_value in best_values.items():
            if pd.isna(best_value):
                continue
            best_mask = np.isclose(metric_values.loc[dataset], best_value)
            for method in metric_values.columns[best_mask]:
                styles.loc[dataset, (method, metric)] += " font-weight: bold;"

    return styles


def _results_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    stats = df.groupby(["dataset", "method"], sort=False)[list(METRICS)].agg(["mean", "std"])
    mean = stats.xs("mean", axis=1, level=1).unstack("method").swaplevel(0, 1, axis=1)
    std = stats.xs("std", axis=1, level=1).unstack("method").swaplevel(0, 1, axis=1)

    methods = [method for method in get_method_names() if method in mean.columns.get_level_values("method")]
    columns = _ordered_columns(methods)
    mean = mean.reindex(columns=columns)
    std = std.reindex(columns=columns)
    table = _format_mean_std(mean, std)

    return table.style.apply(lambda _: _cell_styles(mean), axis=None).to_latex(
        hrules=True,
        clines="skip-last;data",
        convert_css=True,
        caption=caption,
        label=label,
    )


def gen_tables(pargs):
    experiment = EXPERIMENT
    domain = pargs.domain

    main_base_dir = os.path.join(env.root_dir, "main")
    table_dir = os.path.join(env.root_dir, "tables")
    os.makedirs(table_dir, exist_ok=True)

    tbls = []
    accs = get_acc_names()
    for acc in accs:
        name = f"{experiment}_{domain}_{acc}"
        df = (
            Results.load(base_dir=main_base_dir, acc_name=acc, domain=domain)
            .map_column_values("dataset", dataset_map)
            .apply_to_column("dataset", decorate_dataset)
            .apply_to_column("true_cts", lambda ct: acc_from_ct(acc, ct), new_col="true_accs")
            .apply_to_column(["true_accs", "estim_acc_ranges"], get_acc_coverage, new_col="coverage")
            .apply_to_column("estim_acc_ranges", get_acc_amplitude, new_col="amplitude")
            .select_columns(["dataset", "method", "acc_err", "true_accs", "coverage", "amplitude"])
            .df
        )

        tbl = _results_latex_table(
            df,
            caption=f"{acc} results on {domain} datasets.",
            label=f"tab:{experiment}-{domain}-{acc}",
        )
        with open(os.path.join(table_dir, f"{name}.tex"), "w") as f:
            f.write(tbl)

        tbls.append(tbl)
        print(f"table {name} genned")

    pickle_path = os.path.join(table_dir, f"{experiment}_{domain}.pickle")
    with open(pickle_path, "wb") as f:
        pickle.dump(tbls, f)


# def gen_pdf(pargs):
#     experiment = EXPERIMENT
#     domain = pargs.domain
#     table_dir = os.path.join(env.root_dir, "tables")
#     os.makedirs(table_dir, exist_ok=True)
#     pickle_path = os.path.join(table_dir, f"{experiment}_{domain}.pickle")
#     assert os.path.exists(pickle_path), "pickle file does not exist"
#
#     with open(pickle_path, "rb") as f:
#         tbls = pickle.load(f)
#
#     pdf_path = os.path.join(table_dir, f"{experiment}_{domain}.pdf")
#     new_commands = [
#         "\\newcommand{\\leapens}{LEAP-ens}",
#         "\\newcommand{\\leapall}{LEAP-All}",
#         "\\newcommand{\\rqbsall}{RQBS-All}",
#         "\\newcommand{\\pqall}{PQ-All}",
#         "\\newcommand{\\docall}{DoC-All}",
#         "\\newcommand{\\imsall}{IMS-All}",
#         "\\newcommand{\\imslr}{IMS-LR}",
#         "\\newcommand{\\imsknn}{IMS-$k$NN}",
#         "\\newcommand{\\imssvm}{IMS-SVM}",
#         "\\newcommand{\\imsmlp}{IMS-MLP}",
#         "\\newcommand{\\nomslr}{$\\emptyset$-LR}",
#         "\\newcommand{\\nomsknn}{$\\emptyset$-$k$NN}",
#         "\\newcommand{\\nomssvm}{$\\emptyset$-SVM}",
#         "\\newcommand{\\nomstsvm}{$\\emptyset$-TSVM}",
#         "\\newcommand{\\nomsmlp}{$\\emptyset$-MLP}",
#     ]
#     column_alignment = [2, 1, 2], "c"
#     additional_headers = [("oracle", 2), ("IMS", 1), ("TMS", 2)]
#     Table.LatexPDF(
#         pdf_path,
#         tables=tbls,
#         landscape=False,
#         new_commands=new_commands,
#         column_alignment=column_alignment,
#         additional_headers=additional_headers,
#     )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-t", "--tables", action="store_true", help="Generate tables")
    # parser.add_argument("-p", "--pdf", action="store_true", help="Generate PDF from tables")
    parser.add_argument("-a", "--all", action="store_true", help="Generate both tables and PDF")
    parser.add_argument("--text", action="store_const", dest="domain", const="text")
    parser.add_argument("--image", action="store_const", dest="domain", const="image")
    parser.add_argument("--classic", action="store_const", dest="domain", const="classic")
    pargs = parser.parse_args()

    if pargs.domain is None:
        raise ValueError("Please specify a domain.")

    if pargs.tables:
        gen_tables(pargs)
    # if pargs.pdf:
    #     gen_pdf(pargs)
