# the LaTeX macros used by METHOD_LATEX_LABELS and by the method labels of the other tables; every document that prints
# them defines them with this preamble
METHOD_LATEX_MACROS = (
    "\\newcommand{\\pq}{SQS-S}",
    "\\newcommand{\\sqsh}{SQS-H}",
    "\\newcommand{\\sqshbs}{SQS-H\\textsubscript{bs}}",
    "\\newcommand{\\sqsb}{SQS-H\\textsubscript{bay}}",
    "\\newcommand{\\bcap}{BayCAP}",
    "\\newcommand{\\bcappop}{BayCAP-pop}",
    "\\newcommand{\\bcapsample}{BayCAP-sample}",
    "\\newcommand{\\bcaphf}{BayCAP-Hf}",
    "\\newcommand{\\bcapkdey}{BayCAP-KDEy}",
)
METHOD_LATEX_PREAMBLE = "\n".join(METHOD_LATEX_MACROS)

METHOD_LATEX_LABELS = {
    "QuAcc_NxN": "QuAcc",
    "PQ": "\\pq",
    "BayesCAP": "\\bcappop",
    "BayesCAP-real": "\\bcapsample",
    "BayesCAP-EB": "\\bcap{}-EB",
    "BayesCAP-EB-real": "\\bcap{}-EB-real",
    "BayesCAP-H-real": "\\bcap{}-H-real",
    "BayesCAP-Hf-real": "\\bcaphf",
    "BayesCAP-Hfn-real": "\\bcap{}-Hfn-real",
    "BayCAP-KDEy-h": "\\bcap{}-KDEy-h",
    "BayCAP-KDEy-h-real": "\\bcap{}-KDEy-h-real",
    # E3.4 (card WP-31): the BayCAP-KDEy of the project since 2026-11-24; before, \bcapkdey was BayCAP-KDEy-mlp-real
    "BayCAP-KDEy-h-inst": "\\bcapkdey",
    # E3.5 (card WP-33): BayCAP-KDEy with the bandwidth selected for calibrated class probabilities
    "BayCAP-KDEy-h-inst-cb": "\\bcapkdey{}-cb",
    "BayCAP-KDEy-mlp": "\\bcap{}-KDEy-mlp",
    "BayCAP-KDEy-mlp-real": "\\bcap{}-KDEy-mlp-real",
    "BayCAP-EMQ-h": "\\bcap{}-EMQ-h",
    "BayCAP-EMQ-h-real": "\\bcap{}-EMQ-h-real",
    "BayCAP-EMQ-mlp": "\\bcap{}-EMQ-mlp",
    "BayCAP-EMQ-mlp-real": "\\bcap{}-EMQ-mlp-real",
    "BayCAP-KDEy-Hf-h-real": "\\bcap{}-KDEy-Hf-h-real",
    "BayCAP-KDEy-Hf-mlp-real": "\\bcap{}-KDEy-Hf-mlp-real",
    "SQS-H": "\\sqsh",
    "SQS-Hbs": "\\sqshbs",
    "SQS-B": "\\sqsb",
    "SQS-B-loo": "\\sqsb{}-loo",
    "BayCAP-KDEy-mlp-real-loo": "\\bcapkdey{}-loo",
    "BayCAP-KDEy-mlp-real-cal": "\\bcapkdey{}-cal",
    "BayCAP-KDEy-mlp-real-cal-tcal": "\\bcapkdey{}-cal-tcal",
    "O-LEAP": "LEAP",
    # E5.3 with the temperature chosen per pair on calibration bags (tables/tcal_methods.py, card WP-33)
    "BayesCAP-Hf-real-tcal": "\\bcaphf{}-tcal",
    "BayCAP-KDEy-h-inst-tcal": "\\bcapkdey{}-tcal",
    "BayCAP-KDEy-h-inst-cb-tcal": "\\bcapkdey{}-cb-tcal",
}


# the methods of E5.3 with the temperature chosen per pair on calibration bags, built by tables/tcal_methods.py from the
# stored runs (not stored methods): ``<base>-tcal`` -> ``<base>``, for the bases of E5.3 (TEMPERED_BASES)
TCAL_METHODS = {f"{base}-tcal": base for base in ("BayesCAP-Hf-real", "BayCAP-KDEy-h-inst", "BayCAP-KDEy-h-inst-cb")}
