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
    # the SQS methods (card WP-36, user's decision of 2026-11-26): the default ones, with KDEy as quantifier, get the
    # plain labels, and the PACC versions of the paper name their quantifier
    "PQ": "\\pq{}-PACC",
    "PQ-KDEy": "\\pq",
    # the BayCAP families (user's decision of 2026-11-26): the default methods, with the temperature calibrated per pair
    # (WP-34), get the plain labels, and their bases at tau=1 are "-untempered"
    "BayesCAP": "\\bcappop{}-untempered",
    "BayesCAP-real": "\\bcapsample{}-untempered",
    "BayesCAP-EB": "\\bcap{}-EB",
    "BayesCAP-EB-real": "\\bcap{}-EB-real",
    "BayesCAP-H-real": "\\bcap{}-H-real",
    "BayesCAP-Hf-real": "\\bcaphf{}-untempered",
    "BayesCAP-Hfn-real": "\\bcap{}-Hfn-real",
    "BayCAP-KDEy-h": "\\bcap{}-KDEy-h",
    "BayCAP-KDEy-h-real": "\\bcap{}-KDEy-h-real",
    # E3.4 (card WP-31): the BayCAP-KDEy of the project from 2026-11-24 to 2026-11-26, when \bcapkdey became
    # BayCAP-KDEy-h-inst-cb-tcal-keep; before 2026-11-24, \bcapkdey was BayCAP-KDEy-mlp-real
    "BayCAP-KDEy-h-inst": "\\bcapkdey{}-h-inst",
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
    "SQS-H": "\\sqsh{}-PACC",
    "SQS-Hbs": "\\sqshbs{}-PACC",
    "SQS-H-KDEy": "\\sqsh",
    "SQS-Hbs-KDEy": "\\sqshbs",
    "SQS-B": "\\sqsb",
    "SQS-B-loo": "\\sqsb{}-loo",
    "BayCAP-KDEy-mlp-real-loo": "\\bcap{}-KDEy-mlp-real-loo",
    "BayCAP-KDEy-mlp-real-cal": "\\bcap{}-KDEy-mlp-real-cal",
    "BayCAP-KDEy-mlp-real-cal-tcal": "\\bcap{}-KDEy-mlp-real-cal-tcal",
    "O-LEAP": "LEAP",
    # the default AC and CBPE use label-shift binning (E8.3; user's decision of 2026-11-26), the LasCal versions (E8.1)
    # name their calibrator
    "AC-LSB": "AC",
    "L-CBPE-LSB": "CBPE",
    "AC": "AC-LasCal",
    "L-CBPE": "CBPE-LasCal",
    # the methods of WP-34 (experiments/methods.py:TCAL_VARIANTS): tau selected per pair and metric inside the fit; the
    # four default BayCAP methods since 2026-11-26, with the plain labels
    "BayesCAP-tcal": "\\bcappop",
    "BayesCAP-real-tcal": "\\bcapsample",
    "BayesCAP-Hf-real-tcal": "\\bcaphf",
    "BayCAP-KDEy-h-inst-cb-tcal-keep": "\\bcapkdey",
    "BayCAP-KDEy-h-inst-cb-tcal-resel": "\\bcapkdey{}-cb-tcal-resel",
    # the test grid of the -keep variant (WP-34): the base at each tau with the bandwidth of V_fit
    "BayCAP-KDEy-h-inst-cb-keep": "\\bcapkdey{}-cb-keep",
    # E5.3 with the temperature chosen per pair on calibration bags, assembled from the stored runs (tables/tcal_methods.py,
    # card WP-33), for the base whose -tcal method is not stored (the -asm columns of the stored ones were removed on
    # 2026-11-26, after the check of WP-34 found them equal to the stored methods)
    "BayCAP-KDEy-h-inst-tcal": "\\bcapkdey{}-h-inst-tcal",
}


# the bases whose -tcal method is stored since WP-34 (experiments/methods.py:TCAL_VARIANTS, checked by the tests)
STORED_TCAL_BASES = ("BayesCAP", "BayesCAP-real", "BayesCAP-Hf-real", "BayCAP-KDEy-h-inst-cb")
# the methods of E5.3 with the temperature chosen per pair on calibration bags, built by tables/tcal_methods.py from the
# stored runs (not stored methods): ``<base>-tcal`` -> ``<base>``, for the bases of E5.3 (TEMPERED_BASES) whose
# ``<base>-tcal`` is not a stored method; from WP-34 to 2026-11-26 also ``<base>-tcal-asm`` for the stored ones, removed
# after the check of WP-34 (scripts/checks/wp34_tcal_check.py) found them equal to the stored methods on every pair
# and metric
TCAL_METHODS = {
    f"{base}-tcal": base
    for base in ("BayesCAP", "BayesCAP-real", "BayesCAP-Hf-real", "BayCAP-KDEy-h-inst", "BayCAP-KDEy-h-inst-cb")
    if base not in STORED_TCAL_BASES
}
