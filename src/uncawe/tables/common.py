# the LaTeX macros used by METHOD_LATEX_LABELS and by the method labels of the other tables; every document that prints
# them defines them with this preamble
METHOD_LATEX_MACROS = (
    "\\newcommand{\\pq}{SQS-S}",
    "\\newcommand{\\sqsh}{SQS-H}",
    "\\newcommand{\\sqshbs}{SQS-H\\textsubscript{bs}}",
    "\\newcommand{\\sqsb}{SQS-B}",
    "\\newcommand{\\bcap}{BayCAP}",
    "\\newcommand{\\bcaphf}{BayCAP-Hf}",
    "\\newcommand{\\bcapkdey}{BayCAP-KDEy}",
)
METHOD_LATEX_PREAMBLE = "\n".join(METHOD_LATEX_MACROS)

METHOD_LATEX_LABELS = {
    "QuAcc_NxN": "QuAcc",
    "PQ": "\\pq",
    "BayesCAP": "\\bcap",
    "BayesCAP-real": "\\bcap{}-real",
    "BayesCAP-EB": "\\bcap{}-EB",
    "BayesCAP-EB-real": "\\bcap{}-EB-real",
    "BayesCAP-H-real": "\\bcap{}-H-real",
    "BayesCAP-Hf-real": "\\bcaphf",
    "BayesCAP-Hfn-real": "\\bcap{}-Hfn-real",
    "BayCAP-KDEy-h": "\\bcap{}-KDEy-h",
    "BayCAP-KDEy-h-real": "\\bcap{}-KDEy-h-real",
    "BayCAP-KDEy-mlp": "\\bcap{}-KDEy-mlp",
    "BayCAP-KDEy-mlp-real": "\\bcapkdey",
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
    "O-LEAP": "LEAP",
}
