"""Classifier-accuracy prediction methods developed in UNCAWE."""

from uncawe.methods.baycap import (
    BayCAP,
    BetaBinomialFit,
    beta_binomial_loglik,
    eb_concentration,
    fit_beta_binomial,
    realised_tables,
)
from uncawe.methods.baycap_kdey import BayCAPKDEy
from uncawe.methods.baycap_mapls import BayCAPMAPLS
from uncawe.methods.bayes_kdey import BayesKDEy
from uncawe.methods.bayes_mapls import BayesMAPLS
from uncawe.methods.bayesian_quantifier import BayesianQuantifier
from uncawe.methods.sqs_b import SQSB

__all__ = [
    "SQSB",
    "BayCAP",
    "BayCAPKDEy",
    "BayCAPMAPLS",
    "BayesKDEy",
    "BayesMAPLS",
    "BayesianQuantifier",
    "BetaBinomialFit",
    "beta_binomial_loglik",
    "eb_concentration",
    "fit_beta_binomial",
    "realised_tables",
]
