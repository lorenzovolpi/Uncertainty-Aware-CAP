"""Pretrain the classifiers of ``main`` on the binary datasets of ``quapy`` that ``main`` does not use (work-plan card
WP-7, E2.5 of ``docs/ext2_baycap.md``).

The datasets are the $25$ entries of ``UCI_BINARY_DATASETS`` outside ``main`` (``DATASETS``), split into $L$, $V$ and
$U$ by ``cap-ml``'s ``fetch_UCIBinaryDataset`` as in ``main``; the classifiers are those of
:func:`uncawe.pretrain.classic.gen_classifiers`. They are stored under the pretrain folder of the domain
``classic_binary``, so that ``main`` (domain ``classic``) does not see them.

``--sizes`` prints, without training, the sizes of $L$, $V$ and $U$ and the class counts of each dataset, to check that
the classifiers can be trained on the smallest ones (e.g. ``iris.x``, ``wine.x``).
"""

from argparse import ArgumentParser
from collections.abc import Iterable
from logging import Logger
from traceback import print_exception

from cap.data.datasets import fetch_UCIBinaryDataset
from quapy.data import LabelledCollection

from uncawe.pretrain.classic import pretrain
from uncawe.util import get_logger

EXPERIMENT = "pretrain"
DOMAIN = "classic_binary"
COLLECTION = "uci_binary"
# UCI_BINARY_DATASETS without german, mammographic, semeion, spambase and tictactoe, the binary datasets of main
DATASETS = (
    "transfusion",
    "breast-cancer",
    "wdbc",
    "ionosphere",
    "haberman",
    "spectf",
    "sonar",
    "balance.1",
    "balance.3",
    "cmc.1",
    "cmc.2",
    "cmc.3",
    "ctg.1",
    "ctg.2",
    "ctg.3",
    "iris.1",
    "iris.2",
    "iris.3",
    "wine.1",
    "wine.2",
    "wine.3",
    "wine-q-red",
    "wine-q-white",
    "yeast",
    "pageblocks.5",
)


def gen_datasets(
    names: Iterable[str] = DATASETS,
) -> Iterable[tuple[str, str, tuple[LabelledCollection, LabelledCollection, LabelledCollection]]]:
    for name in names:
        yield name, COLLECTION, fetch_UCIBinaryDataset(name)


def print_sizes(names: Iterable[str] = DATASETS):
    print(f"{'dataset':<14} {'|L|':>6} {'|V|':>6} {'|U|':>6}  class counts (L; V; U)")
    for name, _, (L, V, U) in gen_datasets(names):
        counts = "; ".join(" ".join(str(int(c)) for c in split.counts()) for split in (L, V, U))
        print(f"{name:<14} {len(L):>6} {len(V):>6} {len(U):>6}  {counts}")


def main():
    parser = ArgumentParser(description="Pretrain the classifiers of main on the binary datasets of E2.5.")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument("--sizes", action="store_true", help="Print the sizes of the datasets and do not train")
    pargs = parser.parse_args()
    if pargs.sizes:
        print_sizes(pargs.datasets)
        return

    log: Logger = get_logger(id=f"{EXPERIMENT}.{DOMAIN}")
    try:
        log.info("-" * 31 + "  start  " + "-" * 31)
        pretrain(log, datasets=gen_datasets(pargs.datasets), domain=DOMAIN)
    except Exception as e:
        print_exception(e)
        log.warning(e)
    finally:
        log.info("-" * 32 + "  end  " + "-" * 32)


if __name__ == "__main__":
    main()
