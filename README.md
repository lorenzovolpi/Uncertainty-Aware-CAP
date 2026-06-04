# UNCAWE

Questo repository contiene gli esperimenti per valutare metodi di stima con intervalli di confidenza delle prestazioni di un classificatore sotto shift di prevalenza. Il flusso principale e':

1. generare o avere gia' disponibili i file di pretraining dei classificatori;
2. eseguire l'esperimento `main`;
3. rigenerare tabelle LaTeX e plot a partire dai risultati salvati.

Gli output non vengono salvati dentro la repository, ma sotto `cap.env["OUT_DIR"]/uncawe`. Se esiste un file locale `env.json`, `src/uncawe/env.py` lo usa per sovrascrivere la configurazione di `cap` in base all'hostname o alla sezione `global`.

## Setup

Il progetto usa Python `>=3.12` e dipendenze dichiarate in `pyproject.toml`.

```bash
uv sync
```

Gli esempi sotto usano `uv run python -m ...`, che esegue direttamente i moduli Python interessati.

## Pretraining

L'esperimento principale legge metadati e posteriori dei classificatori da:

```text
<CAP_DATA>/pretrain/<domain>/*_info.pkl
<CAP_DATA>/pretrain/<domain>/*_post.npz
```

In codice, il percorso base e' `cap.env["CAP_DATA"]/pretrain`, tramite `uncawe.util.pretrain_basedir()`. Per il dominio `classic` questi file possono essere generati con:

```bash
uv run python -m uncawe.pretrain.classic
```

Il pretraining classic addestra cinque classificatori (`LR`, `kNN`, `SVM`, `MLP`, `RF`) sui dataset UCI selezionati, salva i metadati `*_info.pkl` e i posteriori `*_post.npz`, e salta automaticamente le combinazioni gia' presenti.

## Esperimento principale

Il modulo [src/uncawe/main.py](src/uncawe/main.py) implementa l'esperimento `main`. Per ogni dominio selezionato, carica i `PretrainInfo`, filtra i dataset definiti in `get_filtered_datasets()`, e valuta ogni metodo restituito da `gen_methods()`:

- `DoC`
- `QuAcc_NxN`
- `O-LEAP`
- `L-CBPE`
- `RQBS`
- `PQ`
- `BayesCAP`

Per ciascuna combinazione dataset/classificatore/metodo vengono valutate tre metriche di accuratezza, definite in `get_acc_names()`:

- `vanilla_accuracy`
- `macro-F1`
- `macro-K`

Ogni dataset bundle viene arricchito con `NUM_TEST = 100` campioni di test dal protocollo `UPP`. Lo script calcola, per ogni campione, la contingency table vera, la stima puntuale dell'accuratezza, i campioni dell'intervallo di confidenza, l'errore assoluto (`acc_err`), lo shift rispetto alla prevalenza di training e i tempi medi di train/test.

Per eseguire:

```bash
uv run python -m uncawe.main --classic
uv run python -m uncawe.main --text
uv run python -m uncawe.main --image
```

Il flag del dominio e' obbligatorio: senza `--classic`, `--text` o `--image`, `pargs.domain` resta `None`.

I risultati vengono salvati come Parquet compressi `zstd` in:

```text
<OUT_DIR>/uncawe/main/<domain>/<acc_name>/<dataset>/<classifier>/<method>.parquet
```

Esempio:

```text
<OUT_DIR>/uncawe/main/classic/vanilla_accuracy/ionosphere/LR/PQ.parquet
```

I log dell'esperimento sono scritti in:

```text
<OUT_DIR>/main.log
```

Lo script e' incrementale: se tutti i file attesi per una combinazione esistono gia', quella combinazione viene saltata.

## Tabelle principali

Il modulo [src/uncawe/tables/main.py](src/uncawe/tables/main.py) rigenera tabelle LaTeX a partire dai Parquet di `main`.

Per ogni metrica di accuratezza (`vanilla_accuracy`, `macro-F1`, `macro-K`) carica i risultati da:

```text
<OUT_DIR>/uncawe/main/<domain>/<acc_name>/...
```

Poi costruisce le colonne:

- `acc_err`: errore assoluto tra accuratezza vera e stimata;
- `coverage`: copertura dell'intervallo rispetto alla vera accuratezza;
- `amplitude`: ampiezza dell'intervallo;
- `winkler`: Winkler score dell'intervallo.

La configurazione attuale include in tabella solo:

```python
METRICS = ("acc_err", "winkler")
```

Per ogni riga, i valori sono aggregati per dataset/classificatore/metodo come media e deviazione standard. Le celle migliori sono in grassetto; i marker `\dag` indicano metodi non significativamente diversi dal migliore secondo test di Wilcoxon appaiato con `alpha = 0.01`. I colori sono applicati via `pandas Styler` e convertiti in LaTeX.

Per generare solo i frammenti tabular e il documento LaTeX:

```bash
uv run python -m uncawe.tables.main --classic --tables
```

Per generare anche il PDF, serve `pdflatex` nel `PATH`:

```bash
uv run python -m uncawe.tables.main --classic --all
```

Output di default:

```text
<OUT_DIR>/uncawe/tables/main/main_<domain>.tex
<OUT_DIR>/uncawe/tables/main/tabular/main_<domain>_<acc_name>.tex
<OUT_DIR>/uncawe/tables/main/main_<domain>.pdf
```

E' possibile cambiare directory di output con:

```bash
uv run python -m uncawe.tables.main --classic --tables --table-dir /tmp/uncawe-main-tables
```

Opzioni utili:

- `--metrics acc_err winkler`: limita le metriche, scegliendo tra quelle in `METRICS`;
- `--no-std`: mostra solo le medie, senza deviazione standard;
- `--pdf`: compila il PDF assumendo che i `.tex` siano gia' stati generati;
- `--all`: genera `.tex` e poi compila il PDF.

## Tabelle dei rank

Il modulo [src/uncawe/tables/rank.py](src/uncawe/tables/rank.py) produce tabelle LaTeX con rank medi per classificatore.

Legge gli stessi Parquet dell'esperimento `main`, ma costruisce prima una tabella ridotta con:

- `classifier`
- `dataset`
- `method`
- `acc_err`
- `true_accs`
- `winkler`

Per ogni dataset e classificatore, i metodi vengono ordinati per metrica; per `acc_err` e `winkler` valori piu' bassi sono migliori. I rank sono poi mediati sui dataset per ciascun classificatore. `L-CBPE` e' escluso da queste tabelle tramite `EXCLUDED_METHODS`.

I marker `\dag` indicano metodi non significativamente diversi dal metodo con rank medio migliore, usando Wilcoxon appaiato sui rank per dataset con `alpha = 0.05`.

Comandi:

```bash
uv run python -m uncawe.tables.rank --classic --tables
uv run python -m uncawe.tables.rank --classic --all
```

Output di default:

```text
<OUT_DIR>/uncawe/tables/rank/main_<domain>_rank.tex
<OUT_DIR>/uncawe/tables/rank/tabular/main_<domain>_<acc_name>_rank.tex
<OUT_DIR>/uncawe/tables/rank/main_<domain>_rank.pdf
```

Anche qui si puo' usare `--table-dir`/`--output-dir` per cambiare destinazione, `--metrics` per limitare le metriche, `--pdf` per compilare soltanto e `--all` per generare e compilare.

## Plot amplitude vs coverage

Il modulo [src/uncawe/plots/amplitude_vs_coverage.py](src/uncawe/plots/amplitude_vs_coverage.py) genera un facet plot per confrontare ampiezza media e copertura media degli intervalli.

Il plot usa solo `ACC_NAME = "vanilla_accuracy"` da `src/uncawe/plots/common.py`. I risultati vengono caricati da:

```text
<OUT_DIR>/uncawe/main/<domain>/vanilla_accuracy/...
```

Per ogni riga viene ricostruita la vera accuratezza dalla contingency table e vengono calcolate:

- `coverage`: se l'intervallo contiene il valore vero;
- `amplitude`: ampiezza dell'intervallo.

Poi i valori vengono mediati per `dataset`, `classifier` e `method`. Il grafico usa un facet per metodo, marker diversi per classificatore e linee tratteggiate alle soglie `CONFIDENCE_LEVEL = 0.95` e `1 - CONFIDENCE_LEVEL = 0.05`.

Comando base:

```bash
uv run python -m uncawe.plots.amplitude_vs_coverage --domain classic
```

Opzioni utili:

```bash
uv run python -m uncawe.plots.amplitude_vs_coverage \
  --domain classic \
  --methods DoC QuAcc_NxN O-LEAP RQBS PQ BayesCAP \
  --classifiers LR kNN SVM MLP RF \
  --x-scale linear
```

Output di default:

```text
<OUT_DIR>/uncawe/plots/main/amplitude_vs_coverage/<domain>_vanilla_accuracy.png
<OUT_DIR>/uncawe/plots/main/amplitude_vs_coverage/<domain>_vanilla_accuracy.svg
<OUT_DIR>/uncawe/plots/main/amplitude_vs_coverage/<domain>_vanilla_accuracy.pdf
```

Il suffisso puo' essere esteso con `--add-stem`, ad esempio:

```bash
uv run python -m uncawe.plots.amplitude_vs_coverage --domain classic --add-stem linear-default
```

produce file come:

```text
<OUT_DIR>/uncawe/plots/main/amplitude_vs_coverage/classic_vanilla_accuracy_linear-default.png
```

Per disabilitare alcuni formati:

```bash
uv run python -m uncawe.plots.amplitude_vs_coverage --domain classic --no-svg --no-pdf
```

## Rigenerazione completa

Per il dominio `classic`, una rigenerazione completa tipica e':

```bash
uv run python -m uncawe.pretrain.classic
uv run python -m uncawe.main --classic
uv run python -m uncawe.tables.main --classic --all
uv run python -m uncawe.tables.rank --classic --all
uv run python -m uncawe.plots.amplitude_vs_coverage --domain classic
```

Per rigenerare tabelle e plot non e' necessario rieseguire l'esperimento, a patto che i Parquet in `<OUT_DIR>/uncawe/main/...` siano gia' presenti.
