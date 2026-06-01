import marimo

__generated_with = "0.19.9"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Automatisk klassifisering av dokumenter med LOS-kategorier

    Denne notebooken går gjennom hele prosessen fra rådata til evaluering av klassifisering.

    Hovedideen er:
    - forstå datasettet
    - finne gode kategori-signaler
    - bygge en termbank
    - bruke termene i en LLM-prompt
    - evaluere hvor godt klassifiseringen fungerer
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Denne cellen laster inn alle Python-bibliotekene vi trenger videre i notebooken.

    Vi bruker blant annet:
    - `pandas` for dataanalyse
    - `matplotlib` og `seaborn` for visualisering
    - `requests` for kall mot Elasticsearch og API
    - `spacy` for phrase extraction
    - `sklearn` for splitting og evaluering
    """)
    return


@app.cell
def _():
    import json
    import math
    import os
    import random
    import re
    import time
    from collections import Counter, defaultdict
    from pathlib import Path

    import matplotlib.pyplot as plt
    import pandas as pd
    import requests
    import seaborn as sns
    import spacy
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import StratifiedShuffleSplit

    return (
        Counter,
        Path,
        StratifiedShuffleSplit,
        classification_report,
        confusion_matrix,
        defaultdict,
        json,
        math,
        pd,
        plt,
        random,
        re,
        requests,
        sns,
        spacy,
        time,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Her definerer vi noen sentrale konfigurasjonsverdier som brukes senere.

    Dette gjør det lettere å:
    - se hvilke parametere som faktisk brukes
    - endre dem på ett sted
    - dokumentere baseline-oppsettet tydelig
    """)
    return


@app.cell
def _():
    SIG_TEXT_CONFIG = {
        "size": 50,
        "min_doc_count": 3,
        "background_mode": "match_all",
        "method": "significant_text",
    }

    PROMPT_TOP_N_ES = 15
    PROMPT_TOP_N_SPACY = 12
    PROMPT_TOP_N_HYBRID = 15

    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    from pathlib import Path as _Path
    FIGURES_DIR = _Path("figures")
    FIGURES_DIR.mkdir(exist_ok=True)

    return (
        FIGURES_DIR,
        PROMPT_TOP_N_ES,
        PROMPT_TOP_N_HYBRID,
        PROMPT_TOP_N_SPACY,
        RANDOM_STATE,
        SIG_TEXT_CONFIG,
        TEST_SIZE,
    )


@app.cell(hide_code=True)
def _(SIG_TEXT_CONFIG, TEST_SIZE, mo):
    mo.md(f"""
    Denne cellen viser hvilke hovedparametere som brukes i eksperimentet.

    Nåværende oppsett:
    - Elasticsearch-metode: `{SIG_TEXT_CONFIG["method"]}`
    - antall candidate terms per kategori: `{SIG_TEXT_CONFIG["size"]}`
    - minimum dokumentfrekvens: `{SIG_TEXT_CONFIG["min_doc_count"]}`
    - train/test-splitt: `{round((1 - TEST_SIZE) * 100)} % / {round(TEST_SIZE * 100)} %`

    Dette er viktig for reproduserbarhet og for å kunne sammenligne baseline og forbedret metode.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Denne cellen finner prosjektmappen og leser inn rådatasettet fra `data/dataset.jsonl`.

    Datasettet er utgangspunktet for hele analysen og inneholder LOS-labels på tre nivåer.
    """)
    return


@app.cell
def _(Path, pd):
    if "__file__" in globals():
        PROJECT_ROOT = Path(__file__).resolve().parents[1]
    else:
        PROJECT_ROOT = Path.cwd()

    LOS_FILE = PROJECT_ROOT / "data" / "dataset.jsonl"
    assert LOS_FILE.exists(), f"Fant ikke filen: {LOS_FILE}"

    los = pd.read_json(LOS_FILE, lines=True, encoding="utf-8")
    return LOS_FILE, PROJECT_ROOT, los


@app.cell(hide_code=True)
def _(LOS_FILE, los, mo):
    mo.md(f"""
    Her bekrefter vi at datasettet ble lest inn riktig.

    Filen som ble brukt er:
    `{LOS_FILE}`

    Videre i notebooken jobber vi med `los`, som er hele datasettet som en pandas-tabell.
    Antall rader lest inn: **{los.shape[0]}**
    """)
    return


@app.cell(hide_code=True)
def _(los, mo, pd):
    mo.md("""
    Denne cellen lager en enkel kolonneoversikt.

    Målet er å forstå:
    - hvilke felter som finnes
    - hvilke datatyper de har
    - hvor mye data som faktisk er fylt ut i hver kolonne
    """)
    kolonne_oversikt = pd.DataFrame(
        {
            "Kolonne": los.columns,
            "Datatype": los.dtypes.astype(str),
            "Ikke-null": los.notna().sum().values,
            "Beskrivelse": [
                "Kildeadresse til dokumentet",
                "Dokumentets tittel",
                "Ansvarlig etat eller organisasjon",
                "LOS-klassifisering - nivå 1",
                "LOS-klassifisering - nivå 2",
                "LOS-klassifisering - nivå 3",
                "Referanse til lagret dokument i objektlager",
                "Hele dokumentinnholdet",
                "Ofte brukte ord / metadatafelt",
            ],
        }
    )

    mo.vstack(
        [
            mo.md("## Datasettoversikt"),
            mo.md(f"**Antall rader:** {los.shape[0]}  \n**Antall kolonner:** {los.shape[1]}"),
            kolonne_oversikt,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Nå ser vi på datakvalitet.

    Denne delen svarer på spørsmål som:
    - hvor mange verdier mangler?
    - hvilke kolonner er mest ufullstendige?
    - hvor komplett er label-informasjonen?
    """)
    return


@app.cell
def _(los, pd):
    missing_df = pd.DataFrame(
        {
            "kolonne": los.columns,
            "manglende_verdier": los.isna().sum().values,
            "manglende_verdier_prosent": (los.isna().mean().values * 100).round(2),
        }
    ).sort_values("manglende_verdier_prosent", ascending=False)

    missing_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Denne cellen oppsummerer hvor godt LOS-labelene er fylt ut på nivå 1, nivå 2 og nivå 3.

    Det er spesielt viktig å vite dette før vi begynner å splitte data eller evaluere klassifisering.
    """)
    return


@app.cell
def _(los, pd):
    label_summary = pd.DataFrame(
        {
            "nivå": ["nivå 1", "nivå 2", "nivå 3"],
            "ikke_null": [
                los["nivå 1"].notna().sum(),
                los["nivå 2"].notna().sum(),
                los["nivå 3"].notna().sum(),
            ],
            "andel_ikke_null_prosent": [
                round(los["nivå 1"].notna().mean() * 100, 2),
                round(los["nivå 2"].notna().mean() * 100, 2),
                round(los["nivå 3"].notna().mean() * 100, 2),
            ],
        }
    )

    label_summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Her finner vi alle unike LOS-kategorier som faktisk forekommer i datasettet.

    Dette gir oversikt over klassifikasjonsrommet på nivå 1, nivå 2 og nivå 3.
    """)
    return


@app.cell
def _(los, mo, pd):
    nivaa1_kategorier = pd.DataFrame({"nivå 1": sorted(los["nivå 1"].dropna().unique())})
    nivaa2_kategorier = pd.DataFrame({"nivå 2": sorted(los["nivå 2"].dropna().unique())})
    nivaa3_kategorier = pd.DataFrame({"nivå 3": sorted(los["nivå 3"].dropna().unique())})

    mo.hstack(
        [
            mo.vstack([mo.md("### Nivå 1"), nivaa1_kategorier]),
            mo.vstack([mo.md("### Nivå 2"), nivaa2_kategorier]),
            mo.vstack([mo.md("### Nivå 3"), nivaa3_kategorier]),
        ],
        gap=1,
    )
    return nivaa1_kategorier, nivaa2_kategorier, nivaa3_kategorier


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Denne cellen teller hvor mange dokumenter som finnes i hver nivå-1-kategori.

    Det gir et tidlig bilde av om datasettet er balansert eller om enkelte kategorier dominerer.
    """)
    return


@app.cell
def _(los):
    nivaa1_fordeling = (
        los.dropna(subset=["nivå 1"])
        .groupby("nivå 1")
        .size()
        .reset_index(name="antall_dokumenter")
        .sort_values("antall_dokumenter", ascending=False)
    )

    nivaa1_fordeling
    return (nivaa1_fordeling,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Her visualiserer vi fordelingen av dokumenter per nivå-1-kategori som et stolpediagram.

    Dette gjør det lettere å se klasseubalanse enn bare å lese en tabell.
    """)
    return


@app.cell
def _(FIGURES_DIR, nivaa1_fordeling, plt, sns):
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = sns.color_palette("Blues_d", len(nivaa1_fordeling))
    ax.bar(nivaa1_fordeling["nivå 1"], nivaa1_fordeling["antall_dokumenter"], color=colors)
    ax.set_title("Antall dokumenter per LOS nivå 1-kategori", fontsize=14)
    ax.set_xlabel("LOS nivå 1", fontsize=12)
    ax.set_ylabel("Antall dokumenter", fontsize=12)
    for bar, val in zip(ax.patches, nivaa1_fordeling["antall_dokumenter"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha="center", va="bottom", fontsize=10)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "rq2_class_distribution.pdf", bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "rq2_class_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    LOS er hierarkisk. Derfor lager vi nå en hierarkitabell som viser hvilke kombinasjoner av nivå 1, nivå 2 og nivå 3 som finnes i dataene.

    Dette er nyttig både som datakontroll og for faglig forståelse.
    """)
    return


@app.cell
def _(los):
    hierarki = (
        los[["nivå 1", "nivå 2", "nivå 3"]]
        .dropna(subset=["nivå 1"])
        .drop_duplicates()
        .sort_values(["nivå 1", "nivå 2", "nivå 3"])
    )
    return (hierarki,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Denne cellen lager en dropdown for valg av nivå 1.

    Når man velger en nivå-1-kategori, brukes valget videre til å vise relevante nivå-2-kategorier.
    """)
    return


@app.cell
def _(hierarki, mo):
    nivaa1_valg = mo.ui.dropdown(
        options=sorted(hierarki["nivå 1"].unique()),
        label="Velg nivå 1",
    )
    nivaa1_valg
    return (nivaa1_valg,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Denne cellen lager en dropdown for nivå 2 basert på det valgte nivå 1.

    På den måten får vi en enkel, interaktiv måte å utforske LOS-hierarkiet på.
    """)
    return


@app.cell
def _(hierarki, mo, nivaa1_valg):
    nivaa2_opsjoner = (
        hierarki.loc[hierarki["nivå 1"] == nivaa1_valg.value, "nivå 2"].dropna().unique()
        if nivaa1_valg.value is not None
        else []
    )

    nivaa2_valg = mo.ui.dropdown(
        options=sorted(nivaa2_opsjoner),
        label="Velg nivå 2",
    )
    nivaa2_valg
    return (nivaa2_valg,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Her vises nivå 3-verdiene som hører til valgt kombinasjon av nivå 1 og nivå 2.

    Dette gjør det enklere å kontrollere at hierarkiet ser fornuftig ut i datasettet.
    """)
    return


@app.cell
def _(hierarki, nivaa1_valg, nivaa2_valg, pd):
    nivaa3_tabell = (
        hierarki[
            (hierarki["nivå 1"] == nivaa1_valg.value)
            & (hierarki["nivå 2"] == nivaa2_valg.value)
        ][["nivå 3"]]
        .dropna()
        .drop_duplicates()
        if nivaa2_valg.value is not None
        else pd.DataFrame(columns=["nivå 3"])
    )

    nivaa3_tabell
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Nå beregner vi dokumentlengde i antall ord.

    Dette er nyttig fordi dokumentlengde påvirker:
    - hvor mye tekst som sendes til LLM
    - behov for truncation
    - hvor støyende eller komplekst innholdet kan være
    """)
    return


@app.cell
def _(los):
    los_copy = los.copy()
    los_copy["doc_length"] = los_copy["innhold"].apply(lambda x: len(str(x).split()))
    return (los_copy,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Her plotter vi fordelingen av dokumentlengder som histogram.

    Dette gir et mer intuitivt bilde av om de fleste dokumentene er korte, mellomlange eller svært lange.
    """)
    return


@app.cell
def _(FIGURES_DIR, los_copy, plt, sns):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(los_copy["doc_length"], bins=50, ax=ax, color="steelblue")
    ax.set_title("Distribusjon av dokumentlengder", fontsize=14)
    ax.set_xlabel("Antall ord", fontsize=12)
    ax.set_ylabel("Antall dokumenter", fontsize=12)
    _med = int(los_copy["doc_length"].median())
    ax.axvline(_med, color="red", linestyle="--", linewidth=1.5, label=f"Median: {_med}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "rq2_doc_length_distribution.pdf", bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "rq2_doc_length_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Denne cellen lager en oppsummering av dokumentlengde med min, median, gjennomsnitt og maksimum.

    Det gjør det lettere å beskrive datasettet med noen enkle tall.
    """)
    return


@app.cell
def _(los_copy, pd):
    length_summary = pd.DataFrame(
        {
            "statistikk": ["min", "median", "mean", "max"],
            "verdi": [
                int(los_copy["doc_length"].min()),
                float(los_copy["doc_length"].median()),
                round(float(los_copy["doc_length"].mean()), 2),
                int(los_copy["doc_length"].max()),
            ],
        }
    )
    length_summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Nå lager vi train/test-splitt.

    Vi bruker bare dokumenter som har nivå-1-label, siden nivå 1 er hovedmålet for klassifiseringen.
    Splitten er stratified, slik at kategori-fordelingen blir omtrent lik i train og test.
    """)
    return


@app.cell
def _(RANDOM_STATE, StratifiedShuffleSplit, TEST_SIZE, json, los):
    def to_jsonl(df, path):
        with open(path, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                record = {
                    "url": row["url"],
                    "tittel": str(row["tittel"]),
                    "innhold": str(row["innhold"]),
                    "nivaa1": row["nivå 1"],
                    "nivaa2": row["nivå 2"],
                    "nivaa3": row["nivå 3"],
                    "dokument_id": row["s3_key"],
                    "split": row["split"],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    los_labeled = los.dropna(subset=["nivå 1"]).copy()

    split = StratifiedShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    for train_index, test_index in split.split(los_labeled, los_labeled["nivå 1"]):
        train_set = los_labeled.iloc[train_index].copy()
        test_set = los_labeled.iloc[test_index].copy()

    train_set["split"] = "train"
    test_set["split"] = "test"

    print("Train set shape:", train_set.shape)
    print("Test set shape:", test_set.shape)
    return test_set, to_jsonl, train_set


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Denne cellen skriver train- og test-dataene til JSONL-filer.

    Filene lagres både i:
    - `data/` for lokal bruk
    - `logstash-data/` for videre indeksering i Elasticsearch
    """)
    return


@app.cell
def _(PROJECT_ROOT, test_set, to_jsonl, train_set):
    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(exist_ok=True)

    to_jsonl(train_set, data_dir / "los_train.jsonl")
    to_jsonl(test_set, data_dir / "los_test.jsonl")

    logstash_dir = PROJECT_ROOT / "logstash-data"
    logstash_dir.mkdir(exist_ok=True)

    to_jsonl(train_set, logstash_dir / "dataset_train.jsonl")
    to_jsonl(test_set, logstash_dir / "dataset_test.jsonl")

    print("Skrev train/test JSONL til /data og /logstash-data")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Etter splitting sjekker vi kategori-fordelingen i train og test.

    Dette er en enkel kontroll på at stratified split faktisk ga en rimelig fordeling.
    """)
    return


@app.cell
def _(FIGURES_DIR, pd, plt, sns, test_set, train_set):
    train_dist = train_set.groupby("nivå 1").size().rename("train_count")
    test_dist = test_set.groupby("nivå 1").size().rename("test_count")

    split_distribution = (
        pd.concat([train_dist, test_dist], axis=1)
        .fillna(0)
        .astype(int)
        .reset_index()
        .rename(columns={"nivå 1": "kategori"})
        .sort_values("train_count", ascending=False)
    )

    # ── Train/test split distribution chart ──────────────────────────────────
    import numpy as _np
    _x = _np.arange(len(split_distribution))
    _w = 0.38
    _fig, _ax = plt.subplots(figsize=(12, 5))
    _bars_tr = _ax.bar(_x - _w/2, split_distribution["train_count"], _w,
                       label="Train", color="#4575b4")
    _bars_te = _ax.bar(_x + _w/2, split_distribution["test_count"], _w,
                       label="Test", color="#abd9e9")
    for _bar, _val in zip(list(_bars_tr) + list(_bars_te),
                          list(split_distribution["train_count"]) + list(split_distribution["test_count"])):
        if _val > 0:
            _ax.text(_bar.get_x() + _bar.get_width() / 2,
                     _bar.get_height() + 0.3, str(int(_val)),
                     ha="center", va="bottom", fontsize=9)
    _ax.set_xticks(_x)
    _ax.set_xticklabels(split_distribution["kategori"], rotation=30, ha="right", fontsize=10)
    _ax.set_ylabel("Number of Documents", fontsize=12)
    _ax.set_title("Train / Test Split per LOS Level-1 Category (80/20 stratified)", fontsize=13)
    _ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "rq2_train_test_split.pdf", bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "rq2_train_test_split.png", dpi=150, bbox_inches="tight")
    plt.show()

    split_distribution
    return (split_distribution,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Her setter vi opp tilgang til Elasticsearch og definerer hjelpefunksjoner for søk og aggregasjoner.

    Dette er basen for:
    - henting av kategorier
    - uthenting av significant terms
    - senere henting av dokumenter til evaluering
    """)
    return


@app.cell
def _(SIG_TEXT_CONFIG, json, pd, requests):
    ES_URL = "http://localhost:9200"
    TRAIN_INDEX = "los-documents-train"
    TEST_INDEX = "los-documents-test"


    def es_post(index, body, timeout=30):
        r = requests.post(
            f"{ES_URL}/{index}/_search",
            headers={"Content-Type": "application/json"},
            data=json.dumps(body),
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json()


    def get_categories_from_es(index=TRAIN_INDEX, split_value=None, size=200):
        query = (
            {"match_all": {}}
            if split_value is None
            else {"term": {"split.keyword": split_value}}
        )
        body = {
            "size": 0,
            "query": query,
            "aggs": {"cats": {"terms": {"field": "nivaa1.keyword", "size": size}}},
        }
        res = es_post(index, body)
        return [b["key"] for b in res["aggregations"]["cats"]["buckets"]]


    def significant_terms_for_category(
        level,
        category,
        index=TRAIN_INDEX,
        split_value=None,
        size=None,
        min_doc_count=None,
    ):
        if size is None:
            size = SIG_TEXT_CONFIG["size"]
        if min_doc_count is None:
            min_doc_count = SIG_TEXT_CONFIG["min_doc_count"]

        filters = [{"term": {f"{level}.keyword": category}}]
        if split_value is not None:
            filters.insert(0, {"term": {"split.keyword": split_value}})

        background_filter = {"match_all": {}}
        if split_value is not None:
            background_filter = {
                "bool": {
                    "filter": [{"term": {"split.keyword": split_value}}],
                    "must_not": [{"term": {f"{level}.keyword": category}}],
                }
            }

        query = {
            "size": 0,
            "query": {"bool": {"filter": filters}},
            "aggs": {
                "category_characteristic_terms": {
                    "significant_text": {
                        "field": "innhold",
                        "size": size,
                        "min_doc_count": min_doc_count,
                        "background_filter": background_filter,
                        "mutual_information": {
                            "include_negatives": True,
                            "background_is_superset": True,
                        },
                    }
                }
            },
        }

        res = es_post(index, query)
        buckets = res["aggregations"]["category_characteristic_terms"]["buckets"]

        return pd.DataFrame(
            [
                {
                    f"{level} kategori": category,
                    "ofte brukte begreper": b["key"],
                    "score": round(b["score"], 3),
                    "antall_dokumenter": b["doc_count"],
                }
                for b in buckets
            ]
        )

    return (
        ES_URL,
        TEST_INDEX,
        TRAIN_INDEX,
        get_categories_from_es,
        significant_terms_for_category,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Denne cellen bruker baseline-metoden `significant_text` for å hente candidate terms på nivå 1.

    Tanken er å finne ord som er statistisk overrepresentert i én kategori sammenlignet med resten.
    """)
    return


@app.cell
def _(nivaa1_kategorier, pd, significant_terms_for_category):
    nivaa_1_all_terms = pd.concat(
        [
            significant_terms_for_category("nivaa1", category)
            for category in nivaa1_kategorier["nivå 1"].tolist()
        ],
        ignore_index=True,
    )
    nivaa_1_all_terms
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Her gjør vi det samme for nivå 2.

    Dette brukes ikke direkte i klassifiseringen videre, men er nyttig for analyse og for å følge opp tilbakemeldingene om nivå 2.
    """)
    return


@app.cell
def _(nivaa2_kategorier, pd, significant_terms_for_category):
    nivaa_2_all_terms = pd.concat(
        [
            significant_terms_for_category("nivaa2", category)
            for category in nivaa2_kategorier["nivå 2"].tolist()
        ],
        ignore_index=True,
    )
    nivaa_2_all_terms
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Her gjør vi tilsvarende uthenting av candidate terms for nivå 3.

    Dette gir et mer komplett bilde av hvordan termene ser ut på tvers av LOS-hierarkiet.
    """)
    return


@app.cell
def _(nivaa3_kategorier, pd, significant_terms_for_category):
    nivaa_3_all_terms = pd.concat(
        [
            significant_terms_for_category("nivaa3", category)
            for category in nivaa3_kategorier["nivå 3"].tolist()
        ],
        ignore_index=True,
    )
    nivaa_3_all_terms
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Nå går vi fra ren termuthenting til analyse av overlapp og støy.

    Målet er å forstå:
    - hvilke termer som går igjen i mange kategorier
    - hvilke som er mer kategori-spesifikke
    - hvilke som sannsynligvis bare er støy
    """)
    return


@app.cell
def _(ES_URL, TRAIN_INDEX, pd, requests):
    def candidate_terms_for_category(category, index=TRAIN_INDEX, size=50):
        query = {
            "size": 0,
            "query": {"term": {"nivaa1.keyword": category}},
            "aggs": {
                "terms": {
                    "significant_text": {
                        "field": "innhold",
                        "size": size,
                        "min_doc_count": 3,
                        "background_filter": {"match_all": {}},
                    }
                }
            },
        }

        r = requests.post(
            f"{ES_URL}/{index}/_search",
            headers={"Content-Type": "application/json"},
            json=query,
            timeout=30,
        )
        r.raise_for_status()

        return pd.DataFrame(
            [
                {
                    "kategori": category,
                    "term": b["key"],
                    "score": b["score"],
                    "doc_count": b["doc_count"],
                }
                for b in r.json()["aggregations"]["terms"]["buckets"]
            ]
        )

    return (candidate_terms_for_category,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Denne cellen samler candidate terms for alle nivå-1-kategorier i én tabell.

    Det gjør det mulig å analysere termene på tvers av kategorier i stedet for én kategori av gangen.
    """)
    return


@app.cell
def _(candidate_terms_for_category, nivaa1_kategorier, pd):
    nivaa1_all_terms = pd.concat(
        [candidate_terms_for_category(c) for c in nivaa1_kategorier["nivå 1"]],
        ignore_index=True,
    )
    nivaa1_all_terms
    return (nivaa1_all_terms,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Her teller vi i hvor mange forskjellige kategorier hver Elasticsearch-term forekommer.

    Hvis en term finnes i mange kategorier, er den ofte mindre nyttig som klassifikasjonssignal.
    """)
    return


@app.cell
def _(nivaa1_all_terms):
    term_overlap_es = (
        nivaa1_all_terms.groupby("term")["kategori"]
        .nunique()
        .reset_index(name="num_categories")
        .sort_values("num_categories", ascending=False)
    )

    term_overlap_es
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Denne cellen lager en enkel tokenisering av dokumenttekst på nivå 1.

    Dette er en mer direkte, tekstbasert måte å undersøke overlapp på enn Elasticsearch-termene alene.
    """)
    return


@app.cell
def _(los, re):
    def tokenize(text):
        if not isinstance(text, str):
            return []
        return re.findall(r"[a-zA-ZæøåÆØÅ]{3,}", text.lower())

    tmp = los[["nivå 1", "innhold"]].copy()
    tmp["term"] = tmp["innhold"].apply(tokenize)
    tmp = tmp.explode("term")

    tmp
    return (tmp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Her ser vi hvor mange nivå-1-kategorier hvert råtoken finnes i.

    Dette hjelper oss å skille mellom:
    - generelle språkord
    - mer diskriminerende ord
    """)
    return


@app.cell
def _(tmp):
    term_overlap = (
        tmp.groupby("term")["nivå 1"]
        .nunique()
        .reset_index(name="num_categories")
        .sort_values("num_categories", ascending=False)
    )

    term_overlap.query("num_categories > 1")
    return (term_overlap,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Denne cellen kobler overlapp med hvilke kategorier termene faktisk forekommer i.

    Det gjør det lettere å se konkrete eksempler på språklig overlap mellom kategorier.
    """)
    return


@app.cell
def _(term_overlap, tmp):
    term_categories = (
        tmp.groupby("term")["nivå 1"]
        .apply(lambda x: sorted(set(x)))
        .reset_index(name="kategorier")
    )

    overlap_examples = (
        term_overlap.merge(term_categories, on="term")
        .query("num_categories > 1")
        .head(20)
    )
    overlap_examples
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Nå lager vi en mer samlet termstatistikk for nivå 1.

    For hvert ord ser vi på:
    - hvor mange kategorier det finnes i
    - hvilke kategorier det finnes i
    - hvor ofte det forekommer totalt
    - om det er diskriminerende eller ikke
    """)
    return


@app.cell
def _(tmp):
    term_stats = (
        tmp.groupby("term")
        .agg(
            num_categories=("nivå 1", "nunique"),
            categories=("nivå 1", lambda x: sorted(set(x))),
            total_occurrences=("nivå 1", "size"),
        )
        .reset_index()
    )

    term_stats["is_discriminative"] = term_stats["num_categories"] == 1
    term_stats
    return (term_stats,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Denne cellen viser de mest forekommende diskriminerende termene.

    Dette er et første steg mot å finne ord som faktisk kan være nyttige kategori-signaler.
    """)
    return


@app.cell
def _(term_stats):
    informative_terms = term_stats[term_stats["is_discriminative"]].sort_values(
        "total_occurrences",
        ascending=False,
    )
    informative_terms
    return (informative_terms,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Term Overlap Heatmap

    Viser for hvert par av kategorier: hvor mange termer de deler.
    Høy overlapp mellom to kategorier forklarer presisjonsproblemer (modellen forveksler dem).
    """)
    return


@app.cell
def _(FIGURES_DIR, los, plt, re, sns):
    import numpy as _np
    import pandas as _pd

    def _tokenize(text):
        if not isinstance(text, str):
            return []
        return set(re.findall(r"[a-zA-ZæøåÆØÅ]{4,}", text.lower()))

    _cat_terms = {}
    for _cat, _grp in los.dropna(subset=["nivå 1"]).groupby("nivå 1"):
        _all = set()
        for _t in _grp["innhold"]:
            _all |= _tokenize(_t)
        _cat_terms[_cat] = _all

    _cats = sorted(_cat_terms.keys())
    _overlap_matrix = _np.zeros((len(_cats), len(_cats)), dtype=int)
    for _i, _c1 in enumerate(_cats):
        for _j, _c2 in enumerate(_cats):
            _overlap_matrix[_i, _j] = len(_cat_terms[_c1] & _cat_terms[_c2])

    _overlap_df = _pd.DataFrame(_overlap_matrix, index=_cats, columns=_cats)

    _fig, _ax = plt.subplots(figsize=(9, 7))
    _mask = _np.eye(len(_cats), dtype=bool)  # hide diagonal (self-overlap)
    sns.heatmap(
        _overlap_df, annot=True, fmt="d", cmap="YlOrRd",
        ax=_ax, mask=_mask, linewidths=0.5,
        cbar_kws={"label": "Shared tokens (≥4 chars)"},
    )
    _ax.set_title("Vocabulary Overlap Between LOS Level-1 Categories", fontsize=13)
    _ax.set_xlabel("")
    _ax.set_ylabel("")
    plt.xticks(rotation=35, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "rq2_term_overlap_heatmap.pdf", bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "rq2_term_overlap_heatmap.png", dpi=150, bbox_inches="tight")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Her gjør vi en tilsvarende rå token-overlappsanalyse for nivå 2.

    Dette er lagt inn for å følge opp ønsket om å se nærmere på overlapp også under nivå 1.
    """)
    return


@app.cell
def _(los, re):
    def tokenize_nivaa2(text):
        if not isinstance(text, str):
            return []
        return re.findall(r"[a-zA-ZæøåÆØÅ]{3,}", text.lower())

    tmp_nivaa2 = los.dropna(subset=["nivå 2"])[["nivå 2", "innhold"]].copy()
    tmp_nivaa2["term"] = tmp_nivaa2["innhold"].apply(tokenize_nivaa2)
    tmp_nivaa2 = tmp_nivaa2.explode("term")

    tmp_nivaa2
    return (tmp_nivaa2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Denne cellen teller i hvor mange nivå-2-kategorier hvert ord forekommer.

    På den måten kan vi se om også nivå 2 er preget av høy språklig overlap.
    """)
    return


@app.cell
def _(tmp_nivaa2):
    term_overlap_nivaa2 = (
        tmp_nivaa2.groupby("term")["nivå 2"]
        .nunique()
        .reset_index(name="num_categories")
        .sort_values("num_categories", ascending=False)
    )

    term_overlap_nivaa2.query("num_categories > 1")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Her lager vi en samlet termstatistikk for nivå 2 på samme måte som for nivå 1.

    Dette gjør det mulig å peke på diskriminerende og overlappende termer også på nivå 2.
    """)
    return


@app.cell
def _(tmp_nivaa2):
    term_stats_nivaa2 = (
        tmp_nivaa2.groupby("term")
        .agg(
            num_categories=("nivå 2", "nunique"),
            categories=("nivå 2", lambda x: sorted(set(x))),
            total_occurrences=("nivå 2", "size"),
        )
        .reset_index()
    )
    term_stats_nivaa2["is_discriminative"] = term_stats_nivaa2["num_categories"] == 1
    term_stats_nivaa2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Nå går vi over til spaCy-basert phrase extraction.

    Tanken her er at enkeltord alene ofte er for grove eller for støyende, mens fraser kan gi mer presise signaler.
    """)
    return


@app.cell
def _(spacy):
    def load_spacy_model():
        candidates = [
            "nb_core_news_md",
            "nb_core_news_sm",
            "xx_ent_wiki_sm",
        ]
        for model_name in candidates:
            try:
                nlp = spacy.load(model_name)
                return nlp, model_name
            except Exception:
                continue

        raise ValueError(
            "Fant ingen spaCy-modell. Installer for eksempel: "
            "python -m spacy download nb_core_news_md"
        )

    nlp, spacy_model_name = load_spacy_model()
    print("Loaded spaCy model:", spacy_model_name)
    return (nlp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Her lager vi en egen train-tabell som brukes til term- og phrase extraction.

    Det er viktig at termbanken bygges fra train-settet og ikke fra test, slik at vi unngår lekkasje fra evalueringsdata.
    """)
    return


@app.cell
def _(train_set):
    train_df = train_set.dropna(subset=["nivå 1", "innhold"]).copy()
    train_df["innhold"] = train_df["innhold"].astype(str)
    train_df["nivå 1"] = train_df["nivå 1"].astype(str)

    print("Train rows:", len(train_df))
    print("Categories:", train_df["nivå 1"].nunique())

    train_df[["nivå 1", "tittel"]].head()
    return (train_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Denne cellen definerer en regelbasert kvalitetsfilter-funksjon for termer.

    Den brukes senere til å fjerne ting som:
    - tall
    - URL-er
    - mailto-lenker
    - tekniske strenger
    - åpenbart lite nyttige ord
    """)
    return


@app.cell
def _(re):
    BAD_TERMS = {
        "oslo", "bergen", "torsdag", "mandag", "internett", "steder",
        "artikkel", "last", "skriv", "pdfapi", "api", "printpageaspdfdocument",
        "kontaktoss", "refererer", "ungdom", "unge", "22", "http", "https",
        "www", "mailto", "side", "sider"
    }

    BAD_PATTERNS = [
        r"^\d+$",
        r"^[a-f0-9]{8,}$",
        r".*mailto:.*",
        r".*www\..*",
        r".*\.no.*",
        r".*\.pdf$",
        r".*pdf.*",
        r".*\/.*",
    ]

    def is_good_term(term: str) -> bool:
        t = str(term).strip().lower()

        if not t:
            return False

        if len(t) < 3:
            return False

        if t in BAD_TERMS:
            return False

        for pattern in BAD_PATTERNS:
            if re.match(pattern, t):
                return False

        return True

    return (is_good_term,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Denne cellen definerer hjelpefunksjoner for phrase extraction.

    Vi gjør tre ting:
    - normaliserer tekst
    - rydder bort markdown/støy
    - henter ut kandidat-fraser med spaCy
    """)
    return


@app.cell
def _(re):
    GENERIC_BLACKLIST = {
        "informasjon",
        "les mer",
        "kontakt oss",
        "skjema",
        "side",
        "sider",
        "tjeneste",
        "tjenester",
        "kommune",
        "norge",
        "nav",
        "regelverk",
        "dokument",
        "søknad",
        "rettighet",
        "rettigheter",
        "veiledning",
        "oversikt",
    }

    def normalize_phrase(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\wæøåÆØÅ\- ]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def valid_phrase(text: str) -> bool:
        if not text:
            return False
        if len(text) < 3:
            return False
        if text in GENERIC_BLACKLIST:
            return False

        tokens = text.split()
        if len(tokens) > 5:
            return False
        if all(token.isdigit() for token in tokens):
            return False

        return True

    def clean_markdown(text: str) -> str:
        text = str(text)
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        text = re.sub(r"[*_#>`]", " ", text)
        text = text.replace("•", " ")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def extract_candidate_phrases(text, nlp):
        if text is None:
            return []

        text = clean_markdown(text)
        if not text:
            return []

        doc = nlp(text)
        phrases = []
        current = []

        for token in doc:
            if token.is_space or token.is_punct or token.like_url:
                if current:
                    phrases.append(" ".join(current))
                    current = []
                continue

            if token.is_stop:
                if current:
                    phrases.append(" ".join(current))
                    current = []
                continue

            if token.pos_ in {"NOUN", "PROPN", "ADJ"}:
                token_text = token.lemma_ if token.lemma_ else token.text
                current.append(token_text)
            else:
                if current:
                    phrases.append(" ".join(current))
                    current = []

        if current:
            phrases.append(" ".join(current))

        cleaned = []
        for p in phrases:
            p = normalize_phrase(p)
            if valid_phrase(p):
                cleaned.append(p)

        return list(dict.fromkeys(cleaned))

    return (extract_candidate_phrases,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Nå kjører vi phrase extraction på train-settet.

    Resultatet blir en tabell der hver rad representerer:
    - ett dokument
    - én kategori
    - én frase hentet ut fra dokumentet
    """)
    return


@app.cell
def _(extract_candidate_phrases, nlp, pd, train_df):
    phrase_rows = []

    for idx, row in train_df.iterrows():
        category = row["nivå 1"]
        text = row["innhold"]

        phrases = extract_candidate_phrases(text, nlp) or []

        for phrase in set(phrases):
            phrase_rows.append(
                {
                    "doc_id": idx,
                    "kategori": category,
                    "phrase": phrase,
                }
            )

    phrase_df = pd.DataFrame(phrase_rows)
    print("Extracted phrase rows:", len(phrase_df))
    phrase_df
    return (phrase_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Her scorer vi frasene.

    Scoren prøver å belønne fraser som:
    - finnes i flere dokumenter i samme kategori
    - overlapper lite med andre kategorier
    - gjerne består av mer enn ett ord
    """)
    return


@app.cell
def _(math, phrase_df):
    num_categories = phrase_df["kategori"].nunique()

    in_cat = (
        phrase_df.groupby(["kategori", "phrase"])["doc_id"]
        .nunique()
        .reset_index(name="in_class_df")
    )

    cat_overlap = (
        phrase_df.groupby("phrase")["kategori"]
        .nunique()
        .reset_index(name="num_categories")
    )

    total_df = (
        phrase_df.groupby("phrase")["doc_id"]
        .nunique()
        .reset_index(name="total_doc_freq")
    )

    phrase_scores = (
        in_cat.merge(cat_overlap, on="phrase", how="left")
        .merge(total_df, on="phrase", how="left")
    )

    phrase_scores["cross_category_count"] = phrase_scores["num_categories"] - 1
    phrase_scores["num_tokens"] = phrase_scores["phrase"].str.split().str.len()

    def score_row(row):
        in_class_df = row["in_class_df"]
        overlap = row["cross_category_count"]
        num_tokens = row["num_tokens"]

        exclusivity = math.log(1 + num_categories / (1 + overlap))
        phrase_bonus = 1.15 if num_tokens >= 2 else 1.0

        return in_class_df * exclusivity * phrase_bonus

    phrase_scores["score"] = phrase_scores.apply(score_row, axis=1)

    phrase_scores = phrase_scores.sort_values(
        ["kategori", "score", "in_class_df"],
        ascending=[True, False, False],
    )

    phrase_scores.head(20)
    return (phrase_scores,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Denne cellen filtrerer phrase-scorene til et strengere sett av prompt-termer.

    Vi beholder bare fraser som:
    - finnes i minst to dokumenter i kategorien
    - bare finnes i én kategori
    - har mellom ett og fire ord
    """)
    return


@app.cell
def _(phrase_scores):
    prompt_terms_strict = phrase_scores[
        (phrase_scores["in_class_df"] >= 2)
        & (phrase_scores["num_categories"] == 1)
        & (phrase_scores["num_tokens"].between(1, 4))
    ].copy()

    prompt_terms_strict = prompt_terms_strict.sort_values(
        ["kategori", "score", "in_class_df"],
        ascending=[True, False, False],
    )

    prompt_terms_strict.head(30)
    return (prompt_terms_strict,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Her bygger vi spaCy-termbanken.

    For hver kategori tar vi de best rangerte frasene og lagrer dem som en liste som senere kan brukes direkte i prompten.
    """)
    return


@app.cell
def _(PROMPT_TOP_N_SPACY, defaultdict, prompt_terms_strict):
    def build_prompt_term_bank(df, top_n=12):
        bank = defaultdict(list)

        for category, group in df.groupby("kategori"):
            top_terms = (
                group.sort_values(["score", "in_class_df"], ascending=False)["phrase"]
                .drop_duplicates()
                .head(top_n)
                .tolist()
            )
            bank[category] = top_terms

        return dict(bank)

    prompt_term_bank_spacy = build_prompt_term_bank(prompt_terms_strict, top_n=PROMPT_TOP_N_SPACY)
    prompt_term_bank_spacy
    return (prompt_term_bank_spacy,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Nå bygger vi en Elasticsearch-basert termbank for prompten.

    Dette er fortsatt basert på baseline-metoden, men nå gjør vi det eksplisitt per kategori i et format som kan brukes videre.
    """)
    return


@app.cell
def _(
    PROMPT_TOP_N_ES,
    TRAIN_INDEX,
    get_categories_from_es,
    significant_terms_for_category,
):
    categories = sorted(get_categories_from_es(index=TRAIN_INDEX, split_value=None))

    es_term_bank_raw = {}
    for cat in categories:
        df = significant_terms_for_category(
            level="nivaa1",
            category=cat,
            index=TRAIN_INDEX,
            split_value=None,
            size=PROMPT_TOP_N_ES,
            min_doc_count=2,
        )
        es_term_bank_raw[cat] = df["ofte brukte begreper"].tolist()

    es_term_bank_raw
    return categories, es_term_bank_raw


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Her renser vi termbankene.

    Dette steget er viktig fordi rå candidate terms ofte inneholder:
    - tall
    - tekniske strenger
    - URL-rester
    - metadata
    - andre ord som ikke hjelper en LLM å forstå hovedtemaet
    """)
    return


@app.cell
def _(defaultdict, es_term_bank_raw, is_good_term):
    def clean_term_bank(term_bank):
        cleaned = defaultdict(list)
        removed = []

        for category, terms in term_bank.items():
            seen = set()
            good_terms = []

            for term in terms:
                t = str(term).strip()
                if not is_good_term(t):
                    removed.append({"kategori": category, "term": t, "reason": "quality_filter"})
                    continue

                if t not in seen:
                    seen.add(t)
                    good_terms.append(t)

            cleaned[category] = good_terms

        return dict(cleaned), removed

    es_term_bank, es_removed_rows = clean_term_bank(es_term_bank_raw)
    return clean_term_bank, es_removed_rows, es_term_bank


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Denne cellen bruker den samme rensefunksjonen på spaCy-termbanken.

    På den måten får vi et mer rettferdig sammenligningsgrunnlag mellom ES og spaCy.
    """)
    return


@app.cell
def _(clean_term_bank, prompt_term_bank_spacy):
    prompt_term_bank_spacy_clean, spacy_removed_rows = clean_term_bank(prompt_term_bank_spacy)
    return prompt_term_bank_spacy_clean, spacy_removed_rows


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Nå lager vi en hybrid termbank.

    Den bygges ved å kombinere:
    - rensede Elasticsearch-termer
    - rensede spaCy-termer

    Målet er å få både statistisk sterke ord og mer lesbare fraser inn i samme bank.
    """)
    return


@app.cell
def _(
    PROMPT_TOP_N_HYBRID,
    defaultdict,
    es_term_bank,
    is_good_term,
    prompt_term_bank_spacy_clean,
):
    hybrid_term_bank = defaultdict(list)

    all_categories = sorted(set(es_term_bank.keys()) | set(prompt_term_bank_spacy_clean.keys()))
    for c in all_categories:
        seen = set()
        merged = []

        for term in es_term_bank.get(c, []) + prompt_term_bank_spacy_clean.get(c, []):
            t = str(term).strip()
            if is_good_term(t) and t not in seen:
                seen.add(t)
                merged.append(t)

        hybrid_term_bank[c] = merged[:PROMPT_TOP_N_HYBRID]

    hybrid_term_bank = dict(hybrid_term_bank)
    hybrid_term_bank
    return (hybrid_term_bank,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Denne cellen samler termene som ble fjernet under rensingen.

    Dette fungerer som en enkel kvalitetskontroll og dokumentasjon på at vi faktisk har fjernet støy.
    """)
    return


@app.cell
def _(FIGURES_DIR, es_removed_rows, es_term_bank, es_term_bank_raw, pd, plt, prompt_term_bank_spacy_clean, prompt_term_bank_spacy, spacy_removed_rows, sns):
    removed_terms_df = pd.DataFrame(es_removed_rows + spacy_removed_rows)

    # ── Filtered terms: kept vs removed per category ──────────────────────────
    import numpy as _np

    _rows = []
    for _cat in sorted(set(list(es_term_bank_raw.keys()) + list(prompt_term_bank_spacy.keys()))):
        _es_raw = len(es_term_bank_raw.get(_cat, []))
        _es_kept = len(es_term_bank.get(_cat, []))
        _sp_raw = len(prompt_term_bank_spacy.get(_cat, []))
        _sp_kept = len(prompt_term_bank_spacy_clean.get(_cat, []))
        _rows.append({
            "category": _cat,
            "ES kept": _es_kept,
            "ES removed": _es_raw - _es_kept,
            "spaCy kept": _sp_kept,
            "spaCy removed": _sp_raw - _sp_kept,
        })
    _filt_df = pd.DataFrame(_rows)

    if not _filt_df.empty:
        _x = _np.arange(len(_filt_df))
        _w = 0.2
        _fig, _ax = plt.subplots(figsize=(13, 5))
        _ax.bar(_x - 1.5*_w, _filt_df["ES kept"],     _w, label="ES kept",      color="#2171b5")
        _ax.bar(_x - 0.5*_w, _filt_df["ES removed"],  _w, label="ES removed",   color="#9ecae1", alpha=0.7)
        _ax.bar(_x + 0.5*_w, _filt_df["spaCy kept"],  _w, label="spaCy kept",   color="#238b45")
        _ax.bar(_x + 1.5*_w, _filt_df["spaCy removed"],_w, label="spaCy removed",color="#a1d99b", alpha=0.7)
        _ax.set_xticks(_x)
        _ax.set_xticklabels(_filt_df["category"], rotation=30, ha="right", fontsize=10)
        _ax.set_ylabel("Number of Terms", fontsize=12)
        _ax.set_title("Keyword Terms: Kept vs Removed After Artifact Filtering (per Category)", fontsize=12)
        _ax.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "rq2_terms_kept_vs_removed.pdf", bbox_inches="tight")
        plt.savefig(FIGURES_DIR / "rq2_terms_kept_vs_removed.png", dpi=150, bbox_inches="tight")
        plt.show()

    removed_terms_df.head(50)
    return (removed_terms_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Nå velger vi hvilken termstrategi som skal brukes i den aktive kjøringen.

    Mulige strategier er:
    - `es`
    - `spacy`
    - `hybrid`

    Vi bygger også den ferdige `terms_prompt`-teksten som sendes til modellen.
    """)
    return


@app.cell
def _(es_term_bank, hybrid_term_bank, prompt_term_bank_spacy_clean):
    TERM_STRATEGY = "hybrid"  # "es", "spacy", "hybrid"

    def build_terms_prompt_from_bank(term_bank):
        lines = []
        for category, terms in sorted(term_bank.items()):
            lines.append(f"{category}: " + ", ".join(terms))
        return "\n".join(lines)

    if TERM_STRATEGY == "es":
        selected_bank = es_term_bank
    elif TERM_STRATEGY == "spacy":
        selected_bank = prompt_term_bank_spacy_clean
    else:
        selected_bank = hybrid_term_bank

    terms_prompt = build_terms_prompt_from_bank(selected_bank)
    print("Term strategy:", TERM_STRATEGY)
    print(terms_prompt)
    return TERM_STRATEGY, build_terms_prompt_from_bank, terms_prompt


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    I tillegg til termbanken henter vi ut noen få eksempeldokumenter per kategori fra train-indeksen.

    Disse brukes som ekstra kontekst i prompten, slik at modellen ser konkrete eksempler på hvordan dokumenter i de ulike kategoriene ser ut.
    """)
    return


@app.cell
def _(ES_URL, TRAIN_INDEX, categories, json, requests):
    def es_post_debug(index, body):
        r = requests.post(
            f"{ES_URL}/{index}/_search",
            headers={"Content-Type": "application/json"},
            data=json.dumps(body),
            timeout=30,
        )

        if r.status_code >= 400:
            print("Status:", r.status_code)
            try:
                print(json.dumps(r.json(), ensure_ascii=False)[:5000])
            except Exception:
                print(r.text[:5000])
            r.raise_for_status()

        return r.json()

    def get_examples_for_category(category, index=TRAIN_INDEX, per_class=2, max_chars=1200):
        body = {
            "size": per_class,
            "_source": ["tittel", "innhold", "nivaa1"],
            "query": {"term": {"nivaa1.keyword": category}},
        }
        res = es_post_debug(index, body)
        hits = res["hits"]["hits"]

        out = []
        for h in hits:
            src = h.get("_source", {})
            title = src.get("tittel", "")
            text = (src.get("innhold", "") or "")[:max_chars]
            out.append(f"Title: {title}\nText: {text}\nLabel: {category}\n")
        return out

    def make_example_prompts_from_es(categories, index=TRAIN_INDEX, per_class=2, max_chars=1200):
        parts = []
        for cat in categories:
            parts.extend(
                get_examples_for_category(
                    cat,
                    index=index,
                    per_class=per_class,
                    max_chars=max_chars,
                )
            )
        return "\n---\n".join(parts)

    example_prompts = make_example_prompts_from_es(
        categories,
        index=TRAIN_INDEX,
        per_class=2,
        max_chars=1200,
    )
    return (example_prompts,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Her leser vi API-nøkkel og endpoint fra miljøvariabler.

    Dette er tryggere enn å hardkode nøkler i notebooken.
    Sett gjerne:
    - `AZURE_ANTHROPIC_API_KEY`
    - `AZURE_ANTHROPIC_ENDPOINT`
    før du kjører klassifisering.
    """)
    return


@app.cell
def _():
    import os as _os
    from pathlib import Path as _Path
    from dotenv import load_dotenv as _load_dotenv

    _env_path = _Path(__file__).resolve().parent.parent / ".env"
    if _env_path.exists():
        _load_dotenv(_env_path, override=False)

    api_key = _os.getenv("AZURE_ANTHROPIC_API_KEY", "")
    endpoint = _os.getenv(
        "AZURE_ANTHROPIC_ENDPOINT",
        "https://your-foundry-resource.services.ai.azure.com/anthropic/v1/messages",
    )
    if not api_key:
        print("WARNING: AZURE_ANTHROPIC_API_KEY not set. Classification will be skipped.")
    return api_key, endpoint


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Denne cellen definerer hjelpefunksjoner for LLM-kallet:
    - normalisering av labels
    - truncation av tekst
    - parsing av API-respons
    - hovedfunksjonen som klassifiserer ett dokument

    Prompten som brukes er den detaljerte LOS-prompten du ønsket.
    """)
    return


@app.cell
def _(json, random, re, requests, time):
    def normalize_label(s: str) -> str:
        s = (s or "").strip().lower()
        s = re.sub(r"\s+", " ", s)
        s = s.strip(" .,:;\"'`")
        return s

    def clamp_text(text: str, max_chars: int = 12000) -> str:
        text = (text or "").strip()
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n\n[TRUNCATED]"

    def extract_text_from_anthropic_response(resp_json: dict) -> str:
        content = resp_json.get("content")

        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    return item["text"]
                if isinstance(item, str):
                    return item

        if isinstance(content, str):
            return content

        for k in ("output_text", "completion", "text"):
            v = resp_json.get(k)
            if isinstance(v, str) and v.strip():
                return v

        return ""

    def classify_text_claude(
        title,
        url,
        content,
        terms_prompt,
        categories,
        api_key,
        endpoint,
        few_shot_examples="",
        model="claude-haiku-4-5",
        max_retries=6,
        timeout=30,
    ):
        cats_norm = [normalize_label(c) for c in categories]
        cats_set = set(cats_norm)

        content = clamp_text(content, max_chars=12000)

        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }

        user_prompt = f"""
    Du er en norsk LOS-tag-klassifiserer.

    Oppgaven din er å lese et norsk dokument og velge nøyaktig én LOS-kategori som best beskriver dokumentets hovedtema.

    LOS-kategoriene du kan velge mellom er:
    {", ".join(categories)}

    Hva du skal gjøre:
    Du skal klassifisere dokumentet etter hvilken LOS-kategori det primært tilhører. Du skal ikke bare se på enkeltord, avsender eller URL, men forstå hva dokumentet faktisk handler om, hvilken type behov det dekker, og hvilket samfunnsområde det tilhører.

    Tenk på LOS-kategori som:
    - hvilket livsområde eller tjenesteområde dokumentet hører til
    - hva innbyggeren faktisk trenger hjelp til, informasjon om, eller har rettigheter knyttet til
    - hva som er dokumentets hovedformål

    Viktige regler:
    - Velg nøyaktig én kategori.
    - Velg hovedtema, ikke et side- eller støttetema.
    - Ikke velg kategori bare fordi enkelte ord forekommer ofte.
    - Ikke velg kategori bare fordi avsenderen er for eksempel NAV, Helsenorge, Helsedirektoratet, Udir eller en kommune.
    - Dersom dokumentet dekker flere temaer, velg den kategorien som best beskriver dokumentets viktigste innhold og formål.
    - Bruk tittel, URL og innhold samlet.

    Retningslinjer for kategorisering:
    - Velg "Helse og omsorg" når dokumentet primært handler om helse, diagnose, behandling, helsehjelp, pasientforløp, habilitering, rehabilitering, omsorgstjenester, pleie, oppfølging i helsevesenet, eller rettigheter knyttet til helsetjenester.
    - Velg "Opplæring og utdanning" når dokumentet primært handler om barnehage, skole, SFO, undervisning, opplæring, spesialpedagogikk, PP-tjeneste, læring, utdanningsrettigheter eller tilrettelegging i utdanningsløpet.
    - Velg "Arbeid" når dokumentet primært handler om arbeidsdeltakelse, jobb, arbeidsrettet oppfølging, arbeidsevne, tilrettelegging i arbeidslivet, tiltak for å komme i arbeid eller varig tilrettelagt arbeid.
    - Velg "Sosial og økonomisk trygghet" når dokumentet primært handler om ytelser, stønader, trygd, økonomiske rettigheter, refusjoner, økonomisk støtte, hjelpemidler eller andre støtteordninger.
    - Velg "Kultur-Idrett-Fritid" når dokumentet primært handler om fritidsaktiviteter, kulturtilbud, idrett, deltakelse i aktiviteter, fysisk aktivitet som fritidstilbud, eller organiserte aktivitetstilbud.
    - Velg "Innbygger" når dokumentet primært handler om generelle offentlige tjenester, borgerrettet informasjon, bruk av offentlige digitale tjenester, representasjon, fullmakter eller annen generell innbyggerinformasjon som ikke passer best i de andre kategoriene.

    {f"Eksempler fra treningsdata:{chr(10)}{few_shot_examples}{chr(10)}" if few_shot_examples else ""}
    {f"Kategori-indikative begreper fra treningsdata:{chr(10)}{terms_prompt}{chr(10)}{chr(10)}Bruk begrepene over som støtte til mønstergjenkjenning, men ikke som fasit. Dersom begrepene peker én vei, men innholdets hovedformål peker en annen vei, skal du følge innholdets hovedformål." if terms_prompt else ""}

    Arbeidsmåte:
    1. Lag en kort oppsummering av dokumentet med fokus på LOS-klassifisering.
    2. Beskriv hva dokumentet hovedsakelig hjelper brukeren med.
    3. Vurder hvilke kategorier som er mest sannsynlige.
    4. Velg én endelig LOS-kategori.

    Dokument:
    Tittel: {title}
    URL: {url}
    Innhold:
    {content}

    Returner kun selve LOS-kategorien.
    Ikke skriv oppsummering.
    Ikke skriv begrunnelse.
    Ikke skriv markdown.
    Ikke skriv punktliste.
    Ikke skriv ekstra tekst.

    Gyldige svar er kun én av disse verdiene:
    {", ".join(categories)}
    """

        data = {
            "model": model,
            "system": "Du er en LOS-klassifiserer for norske offentlige dokumenter. Svar med kun én kategori.",
            "messages": [{"role": "user", "content": user_prompt}],
            "max_tokens": 40,
            "temperature": 0,
        }

        backoff = 2.0

        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.post(endpoint, headers=headers, json=data, timeout=timeout)
            except requests.RequestException as e:
                print(f"[DEBUG] Request exception on attempt {attempt}: {e}")
                if attempt == max_retries:
                    return "failed"
                time.sleep(backoff + random.random())
                backoff = min(backoff * 1.7, 30)
                continue

            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                wait = int(retry_after) if (retry_after and retry_after.isdigit()) else int(backoff)
                print(f"[DEBUG] 429 rate limit, waiting {wait}s (attempt {attempt})")
                time.sleep(wait + 1 + random.random())
                backoff = min(backoff * 1.5, 60)
                continue

            if resp.status_code != 200:
                print(f"\n[DEBUG] HTTP {resp.status_code} (attempt {attempt})")
                try:
                    print(json.dumps(resp.json(), ensure_ascii=False)[:4000])
                except Exception:
                    print(resp.text[:4000])
                return "failed"

            try:
                resp_json = resp.json()
            except Exception:
                print("[DEBUG] Could not parse JSON. Raw:", resp.text[:1000])
                return "failed"

            raw_text = extract_text_from_anthropic_response(resp_json)
            pred = normalize_label(raw_text)

            if not pred:
                print("\n[DEBUG] Empty model output.")
                print("resp_json keys:", list(resp_json.keys()))
                print("resp_json snippet:", json.dumps(resp_json, ensure_ascii=False)[:2000])
                return "failed"

            if pred in cats_set:
                return pred

            for c in cats_norm:
                if c in pred:
                    return c

            print("\n[DEBUG] Output not in label set.")
            print("raw_text:", raw_text[:300])
            print("normalized:", pred)
            return "failed"

        return "failed"

    return classify_text_claude, normalize_label


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Her definerer vi hjelpefunksjoner for evaluering.

    Først henter vi alle dokumentene fra testindeksen, og deretter klassifiserer vi dem én etter én og sammenligner prediksjonene med fasit.
    """)
    return


@app.cell
def _(Counter, ES_URL, classification_report, requests):
    def get_all_docs(index, query=None, size=1000):
        if query is None:
            query = {"match_all": {}}

        r = requests.post(
            f"{ES_URL}/{index}/_search",
            json={
                "size": size,
                "_source": ["innhold", "nivaa1", "split", "tittel", "url", "dokument_id"],
                "query": query,
            },
            timeout=30,
        )
        r.raise_for_status()
        res = r.json()

        hits = res.get("hits", {}).get("hits", [])
        return [h.get("_source", {}) for h in hits]

    def evaluate_from_es(
        categories,
        classify_text_claude,
        normalize_label,
        api_key,
        endpoint,
        terms_prompt,
        index,
        split_value=None,
        few_shot_examples="",
    ):
        y_true, y_pred = [], []
        rows = []
        fail = 0
        processed = 0

        query = {"match_all": {}} if split_value is None else {"term": {"split.keyword": split_value}}

        docs = get_all_docs(index=index, query=query, size=1000)
        print(f"Fetched {len(docs)} docs")

        for ex in docs:
            processed += 1

            title = ex.get("tittel", "unknown")
            url = ex.get("url", "")
            true = (ex.get("nivaa1") or "").strip()

            print(f"\nProcessing document {processed}")
            print(f"Title: {title}")
            print(f"URL: {url}")

            if not true or true.lower() == "nan":
                print("Skipping (no label)")
                continue

            try:
                pred = classify_text_claude(
                    title=title,
                    url=url,
                    content=ex.get("innhold", ""),
                    terms_prompt=terms_prompt,
                    categories=categories,
                    api_key=api_key,
                    endpoint=endpoint,
                    few_shot_examples=few_shot_examples,
                )
            except Exception as e:
                print(f"[WARN] Classification failed for '{title}': {e}")
                pred = "failed"

            print(f"True: {true}")
            print(f"Pred: {pred}")

            true_norm = normalize_label(true)
            y_true.append(true_norm)
            y_pred.append(pred)

            rows.append(
                {
                    "tittel": title,
                    "url": url,
                    "true": true_norm,
                    "pred": pred,
                    "correct": true_norm == pred,
                }
            )

            if pred == "failed":
                fail += 1

        correct = sum(1 for t, p in zip(y_true, y_pred) if p == t)
        acc = correct / len(y_true) if y_true else 0.0

        report = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        ) if y_true else {}
        macro_f1 = round(report.get("macro avg", {}).get("f1-score", 0.0), 4)
        macro_precision = round(report.get("macro avg", {}).get("precision", 0.0), 4)
        macro_recall = round(report.get("macro avg", {}).get("recall", 0.0), 4)

        return {
            "n": len(y_true),
            "accuracy": acc,
            "macro_f1": macro_f1,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "failed": fail,
            "failed_rate": fail / len(y_true) if y_true else 0.0,
            "pred_distribution": Counter(y_pred),
            "y_true": y_true,
            "y_pred": y_pred,
            "rows": rows,
            "processed": processed,
            "fetched": len(docs),
            "classification_report": report,
        }

    return (evaluate_from_es,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Denne cellen sammenligner tre termstrategier:

    - `es`: renset Elasticsearch-termbank
    - `spacy`: renset spaCy-termbank
    - `hybrid`: kombinert og renset termbank

    Poenget er å måle om den forbedrede metoden faktisk gjør det bedre enn baseline.
    """)
    return


@app.cell
def _(
    FIGURES_DIR,
    TEST_INDEX,
    api_key,
    build_terms_prompt_from_bank,
    categories,
    classify_text_claude,
    endpoint,
    es_term_bank,
    evaluate_from_es,
    example_prompts,
    hybrid_term_bank,
    normalize_label,
    pd,
    plt,
    prompt_term_bank_spacy_clean,
    sns,
):
    # "baseline" = description-only: few-shot examples but NO keyword enrichment
    strategy_banks = {
        "baseline": {},
        "es": es_term_bank,
        "spacy": prompt_term_bank_spacy_clean,
        "hybrid": hybrid_term_bank,
    }

    strategy_results = {}
    comparison_rows = []

    PRODUCTION_THRESHOLD = 0.70

    if api_key:
        for strategy_name, bank in strategy_banks.items():
            print(f"\n=== Evaluating strategy: {strategy_name} ===")
            terms_prompt_strategy = build_terms_prompt_from_bank(bank)

            res = evaluate_from_es(
                categories=categories,
                classify_text_claude=classify_text_claude,
                normalize_label=normalize_label,
                api_key=api_key,
                endpoint=endpoint,
                terms_prompt=terms_prompt_strategy,
                index=TEST_INDEX,
                split_value=None,
                few_shot_examples=example_prompts,
            )

            strategy_results[strategy_name] = res
            meets_threshold = res["macro_f1"] >= PRODUCTION_THRESHOLD
            comparison_rows.append(
                {
                    "strategy": strategy_name,
                    "macro_f1": res["macro_f1"],
                    "macro_precision": res["macro_precision"],
                    "macro_recall": res["macro_recall"],
                    "accuracy": round(res["accuracy"], 4),
                    "failed_rate": round(res["failed_rate"], 4),
                    "n": res["n"],
                    "meets_threshold": meets_threshold,
                }
            )
            print(f"  macro-F1: {res['macro_f1']:.3f} {'meets 0.70 threshold' if meets_threshold else 'below 0.70 threshold'}")
    else:
        print("Skipping comparison because API key is missing.")

    comparison_df = pd.DataFrame(comparison_rows)

    # --- Strategy comparison bar chart (thesis figure) ---
    if not comparison_df.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        x = range(len(comparison_df))
        bars = ax.bar(
            [r["strategy"] for r in comparison_rows],
            [r["macro_f1"] for r in comparison_rows],
            color=sns.color_palette("Blues_d", len(comparison_rows)),
        )
        ax.axhline(PRODUCTION_THRESHOLD, color="red", linestyle="--",
                   linewidth=1.5, label=f"Production threshold ({PRODUCTION_THRESHOLD})")
        for bar, val in zip(bars, [r["macro_f1"] for r in comparison_rows]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Term strategy", fontsize=12)
        ax.set_ylabel("Macro-averaged F1", fontsize=12)
        ax.set_title("LOS Classification: Macro-F1 by Term Strategy", fontsize=13)
        ax.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "rq2_strategy_comparison.pdf", bbox_inches="tight")
        plt.savefig(FIGURES_DIR / "rq2_strategy_comparison.png", dpi=150, bbox_inches="tight")
        plt.show()

    comparison_df
    return comparison_df, strategy_results


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Denne cellen kjører én aktiv evaluering med strategien valgt tidligere i notebooken.

    Dette er nyttig hvis man vil ha en “hovedkjøring” i tillegg til sammenligningen mellom strategier.
    """)
    return


@app.cell
def _(
    TERM_STRATEGY,
    TEST_INDEX,
    api_key,
    categories,
    classify_text_claude,
    endpoint,
    evaluate_from_es,
    example_prompts,
    normalize_label,
    terms_prompt,
):
    results = {}
    if api_key:
        results = evaluate_from_es(
            categories=categories,
            classify_text_claude=classify_text_claude,
            normalize_label=normalize_label,
            api_key=api_key,
            endpoint=endpoint,
            terms_prompt=terms_prompt,
            index=TEST_INDEX,
            split_value=None,
            few_shot_examples=example_prompts,
        )
        print("Active TERM_STRATEGY:", TERM_STRATEGY)
        print(f"n={results['fetched']} fetched, {results['processed']} processed")
        print(f"Accuracy:   {results['accuracy']:.4f}")
        print(f"Macro-F1:   {results['macro_f1']:.4f}")
        print(f"Macro-P:    {results['macro_precision']:.4f}")
        print(f"Macro-R:    {results['macro_recall']:.4f}")
    else:
        print("Skipping single-run evaluation because API key is missing.")
    return (results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Nå lager vi en enkel oppsummering av evalueringsresultatet.

    Denne tabellen viser de viktigste nøkkeltallene fra den aktive kjøringen.
    """)
    return


@app.cell
def _(pd, results):
    if results:
        summary_df = pd.DataFrame(
            {
                "mål": ["n", "accuracy", "failed", "failed_rate", "processed"],
                "verdi": [
                    results["n"],
                    round(results["accuracy"], 4),
                    results["failed"],
                    round(results["failed_rate"], 4),
                    results["processed"],
                ],
            }
        )
    else:
        summary_df = pd.DataFrame()

    summary_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Denne cellen lager en klassifikasjonsrapport.

    Rapporten viser blant annet:
    - precision
    - recall
    - f1-score
    per kategori
    """)
    return


@app.cell
def _(classification_report, pd, results):
    if results and results["y_true"]:
        report_dict = classification_report(
            results["y_true"],
            results["y_pred"],
            output_dict=True,
            zero_division=0,
        )
        report_df = pd.DataFrame(report_dict).transpose()
    else:
        report_df = pd.DataFrame()

    report_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Her lager vi en confusion matrix.

    Den viser hvilke kategorier modellen oftest forveksler med hverandre.
    """)
    return


@app.cell
def _(FIGURES_DIR, TERM_STRATEGY, confusion_matrix, pd, plt, results, sns):
    if results and results["y_true"]:
        labels = sorted(set(results["y_true"]) | set(results["y_pred"]))
        cm = confusion_matrix(results["y_true"], results["y_pred"], labels=labels)
        confusion_df = pd.DataFrame(cm, index=labels, columns=labels)

        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(
            confusion_df,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            linewidths=0.5,
        )
        ax.set_title(f"Confusion Matrix - strategy: {TERM_STRATEGY}", fontsize=13)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True", fontsize=11)
        plt.xticks(rotation=40, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"rq2_confusion_matrix_{TERM_STRATEGY}.pdf", bbox_inches="tight")
        plt.savefig(FIGURES_DIR / f"rq2_confusion_matrix_{TERM_STRATEGY}.png", dpi=150, bbox_inches="tight")
        plt.show()
    else:
        confusion_df = pd.DataFrame()

    confusion_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Denne cellen viser fordelingen av prediksjoner.

    Det gjør det lettere å se om modellen overbruker enkelte kategorier.
    """)
    return


@app.cell
def _(pd, results):
    if results:
        pred_distribution_df = (
            pd.DataFrame(results["pred_distribution"].items(), columns=["prediksjon", "antall"])
            .sort_values("antall", ascending=False)
            .reset_index(drop=True)
        )
    else:
        pred_distribution_df = pd.DataFrame()

    pred_distribution_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Her viser vi de første radene i den detaljerte prediksjonstabellen.

    Dette gjør det mulig å inspisere konkrete dokumenter, fasit og prediksjon side om side.
    """)
    return


@app.cell
def _(pd, results):
    if results:
        prediction_rows_df = pd.DataFrame(results["rows"])
    else:
        prediction_rows_df = pd.DataFrame()

    prediction_rows_df.head(50)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Per-kategori F1-score

    Denne figuren viser F1-score per LOS-kategori for den aktive kjøringen.
    Den hjelper med å identifisere hvilke kategorier som er vanskeligst å klassifisere riktig.
    """)
    return


@app.cell
def _(FIGURES_DIR, TERM_STRATEGY, classification_report, pd, plt, results, sns):
    if results and results["y_true"]:
        _report = classification_report(
            results["y_true"], results["y_pred"],
            output_dict=True, zero_division=0,
        )
        _rows = [
            {"category": k, "f1": round(v["f1-score"], 3),
             "precision": round(v["precision"], 3),
             "recall": round(v["recall"], 3),
             "support": int(v["support"])}
            for k, v in _report.items()
            if k not in ("accuracy", "macro avg", "weighted avg")
        ]
        per_cat_df = pd.DataFrame(_rows).sort_values("f1", ascending=True)

        fig, ax = plt.subplots(figsize=(10, max(4, len(_rows) * 0.55)))
        colors = ["#d73027" if f < 0.70 else "#4575b4" for f in per_cat_df["f1"]]
        bars = ax.barh(per_cat_df["category"], per_cat_df["f1"], color=colors)
        ax.axvline(0.70, color="red", linestyle="--", linewidth=1.5,
                   label="Production threshold (0.70)")
        for bar, row in zip(bars, per_cat_df.itertuples()):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{row.f1:.3f} (n={row.support})",
                    va="center", fontsize=10)
        ax.set_xlim(0, 1.15)
        ax.set_xlabel("F1-score", fontsize=12)
        ax.set_title(f"Per-Category F1 - strategy: {TERM_STRATEGY}", fontsize=13)
        ax.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"rq2_per_category_f1_{TERM_STRATEGY}.pdf", bbox_inches="tight")
        plt.savefig(FIGURES_DIR / f"rq2_per_category_f1_{TERM_STRATEGY}.png", dpi=150, bbox_inches="tight")
        plt.show()
    else:
        per_cat_df = pd.DataFrame()

    per_cat_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Precision vs Recall per kategori

    Spredningsdiagram der hvert punkt er én kategori.
    Størrelsen på punktet viser antall testdokumenter (support).
    Kategorier i øvre høyre hjørne er best; kategorier nede til venstre trenger forbedring.
    """)
    return


@app.cell
def _(FIGURES_DIR, TERM_STRATEGY, classification_report, pd, plt, results, sns):
    if results and results["y_true"]:
        _report = classification_report(
            results["y_true"], results["y_pred"],
            output_dict=True, zero_division=0,
        )
        _pr_rows = [
            {"category": k,
             "precision": round(v["precision"], 3),
             "recall": round(v["recall"], 3),
             "f1": round(v["f1-score"], 3),
             "support": int(v["support"])}
            for k, v in _report.items()
            if k not in ("accuracy", "macro avg", "weighted avg")
        ]
        _pr_df = pd.DataFrame(_pr_rows)

        import numpy as _np
        _fig, _ax = plt.subplots(figsize=(9, 7))
        _colors = sns.color_palette("Set2", len(_pr_df))
        for (_i, _row), _color in zip(_pr_df.iterrows(), _colors):
            _ax.scatter(_row["recall"], _row["precision"],
                        s=max(60, _row["support"] * 15),
                        color=_color, alpha=0.85, edgecolors="white", linewidth=0.8)
            _ax.annotate(
                _row["category"],
                (_row["recall"], _row["precision"]),
                textcoords="offset points", xytext=(6, 4), fontsize=9,
            )
        _ax.axhline(0.70, color="red", linestyle="--", linewidth=1, alpha=0.5, label="P = 0.70")
        _ax.axvline(0.70, color="orange", linestyle="--", linewidth=1, alpha=0.5, label="R = 0.70")
        _ax.set_xlim(-0.05, 1.1)
        _ax.set_ylim(-0.05, 1.1)
        _ax.set_xlabel("Recall", fontsize=12)
        _ax.set_ylabel("Precision", fontsize=12)
        _ax.set_title(
            f"Precision vs Recall per Category - strategy: {TERM_STRATEGY}\n"
            f"(bubble size ∝ test support)",
            fontsize=12,
        )
        _ax.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"rq2_precision_recall_scatter_{TERM_STRATEGY}.pdf", bbox_inches="tight")
        plt.savefig(FIGURES_DIR / f"rq2_precision_recall_scatter_{TERM_STRATEGY}.png", dpi=150, bbox_inches="tight")
        plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Produksjonsterskel - oppsummering

    Sjekker om klassifiseringen er klar for produksjonsbruk basert på macro-F1 ≥ 0.70.
    """)
    return


@app.cell
def _(mo, results):
    if results:
        _f1 = results["macro_f1"]
        _meets = _f1 >= 0.70
        mo.md(f"""
        | Mål | Verdi |
        |-----|-------|
        | Macro-F1 | **{_f1:.4f}** |
        | Produksjonsterskel | 0.70 |
        | Klar for produksjon? | {"**Ja**" if _meets else "**Nei**"} |
        | Feilrate (API-feil) | {results['failed_rate']:.2%} |
        | Antall testdokumenter | {results['n']} |
        """)
    return


@app.cell(hide_code=True)
def _(TERM_STRATEGY, mo):
    mo.md(f"""
    ## Oppsummering

    Notebooken er nå organisert som en full pipeline med forklaring i hvert steg:

    - innlasting og EDA
    - labeldekning og kategorioversikt
    - hierarki og dokumentlengde
    - train/test-splitt
    - Elasticsearch baseline (beskrivelse-kun, ingen nøkkelord)
    - overlappsanalyse på nivå 1 og nivå 2
    - spaCy phrase extraction
    - scoring og rensing av termer
    - bygging av ES-, spaCy- og hybrid-termbank
    - klassifisering med LLM (few-shot + nøkkelordberikt prompt)
    - evaluering og sammenligning (macro-F1 som primærmål)

    Aktiv termstrategi i denne kjøringen: **{TERM_STRATEGY}**
    """)
    return


if __name__ == "__main__":
    app.run()
