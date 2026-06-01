"""
End-to-end RQ2 LOS classification pipeline.

Replaces the marimo notebook with a deterministic, headless run that produces
all figures and a results JSON that the thesis can consume.

Run: python run_classification.py [--mini]
  --mini limits the test set to 10 random docs per category for a quick sanity run.
"""

import argparse
import io
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# Force UTF-8 stdout on Windows so we can print Norwegian + arrows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import spacy
from dotenv import load_dotenv
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             accuracy_score)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT.parent / "Figures" / "RQ2"
RESULTS_DIR = ROOT / "results_output"
FIG_DIR.mkdir(exist_ok=True, parents=True)
RESULTS_DIR.mkdir(exist_ok=True)

ES_URL = "http://localhost:9200"
TRAIN_INDEX = "los-documents-train"
TEST_INDEX = "los-documents-test"

CATEGORIES = [
    "Helse og omsorg",
    "Opplæring og utdanning",
    "Arbeid",
    "Sosial og økonomisk trygghet",
    "Kultur-Idrett-Fritid",
    "Innbygger",
]

CATEGORY_GUIDELINES = {
    "Helse og omsorg": "helse, diagnose, behandling, helsehjelp, pasientforløp, habilitering, rehabilitering, omsorgstjenester, pleie, oppfølging i helsevesenet, eller rettigheter knyttet til helsetjenester.",
    "Opplæring og utdanning": "barnehage, skole, SFO, undervisning, opplæring, spesialpedagogikk, PP-tjeneste, læring, utdanningsrettigheter eller tilrettelegging i utdanningsløpet.",
    "Arbeid": "arbeidsdeltakelse, jobb, arbeidsrettet oppfølging, arbeidsevne, tilrettelegging i arbeidslivet, tiltak for å komme i arbeid eller varig tilrettelagt arbeid.",
    "Sosial og økonomisk trygghet": "ytelser, stønader, trygd, økonomiske rettigheter, refusjoner, økonomisk støtte, hjelpemidler eller andre støtteordninger.",
    "Kultur-Idrett-Fritid": "fritidsaktiviteter, kulturtilbud, idrett, deltakelse i aktiviteter, fysisk aktivitet som fritidstilbud, eller organiserte aktivitetstilbud.",
    "Innbygger": "generelle offentlige tjenester, borgerrettet informasjon, bruk av offentlige digitale tjenester, representasjon, fullmakter eller annen generell innbyggerinformasjon som ikke passer best i de andre kategoriene.",
}

SIG_TEXT_CONFIG = {"size": 50, "min_doc_count": 3}

BAD_TERMS = {
    "oslo", "bergen", "torsdag", "mandag", "internett", "steder",
    "artikkel", "last", "skriv", "pdfapi", "api", "printpageaspdfdocument",
    "kontaktoss", "refererer", "ungdom", "unge", "22", "http", "https",
    "www", "mailto", "side", "sider",
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

SEED = 42
TRUNCATE_CHARS = 12000
# For the summarize-then-classify strategy: how much of the full document to feed
# the summarizer. Far above the 12k classify clamp, so the summary can see the tail
# of long documents that truncation would otherwise drop. Bounded to protect against
# the >300k-char outliers in the corpus.
SUMMARY_MAX_INPUT_CHARS = 50000

# Load .env
load_dotenv(ROOT.parent / ".env", override=False)
load_dotenv(ROOT / ".env", override=False)

API_KEY = os.getenv("AZURE_ANTHROPIC_API_KEY", "")
ENDPOINT = os.getenv(
    "AZURE_ANTHROPIC_ENDPOINT",
    "https://your-foundry-resource.services.ai.azure.com/anthropic/v1/messages",
)
MODEL_NAME = "claude-haiku-4-5"

AZURE_OAI_KEY = os.getenv("AZURE_API_KEY", "")
AZURE_OAI_BASE = os.getenv("AZURE_API_BASE", "").rstrip("/")
AZURE_OAI_VERSION = os.getenv("AZURE_API_VERSION", "2024-12-01-preview")
AZURE_OAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")

# GPT-5.1 lives on the same Azure OpenAI resource as GPT-4.1; only the
# deployment name and the api-version differ. Reuse the values from the RQ3
# .env (RQ3 swaps its evaluator to this same gpt-5.1 deployment). GPT-5.1 is a
# reasoning model and needs a 2025-era api-version.
AZURE_OAI_DEPLOYMENT_GPT51 = os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT51", "gpt-5.1-chat")
AZURE_OAI_VERSION_GPT51 = os.getenv("AZURE_API_VERSION_GPT51", "2025-04-01-preview")

# Selected at runtime by --model. Defaults to haiku for backward compatibility.
ACTIVE_MODEL = "haiku"
# Selected at runtime by --confidence. When True the phase-2 classify prompt asks
# for JSON {kategori, confidence, begrunnelse} so a confidence/abstain analysis is
# possible. Off by default so existing label-only runs are byte-identical.
CONFIDENCE_MODE = False


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def normalize_label(s):
    """Strip whitespace, normalize unicode."""
    if s is None:
        return None
    return str(s).strip()


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            label = d.get("nivå 1") or d.get("nivaa1")
            if not label or label == "NaN":
                continue
            rows.append({
                "url": d.get("url"),
                "tittel": d.get("tittel") or "",
                "innhold": d.get("innhold") or "",
                "nivaa1": normalize_label(label),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------

def class_distribution_plot(train_df, test_df):
    counts = pd.DataFrame({
        "train": train_df["nivaa1"].value_counts().reindex(CATEGORIES, fill_value=0),
        "test": test_df["nivaa1"].value_counts().reindex(CATEGORIES, fill_value=0),
    })
    counts["total"] = counts.sum(axis=1)
    counts.to_csv(RESULTS_DIR / "class_distribution.csv")

    # combined chart
    fig, ax = plt.subplots(figsize=(10, 5))
    counts[["train", "test"]].plot(kind="bar", ax=ax, color=["#3b7dd8", "#e07a3a"], edgecolor="black")
    ax.set_xlabel("LOS level-1 category")
    ax.set_ylabel("Document count")
    ax.set_title("Class distribution across train and test partitions")
    ax.set_xticklabels(counts.index, rotation=30, ha="right")
    ax.legend(title="Partition")
    for i, (tr, te) in enumerate(zip(counts["train"], counts["test"])):
        ax.text(i - 0.15, tr + 3, str(tr), ha="center", fontsize=8)
        ax.text(i + 0.15, te + 3, str(te), ha="center", fontsize=8)
    plt.tight_layout()
    save_fig("rq2_class_distribution", fig)

    # train_test_split-only (similar but emphasizes split balance)
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    proportion = counts[["train", "test"]].div(counts["total"], axis=0) * 100
    proportion.plot(kind="bar", stacked=True, ax=ax2, color=["#3b7dd8", "#e07a3a"], edgecolor="black")
    ax2.set_xlabel("LOS level-1 category")
    ax2.set_ylabel("Proportion of category (%)")
    ax2.set_title("Stratified train/test split - proportion per category")
    ax2.set_xticklabels(proportion.index, rotation=30, ha="right")
    ax2.set_ylim(0, 110)
    ax2.legend(title="Partition", loc="lower right")
    for i, (tr_pct, te_pct) in enumerate(zip(proportion["train"], proportion["test"])):
        ax2.text(i, 50, f"{tr_pct:.0f}%", ha="center", color="white", fontsize=9)
        ax2.text(i, tr_pct + te_pct/2, f"{te_pct:.0f}%", ha="center", color="white", fontsize=9)
    plt.tight_layout()
    save_fig("rq2_train_test_split", fig2)

    return counts


def doc_length_plot(train_df, test_df):
    all_df = pd.concat([train_df, test_df], ignore_index=True)
    word_counts = all_df["innhold"].apply(lambda s: len(s.split()))
    stats = {
        "mean": float(word_counts.mean()),
        "median": float(word_counts.median()),
        "min": int(word_counts.min()),
        "max": int(word_counts.max()),
        "std": float(word_counts.std()),
        "pct_truncated_12k": float((all_df["innhold"].str.len() > TRUNCATE_CHARS).mean() * 100),
    }
    print(f"Doc length stats: {stats}")
    pd.Series(stats).to_csv(RESULTS_DIR / "doc_length_stats.csv")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(word_counts, bins=50, color="#3b7dd8", edgecolor="black")
    ax.axvline(stats["median"], color="red", linestyle="--", label=f"Median = {stats['median']:.0f}")
    ax.axvline(stats["mean"], color="orange", linestyle="--", label=f"Mean = {stats['mean']:.0f}")
    ax.set_xlabel("Document length (words)")
    ax.set_ylabel("Number of documents")
    ax.set_title(f"Document length distribution (n={len(all_df)})")
    ax.legend()
    plt.tight_layout()
    save_fig("rq2_doc_length_distribution", fig)
    return stats


# ---------------------------------------------------------------------------
# Significant terms
# ---------------------------------------------------------------------------

def es_post(index, body):
    r = requests.post(f"{ES_URL}/{index}/_search", json=body, timeout=30)
    r.raise_for_status()
    return r.json()


def is_good_term(term):
    t = str(term).strip().lower()
    if not t or len(t) < 3:
        return False
    if t in BAD_TERMS:
        return False
    for pat in BAD_PATTERNS:
        if re.match(pat, t):
            return False
    return True


def significant_terms_for_category(category, index=TRAIN_INDEX, size=None, min_doc_count=None):
    size = size or SIG_TEXT_CONFIG["size"]
    min_doc_count = min_doc_count or SIG_TEXT_CONFIG["min_doc_count"]
    body = {
        "size": 0,
        "query": {"bool": {"filter": [{"term": {"nivaa1.keyword": category}}]}},
        "aggs": {
            "cat_terms": {
                "significant_text": {
                    "field": "innhold",
                    "size": size,
                    "min_doc_count": min_doc_count,
                    "background_filter": {"match_all": {}},
                    "mutual_information": {
                        "include_negatives": True,
                        "background_is_superset": True,
                    },
                }
            }
        },
    }
    res = es_post(index, body)
    return [b["key"] for b in res["aggregations"]["cat_terms"]["buckets"]]


def build_es_term_bank(top_n_per_cat=12):
    raw = {}
    cleaned = {}
    for cat in CATEGORIES:
        try:
            terms = significant_terms_for_category(cat)
        except Exception as e:
            print(f"  warning: significant_terms failed for {cat}: {e}")
            terms = []
        raw[cat] = terms
        cleaned[cat] = [t for t in terms if is_good_term(t)][:top_n_per_cat]
    return raw, cleaned


def terms_kept_vs_removed_plot(raw_terms):
    rows = []
    for cat in CATEGORIES:
        all_terms = raw_terms.get(cat, [])
        kept = [t for t in all_terms if is_good_term(t)]
        removed = [t for t in all_terms if not is_good_term(t)]
        rows.append({"cat": cat, "kept": len(kept), "removed": len(removed)})
    df = pd.DataFrame(rows).set_index("cat")
    df.to_csv(RESULTS_DIR / "terms_kept_vs_removed.csv")

    fig, ax = plt.subplots(figsize=(10, 5))
    df.plot(kind="bar", stacked=True, ax=ax, color=["#3b7dd8", "#aaaaaa"], edgecolor="black")
    ax.set_xlabel("LOS level-1 category")
    ax.set_ylabel("Number of significant terms")
    ax.set_title("ES significant_text terms: kept vs removed after artifact filtering")
    ax.set_xticklabels(df.index, rotation=30, ha="right")
    ax.legend(["kept", "removed (UUIDs/URLs/PDFs/short)"])
    for i, (k, r) in enumerate(zip(df["kept"], df["removed"])):
        ax.text(i, k + r + 0.5, f"{k}/{k+r}", ha="center", fontsize=9)
    plt.tight_layout()
    save_fig("rq2_terms_kept_vs_removed", fig)


def term_overlap_heatmap(es_terms):
    sets = {cat: set(es_terms.get(cat, [])) for cat in CATEGORIES}
    n = len(CATEGORIES)
    M = np.zeros((n, n))
    for i, ci in enumerate(CATEGORIES):
        for j, cj in enumerate(CATEGORIES):
            if not sets[ci] or not sets[cj]:
                continue
            inter = len(sets[ci] & sets[cj])
            union = len(sets[ci] | sets[cj])
            M[i, j] = inter / union if union > 0 else 0

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        M, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=CATEGORIES, yticklabels=CATEGORIES, ax=ax, cbar_kws={"label": "Jaccard overlap"}
    )
    ax.set_title("Vocabulary overlap (Jaccard) between categories - ES terms")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    save_fig("rq2_term_overlap_heatmap", fig)


# ---------------------------------------------------------------------------
# spaCy noun-phrase keywords
# ---------------------------------------------------------------------------

def build_spacy_term_bank(train_df, top_n_per_cat=12):
    print("Loading spaCy nb_core_news_md...")
    nlp = spacy.load("nb_core_news_md", disable=["parser", "ner"])
    nlp.max_length = 2_000_000
    cat_to_phrases = defaultdict(Counter)

    for _, row in train_df.iterrows():
        cat = row["nivaa1"]
        text = (row["tittel"] + " " + row["innhold"])[:50_000]
        doc = nlp(text)
        # take noun phrases via POS tagger (no parser, so do simple NOUN+ADJ+PROPN windowing)
        tokens = [t for t in doc if not t.is_stop and not t.is_punct and t.pos_ in ("NOUN", "PROPN", "ADJ")]
        for t in tokens:
            lemma = t.lemma_.lower()
            if is_good_term(lemma):
                cat_to_phrases[cat][lemma] += 1

    # exclusivity scoring: term frequency in cat / total frequency across cats
    all_freq = Counter()
    for cat, c in cat_to_phrases.items():
        all_freq.update(c)

    cleaned = {}
    for cat in CATEGORIES:
        scored = []
        for term, n in cat_to_phrases[cat].items():
            total = all_freq[term]
            if total < 3:
                continue
            exclusivity = n / total
            score = exclusivity * np.log1p(n)
            scored.append((term, score, n, exclusivity))
        scored.sort(key=lambda x: -x[1])
        cleaned[cat] = [t for t, _, _, _ in scored[:top_n_per_cat]]
    return cleaned


# ---------------------------------------------------------------------------
# Few-shot examples
# ---------------------------------------------------------------------------

def fetch_fewshot_examples(train_df, k_per_cat=2, max_chars=1200):
    """Pick k_per_cat examples per category from the training set, prefer mid-length docs."""
    rng = random.Random(SEED)
    examples = {}
    for cat in CATEGORIES:
        sub = train_df[train_df["nivaa1"] == cat]
        if len(sub) == 0:
            examples[cat] = []
            continue
        # sort by absolute distance from median length, take closest
        median_len = sub["innhold"].str.len().median()
        sub = sub.assign(dist=(sub["innhold"].str.len() - median_len).abs())
        sub = sub.sort_values("dist").head(min(k_per_cat * 3, len(sub)))
        picks = sub.sample(n=min(k_per_cat, len(sub)), random_state=SEED)
        examples[cat] = [
            {
                "tittel": r["tittel"],
                "url": r["url"],
                "innhold": r["innhold"][:max_chars],
            }
            for _, r in picks.iterrows()
        ]
    return examples


def format_fewshot_block(examples_by_cat):
    parts = []
    for cat in CATEGORIES:
        for ex in examples_by_cat.get(cat, []):
            parts.append(f"--- Eksempel: kategori = {cat} ---")
            parts.append(f"Tittel: {ex['tittel']}")
            snippet = ex["innhold"][:600].replace("\n\n\n", "\n\n").strip()
            parts.append(f"Innhold (utdrag): {snippet}")
            parts.append("")
    return "\n".join(parts)


def format_terms_block(term_bank):
    parts = []
    for cat in CATEGORIES:
        terms = term_bank.get(cat, [])
        if terms:
            parts.append(f"- {cat}: {', '.join(terms)}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Prompt assembly + Claude API
# ---------------------------------------------------------------------------

def clamp(text, n=TRUNCATE_CHARS):
    return text[:n]


def build_prompt(title, url, content, terms_prompt, fewshot_block, confidence=False):
    cats_str = ", ".join(CATEGORIES)
    guidelines = "\n".join(
        f"- Velg \"{cat}\" når dokumentet primært handler om {desc}"
        for cat, desc in CATEGORY_GUIDELINES.items()
    )
    fewshot_section = f"Eksempler fra treningsdata:\n{fewshot_block}\n" if fewshot_block else ""
    terms_section = (
        f"Kategori-indikative begreper fra treningsdata:\n{terms_prompt}\n\n"
        f"Bruk begrepene over som støtte til mønstergjenkjenning, men ikke som fasit. "
        f"Dersom begrepene peker en vei, men innholdets hovedformål peker en annen vei, "
        f"skal du følge innholdets hovedformål.\n"
        if terms_prompt else ""
    )
    if confidence:
        closing = (
            "Svar KUN med gyldig JSON (ingen markdown, ingen ekstra tekst) på nøyaktig dette formatet:\n"
            '{"kategori": "<en av kategoriene under>", "confidence": <desimaltall mellom 0.0 og 1.0>, "begrunnelse": "<en kort setning>"}\n'
            'der "kategori" må være nøyaktig en av disse verdiene:\n' + cats_str + "\n"
            'og "confidence" angir hvor sikker du er på kategorien (1.0 = helt sikker, 0.0 = ren gjetning).'
        )
    else:
        closing = (
            "Returner kun selve LOS-kategorien.\n"
            "Ikke skriv oppsummering. Ikke skriv begrunnelse.\n"
            "Ikke skriv markdown. Ikke skriv punktliste. Ikke skriv ekstra tekst.\n\n"
            "Gyldige svar er kun en av disse verdiene:\n" + cats_str
        )
    return f"""
Du er en norsk LOS-tag-klassifiserer.

Oppgaven din er å lese et norsk dokument og velge nøyaktig én LOS-kategori som best beskriver dokumentets hovedtema.

LOS-kategoriene du kan velge mellom er:
{cats_str}

Hva du skal gjøre:
Du skal klassifisere dokumentet etter hvilken LOS-kategori det primært tilhører. Du skal ikke bare se på enkeltord, avsender eller URL, men forstå hva dokumentet faktisk handler om, hvilken type behov det dekker, og hvilket samfunnsområde det tilhører.

Viktige regler:
- Velg nøyaktig én kategori.
- Velg hovedtema, ikke et side- eller støttetema.
- Ikke velg kategori bare fordi enkelte ord forekommer ofte.
- Ikke velg kategori bare fordi avsenderen er for eksempel NAV, Helsenorge, Helsedirektoratet, Udir eller en kommune.
- Bruk tittel, URL og innhold samlet.

Retningslinjer for kategorisering:
{guidelines}

{fewshot_section}{terms_section}
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

{closing}
""".strip()


SYSTEM_PROMPT = "Du er en LOS-klassifiserer for norske offentlige dokumenter. Svar med kun én kategori."


def call_claude(user_prompt, max_retries=6, timeout=60, max_tokens=64, system=SYSTEM_PROMPT):
    headers = {
        "Content-Type": "application/json",
        "x-api-key": API_KEY,
        "anthropic-version": "2023-06-01",
    }
    body = {
        "model": MODEL_NAME,
        "system": system,
        "max_tokens": max_tokens,
        "temperature": 0,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    last_err = None
    for attempt in range(max_retries):
        try:
            r = requests.post(ENDPOINT, headers=headers, json=body, timeout=timeout)
            if r.status_code == 429 or r.status_code >= 500:
                wait = 2 ** attempt
                time.sleep(wait)
                last_err = f"http {r.status_code}"
                continue
            r.raise_for_status()
            data = r.json()
            text = data["content"][0]["text"].strip()
            return text
        except requests.exceptions.RequestException as e:
            last_err = str(e)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"call_claude failed after {max_retries} retries: {last_err}")


def call_gpt41(user_prompt, max_retries=6, timeout=60, max_tokens=64, system=SYSTEM_PROMPT):
    if not AZURE_OAI_KEY or not AZURE_OAI_BASE:
        raise RuntimeError("AZURE_API_KEY / AZURE_API_BASE not set")
    url = f"{AZURE_OAI_BASE}/openai/deployments/{AZURE_OAI_DEPLOYMENT}/chat/completions?api-version={AZURE_OAI_VERSION}"
    headers = {"Content-Type": "application/json", "api-key": AZURE_OAI_KEY}
    body = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    last_err = None
    for attempt in range(max_retries):
        try:
            r = requests.post(url, headers=headers, json=body, timeout=timeout)
            if r.status_code == 429 or r.status_code >= 500:
                wait = 2 ** attempt
                time.sleep(wait)
                last_err = f"http {r.status_code}"
                continue
            r.raise_for_status()
            data = r.json()
            text = data["choices"][0]["message"]["content"].strip()
            return text
        except requests.exceptions.RequestException as e:
            last_err = str(e)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"call_gpt41 failed after {max_retries} retries: {last_err}")


def call_gpt51(user_prompt, max_retries=6, timeout=120, max_tokens=64, system=SYSTEM_PROMPT):
    """Call GPT-5.1 via Azure OpenAI.

    GPT-5.1 is a reasoning model, so the request differs from call_gpt41:
      * uses max_completion_tokens (max_tokens is rejected by reasoning models),
      * omits temperature (only the default value is supported),
      * sets reasoning_effort low to keep single-label classification cheap,
      * budgets extra completion tokens, since reasoning tokens are consumed
        against the same limit before any answer text is emitted.
    Needs a gpt-5.1 deployment and a 2025-era api-version
    (AZURE_OPENAI_DEPLOYMENT_GPT51 / AZURE_API_VERSION_GPT51); align these with
    the RQ3 .env. Falls back gracefully if the api-version rejects
    reasoning_effort or if the reasoning budget is exhausted before an answer.
    """
    if not AZURE_OAI_KEY or not AZURE_OAI_BASE:
        raise RuntimeError("AZURE_API_KEY / AZURE_API_BASE not set")
    url = (f"{AZURE_OAI_BASE}/openai/deployments/{AZURE_OAI_DEPLOYMENT_GPT51}"
           f"/chat/completions?api-version={AZURE_OAI_VERSION_GPT51}")
    headers = {"Content-Type": "application/json", "api-key": AZURE_OAI_KEY}
    body = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
        # head-room: reasoning tokens are billed before the label is produced
        "max_completion_tokens": max(512, max_tokens * 8),
        "reasoning_effort": "minimal",
    }
    last_err = None
    for attempt in range(max_retries):
        try:
            r = requests.post(url, headers=headers, json=body, timeout=timeout)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 ** attempt)
                last_err = f"http {r.status_code}"
                continue
            if r.status_code == 400 and "reasoning_effort" in body:
                # some api-versions reject reasoning_effort; drop it and retry
                last_err = f"http 400: {r.text[:200]}"
                body.pop("reasoning_effort", None)
                continue
            if r.status_code in (401, 403, 404):
                # access/config error (firewall, wrong key, missing deployment) -
                # not transient. Fail fast with a clear message instead of burning
                # the retry backoff on every document.
                raise RuntimeError(f"call_gpt51 access error http {r.status_code}: {r.text[:300]}")
            r.raise_for_status()
            data = r.json()
            text = (data["choices"][0]["message"].get("content") or "").strip()
            if not text:
                # reasoning consumed the whole budget; raise it and retry
                last_err = "empty content (reasoning budget exhausted)"
                body["max_completion_tokens"] = min(body["max_completion_tokens"] * 2, 4096)
                continue
            return text
        except requests.exceptions.RequestException as e:
            last_err = str(e)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"call_gpt51 failed after {max_retries} retries: {last_err}")


def call_llm(user_prompt, max_tokens=64, system=SYSTEM_PROMPT):
    """Dispatch to the active LLM based on ACTIVE_MODEL."""
    if ACTIVE_MODEL == "gpt41":
        return call_gpt41(user_prompt, max_tokens=max_tokens, system=system)
    if ACTIVE_MODEL == "gpt51":
        return call_gpt51(user_prompt, max_tokens=max_tokens, system=system)
    return call_claude(user_prompt, max_tokens=max_tokens, system=system)


def parse_label(raw):
    """Map raw model output to a canonical category string."""
    raw = raw.strip().strip('"').strip("'").strip().lower()
    for cat in CATEGORIES:
        if raw == cat.lower():
            return cat
    # partial matches
    for cat in CATEGORIES:
        if cat.lower() in raw or raw in cat.lower():
            return cat
    # token match
    for cat in CATEGORIES:
        first_word = cat.lower().split()[0]
        if first_word in raw:
            return cat
    return None


def parse_confidence_json(raw):
    """Parse a confidence-mode JSON reply into (label, confidence, reasoning).
    Robust to code fences / surrounding text; falls back to bare-label parsing
    with confidence=NaN so a malformed reply is still scored on its label."""
    import re as _re
    text = raw.strip()
    if text.startswith("```"):
        text = _re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = _re.sub(r"\n?```$", "", text).strip()
    obj = None
    try:
        obj = json.loads(text)
    except Exception:
        m = _re.search(r"\{.*\}", text, _re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:
                obj = None
    if isinstance(obj, dict):
        label = parse_label(str(obj.get("kategori", "")))
        try:
            conf = max(0.0, min(1.0, float(obj.get("confidence"))))
        except Exception:
            conf = float("nan")
        reasoning = str(obj.get("begrunnelse", "") or "")
        return label, conf, reasoning
    return parse_label(raw), float("nan"), ""


# ---------------------------------------------------------------------------
# Summarize-then-classify (phase 1: topical summary of the full document)
# ---------------------------------------------------------------------------

SUMMARY_SYSTEM_PROMPT = (
    "Du er en ekspert på norsk offentlig tjenesteklassifisering (LOS). "
    "Du skriver korte, klassifiseringsrettede sammendrag av offentlige dokumenter."
)


def build_summary_prompt(title, content):
    cats_str = "\n".join(f"- {cat}" for cat in CATEGORIES)
    return f"""
Analyser følgende norske dokument og skriv et kort sammendrag som er nyttig for å klassifisere det til én LOS-kategori. Fokuser på:
1. Hovedtema: Hva handler dokumentet hovedsakelig om?
2. Målgruppe: Hvem er dokumentet rettet mot (for eksempel pasienter, pårørende, fagpersoner)?
3. Tjenestetype: Hvilken type offentlig tjeneste beskrives?
4. Nøkkelord: List 5-10 domenespesifikke nøkkelord.

LOS-kategoriene er oppgitt kun som kontekst (ikke velg en kategori her):
{cats_str}

Dokument:
Tittel: {title}
Innhold:
{content}

Skriv sammendraget på norsk, maks 200 ord. Ikke velg en LOS-kategori. Skriv kun selve sammendraget.
""".strip()


_SUMMARY_CACHE = None


def summary_cache_path():
    return RESULTS_DIR / f"summary_cache_{ACTIVE_MODEL}.json"


def load_summary_cache():
    """Per-model on-disk cache so summaries are computed once and survive a crash."""
    global _SUMMARY_CACHE
    if _SUMMARY_CACHE is None:
        p = summary_cache_path()
        if p.exists():
            with open(p, encoding="utf-8") as f:
                _SUMMARY_CACHE = json.load(f)
            print(f"  loaded {len(_SUMMARY_CACHE)} cached summaries from {p.name}")
        else:
            _SUMMARY_CACHE = {}
    return _SUMMARY_CACHE


def save_summary_cache():
    if _SUMMARY_CACHE is not None:
        with open(summary_cache_path(), "w", encoding="utf-8") as f:
            json.dump(_SUMMARY_CACHE, f, ensure_ascii=False, indent=2)


def summarize_doc(title, url, content):
    """Phase 1: summarize the (near-)full document; cached per model by url."""
    cache = load_summary_cache()
    key = url or title
    cached = cache.get(key)
    if cached and not cached.startswith("ERROR:"):
        return cached
    prompt = build_summary_prompt(title, content[:SUMMARY_MAX_INPUT_CHARS])
    try:
        summary = call_llm(prompt, max_tokens=500, system=SUMMARY_SYSTEM_PROMPT).strip()
    except Exception as e:
        summary = f"ERROR: {e}"
    cache[key] = summary
    return summary


# ---------------------------------------------------------------------------
# Run a strategy
# ---------------------------------------------------------------------------

def classify_strategy(strategy, test_df, term_bank_es, term_bank_spacy, term_bank_hybrid, fewshot_block):
    use_summary = strategy == "summary"
    if strategy in ("baseline", "summary"):
        terms = ""
    elif strategy == "es":
        terms = format_terms_block(term_bank_es)
    elif strategy == "spacy":
        terms = format_terms_block(term_bank_spacy)
    elif strategy == "hybrid":
        terms = format_terms_block(term_bank_hybrid)
    else:
        raise ValueError(strategy)

    rows = []
    print(f"\n=== Classifying {len(test_df)} documents with strategy '{strategy}' ===")
    t0 = time.time()
    for i, (_, row) in enumerate(test_df.iterrows()):
        # Summary strategy: phase 1 summarizes the near-full doc, then phase 2
        # classifies that summary. All other strategies classify the 12k-clamped
        # document. Everything else in the prompt is identical, so the only
        # difference between baseline and summary is summarize-vs-truncate.
        if use_summary:
            doc_content = summarize_doc(row["tittel"], row["url"], row["innhold"])
        else:
            doc_content = clamp(row["innhold"])
        prompt = build_prompt(
            title=row["tittel"],
            url=row["url"],
            content=doc_content,
            terms_prompt=terms,
            fewshot_block=fewshot_block,
            confidence=CONFIDENCE_MODE,
        )
        try:
            raw = call_llm(prompt, max_tokens=256 if CONFIDENCE_MODE else 64)
        except Exception as e:
            raw = f"ERROR: {e}"
        if CONFIDENCE_MODE:
            pred, conf, reasoning = parse_confidence_json(raw)
        else:
            pred, conf, reasoning = parse_label(raw), float("nan"), ""
        rows.append({
            "url": row["url"],
            "tittel": row["tittel"],
            "true": row["nivaa1"],
            "pred": pred,
            "confidence": conf,
            "reasoning": reasoning,
            "raw": raw,
        })
        if (i + 1) % 20 == 0 or i == len(test_df) - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  [{strategy}] {i+1}/{len(test_df)}  ({rate:.1f} docs/s, {elapsed:.0f}s elapsed)")
            if use_summary:
                save_summary_cache()
    if use_summary:
        save_summary_cache()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Metrics + figures per strategy
# ---------------------------------------------------------------------------

def compute_metrics(preds_df):
    df = preds_df.dropna(subset=["pred"]).copy()
    if len(df) == 0:
        return {"macro_f1": 0, "macro_precision": 0, "macro_recall": 0, "accuracy": 0,
                "per_cat": {}, "n": 0, "n_total": len(preds_df),
                "n_unparseable": int(preds_df["pred"].isna().sum())}
    y_true = df["true"].tolist()
    y_pred = df["pred"].tolist()
    macro_f1 = f1_score(y_true, y_pred, labels=CATEGORIES, average="macro", zero_division=0)
    macro_p = precision_score(y_true, y_pred, labels=CATEGORIES, average="macro", zero_division=0)
    macro_r = recall_score(y_true, y_pred, labels=CATEGORIES, average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    rep = classification_report(y_true, y_pred, labels=CATEGORIES, output_dict=True, zero_division=0)
    per_cat = {cat: {
        "precision": rep[cat]["precision"],
        "recall": rep[cat]["recall"],
        "f1": rep[cat]["f1-score"],
        "support": int(rep[cat]["support"]),
    } for cat in CATEGORIES}
    return {
        "macro_f1": float(macro_f1),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "accuracy": float(acc),
        "per_cat": per_cat,
        "n": len(df),
        "n_total": len(preds_df),
        "n_unparseable": int(preds_df["pred"].isna().sum()),
    }


def confusion_plot(preds_df, strategy):
    df = preds_df.dropna(subset=["pred"]).copy()
    if len(df) == 0:
        return
    cm = confusion_matrix(df["true"], df["pred"], labels=CATEGORIES)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CATEGORIES, yticklabels=CATEGORIES, ax=ax,
        cbar_kws={"label": "count"},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix - strategy '{strategy}' (n={len(df)})")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    save_fig(f"rq2_confusion_matrix_{strategy}", fig)


def per_category_f1_plot(metrics, strategy):
    cats = CATEGORIES
    f1s = [metrics["per_cat"][c]["f1"] for c in cats]
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#3b7dd8" if v >= 0.7 else "#e07a3a" if v >= 0.5 else "#aaaaaa" for v in f1s]
    bars = ax.barh(cats, f1s, color=colors, edgecolor="black")
    ax.axvline(0.7, color="red", linestyle="--", label="Production threshold = 0.70")
    ax.axvline(0.5, color="orange", linestyle=":", label="Minimum acceptable = 0.50")
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Per-category F1")
    ax.set_title(f"Per-category F1 - strategy '{strategy}' (macro-F1 = {metrics['macro_f1']:.3f})")
    for i, v in enumerate(f1s):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=9)
    ax.legend(loc="lower right")
    plt.tight_layout()
    save_fig(f"rq2_per_category_f1_{strategy}", fig)


def precision_recall_scatter(metrics, strategy):
    fig, ax = plt.subplots(figsize=(8, 6))
    for cat in CATEGORIES:
        m = metrics["per_cat"][cat]
        size = max(50, m["support"] * 30)
        ax.scatter(m["precision"], m["recall"], s=size, alpha=0.6, label=f"{cat} (n={m['support']})")
        ax.annotate(cat, (m["precision"] + 0.01, m["recall"] + 0.01), fontsize=8)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="P = R")
    ax.set_xlabel("Precision")
    ax.set_ylabel("Recall")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"Precision vs recall per category - strategy '{strategy}' (bubble size = test support)")
    plt.tight_layout()
    save_fig(f"rq2_precision_recall_scatter_{strategy}", fig)


# ---------------------------------------------------------------------------
# Cross-strategy figures
# ---------------------------------------------------------------------------

def strategy_comparison_plot(all_metrics):
    strategies = list(all_metrics.keys())
    macro_f1 = [all_metrics[s]["macro_f1"] for s in strategies]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#3b7dd8" if v >= 0.7 else "#e07a3a" if v >= 0.5 else "#aaaaaa" for v in macro_f1]
    bars = ax.bar(strategies, macro_f1, color=colors, edgecolor="black")
    ax.axhline(0.7, color="red", linestyle="--", label="Production threshold = 0.70")
    ax.axhline(0.5, color="orange", linestyle=":", label="Minimum acceptable = 0.50")
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 1.0)
    ax.set_title("Strategy comparison - macro-F1 on held-out test set")
    for bar, v in zip(bars, macro_f1):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)
    ax.legend()
    plt.tight_layout()
    save_fig("rq2_strategy_comparison", fig)


def macro_metrics_grouped_bar(all_metrics):
    strategies = list(all_metrics.keys())
    metrics_keys = ["macro_precision", "macro_recall", "macro_f1", "accuracy"]
    labels = ["Macro-P", "Macro-R", "Macro-F1", "Accuracy"]
    data = np.array([[all_metrics[s][k] for k in metrics_keys] for s in strategies])

    x = np.arange(len(strategies))
    w = 0.2
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, lab in enumerate(labels):
        ax.bar(x + i * w - 1.5 * w, data[:, i], w, label=lab, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-strategy aggregate metrics on held-out test set")
    ax.axhline(0.7, color="red", linestyle="--", alpha=0.5)
    ax.legend(ncol=4, loc="lower right")
    plt.tight_layout()
    save_fig("rq2_macro_metrics_grouped", fig)


def per_cat_f1_all_strategies(all_metrics):
    strategies = list(all_metrics.keys())
    df = pd.DataFrame(
        {s: [all_metrics[s]["per_cat"][c]["f1"] for c in CATEGORIES] for s in strategies},
        index=CATEGORIES,
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    df.plot(kind="bar", ax=ax, edgecolor="black")
    ax.axhline(0.7, color="red", linestyle="--", label="Production threshold = 0.70")
    ax.set_ylabel("F1 score")
    ax.set_xlabel("LOS level-1 category")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-category F1 across all four strategies")
    ax.set_xticklabels(df.index, rotation=30, ha="right")
    ax.legend(title="Strategy", ncol=5, loc="lower right")
    plt.tight_layout()
    save_fig("rq2_per_category_f1_all_strategies", fig)
    df.to_csv(RESULTS_DIR / "per_category_f1_all_strategies.csv")
    return df


def delta_f1_baseline_vs_best(all_metrics, baseline="baseline"):
    if baseline not in all_metrics:
        return
    enriched = [s for s in all_metrics if s != baseline]
    if not enriched:
        return
    best = max(enriched, key=lambda s: all_metrics[s]["macro_f1"])
    deltas = []
    for c in CATEGORIES:
        d = all_metrics[best]["per_cat"][c]["f1"] - all_metrics[baseline]["per_cat"][c]["f1"]
        deltas.append(d)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#3b7dd8" if v > 0 else "#e07a3a" for v in deltas]
    ax.bar(CATEGORIES, deltas, color=colors, edgecolor="black")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel(f"Δ F1 ({best} − {baseline})")
    ax.set_title(f"Per-category F1 improvement: {best} vs {baseline}")
    ax.set_xticklabels(CATEGORIES, rotation=30, ha="right")
    for i, d in enumerate(deltas):
        sign = "+" if d > 0 else ""
        ax.text(i, d + (0.01 if d >= 0 else -0.03), f"{sign}{d:.2f}", ha="center", fontsize=9)
    plt.tight_layout()
    save_fig(f"rq2_delta_f1_{baseline}_to_{best}", fig)


def confusion_diff_plot(all_preds, baseline="baseline", best=None):
    if best is None or best not in all_preds:
        return
    if baseline not in all_preds:
        return

    db = all_preds[baseline].dropna(subset=["pred"])
    de = all_preds[best].dropna(subset=["pred"])
    common = db.merge(de, on="url", suffixes=("_b", "_e"))
    if len(common) == 0:
        return

    fixed = common[(common["true_b"] == common["pred_e"]) & (common["true_b"] != common["pred_b"])]
    broken = common[(common["true_b"] == common["pred_b"]) & (common["true_b"] != common["pred_e"])]
    persistent = common[(common["true_b"] != common["pred_b"]) & (common["true_b"] != common["pred_e"])]
    correct = common[(common["true_b"] == common["pred_b"]) & (common["true_b"] == common["pred_e"])]

    print(f"\n=== Diff {baseline} -> {best} ===")
    print(f"  fixed by enrichment: {len(fixed)}")
    print(f"  broken by enrichment: {len(broken)}")
    print(f"  persistent errors: {len(persistent)}")
    print(f"  correctly classified by both: {len(correct)}")

    counts = pd.DataFrame({
        "Fixed by enrichment": [len(fixed)],
        "Broken by enrichment": [len(broken)],
        "Persistent errors": [len(persistent)],
        "Both correct": [len(correct)],
    }, index=[f"{baseline} → {best}"])

    fig, ax = plt.subplots(figsize=(9, 4))
    counts.plot(kind="barh", stacked=True, ax=ax,
                color=["#3b7dd8", "#e07a3a", "#aaaaaa", "#5cb85c"],
                edgecolor="black")
    ax.set_xlabel("Number of test documents")
    ax.set_title(f"Where keyword enrichment helps and hurts: {baseline} → {best}")
    for i, total in enumerate([len(common)]):
        offset = 0
        for col, val in counts.iloc[i].items():
            if val > 0:
                ax.text(offset + val/2, i, f"{int(val)}", ha="center", va="center", fontsize=10, color="white")
            offset += val
    ax.legend(loc="lower right")
    plt.tight_layout()
    save_fig(f"rq2_diff_{baseline}_to_{best}", fig)

    # also save a sample of fixed/broken examples for the report
    sample = pd.concat([
        fixed.head(5).assign(category="FIXED"),
        broken.head(5).assign(category="BROKEN"),
        persistent.head(5).assign(category="PERSISTENT"),
    ])
    if len(sample):
        sample[["category", "url", "tittel_b", "true_b", "pred_b", "pred_e"]].to_csv(
            RESULTS_DIR / f"diff_{baseline}_to_{best}_examples.csv", index=False
        )


def keywords_table_plot(es_terms, spacy_terms, top_n=8):
    """Visual table showing top keywords per category for each source."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis("off")
    rows = ["Category", "ES top terms (filtered)", "spaCy top phrases"]
    table_data = []
    for cat in CATEGORIES:
        es_top = es_terms.get(cat, [])[:top_n]
        sp_top = spacy_terms.get(cat, [])[:top_n]
        table_data.append([
            cat,
            ", ".join(es_top) if es_top else "-",
            ", ".join(sp_top) if sp_top else "-",
        ])

    table = ax.table(
        cellText=table_data,
        colLabels=rows,
        cellLoc="left", colLoc="left", loc="center",
        colWidths=[0.20, 0.40, 0.40],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)
    ax.set_title("Top category-indicative terms per LOS category and keyword source", pad=12)
    plt.tight_layout()
    save_fig("rq2_top_terms_per_category", fig)


# ---------------------------------------------------------------------------
# Save figures helper
# ---------------------------------------------------------------------------

def save_fig(name, fig):
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global ACTIVE_MODEL, DATA_DIR, RESULTS_DIR, FIG_DIR, CONFIDENCE_MODE
    parser = argparse.ArgumentParser()
    parser.add_argument("--mini", action="store_true", help="run on a small subset for sanity")
    parser.add_argument("--strategies", nargs="+",
                        default=["baseline", "es", "spacy", "hybrid"])
    parser.add_argument("--model", choices=["haiku", "gpt41", "gpt51"], default="haiku",
                        help="which LLM to call (default haiku)")
    parser.add_argument("--data-dir", default=None,
                        help="override data directory (must contain dataset_train.jsonl and dataset_test.jsonl)")
    parser.add_argument("--output-dir", default=None,
                        help="override results output directory")
    parser.add_argument("--no-figures", action="store_true",
                        help="skip per-strategy and cross-strategy figure generation")
    parser.add_argument("--no-eda", action="store_true",
                        help="skip EDA figure generation (class_distribution, doc_length, etc)")
    parser.add_argument("--confidence", action="store_true",
                        help="ask the classifier for JSON {kategori, confidence, begrunnelse} "
                             "and record confidence/reasoning (for calibration + abstain analysis)")
    args = parser.parse_args()

    ACTIVE_MODEL = args.model
    CONFIDENCE_MODE = args.confidence
    if args.data_dir:
        DATA_DIR = Path(args.data_dir)
    if args.output_dir:
        RESULTS_DIR = Path(args.output_dir)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    np.random.seed(SEED)
    random.seed(SEED)
    sns.set_theme(style="whitegrid")

    print("=" * 80)
    print(f"RQ2 LOS Classification - End-to-end pipeline (model={ACTIVE_MODEL})")
    print(f"  data_dir={DATA_DIR}")
    print(f"  output_dir={RESULTS_DIR}")
    print("=" * 80)

    # 1. Load data
    train_df = load_jsonl(DATA_DIR / "dataset_train.jsonl")
    test_df = load_jsonl(DATA_DIR / "dataset_test.jsonl")
    print(f"Train: {len(train_df)} docs")
    print(f"Test : {len(test_df)} docs")

    # 2. EDA figures (optional)
    if not args.no_eda:
        print("\n--- EDA ---")
        counts = class_distribution_plot(train_df, test_df)
        length_stats = doc_length_plot(train_df, test_df)
    else:
        counts = None
        length_stats = None

    # 3. Term banks - only built if a requested strategy actually needs keywords.
    # baseline and summary use no term bank, so a `baseline summary` run skips
    # Elasticsearch and spaCy entirely (no localhost:9200 dependency).
    needs_keywords = any(s in ("es", "spacy", "hybrid") for s in args.strategies)
    if needs_keywords:
        print("\n--- Significant terms (ES) ---")
        es_raw, es_clean = build_es_term_bank()
        for cat in CATEGORIES:
            print(f"  {cat[:30]:30s} kept {len(es_clean[cat])} of {len(es_raw[cat])}")
        if not args.no_figures:
            terms_kept_vs_removed_plot(es_raw)
            term_overlap_heatmap(es_clean)

        print("\n--- spaCy noun-phrase keywords ---")
        spacy_clean = build_spacy_term_bank(train_df)
        for cat in CATEGORIES:
            print(f"  {cat[:30]:30s} {len(spacy_clean[cat])} phrases")

        hybrid_clean = {}
        for cat in CATEGORIES:
            merged = list(dict.fromkeys(es_clean[cat] + spacy_clean[cat]))
            hybrid_clean[cat] = merged[:15]

        if not args.no_figures:
            keywords_table_plot(es_clean, spacy_clean)
    else:
        print("\n--- Skipping ES + spaCy term banks (no keyword strategy requested) ---")
        es_clean = {c: [] for c in CATEGORIES}
        spacy_clean = {c: [] for c in CATEGORIES}
        hybrid_clean = {c: [] for c in CATEGORIES}

    # 4. Few-shot examples
    print("\n--- Few-shot examples ---")
    fewshot = fetch_fewshot_examples(train_df, k_per_cat=2)
    fewshot_block = format_fewshot_block(fewshot)
    print(f"  total few-shot examples: {sum(len(v) for v in fewshot.values())}")

    # 5. Mini-test or full
    if args.mini:
        # take min(2, n) per category
        sub = []
        for cat in CATEGORIES:
            sc = test_df[test_df["nivaa1"] == cat].head(2)
            sub.append(sc)
        test_df_run = pd.concat(sub, ignore_index=True)
        print(f"\n[MINI MODE] Running on {len(test_df_run)} documents")
    else:
        test_df_run = test_df

    # 6. Classify per strategy
    all_preds = {}
    all_metrics = {}
    for strategy in args.strategies:
        preds = classify_strategy(strategy, test_df_run, es_clean, spacy_clean, hybrid_clean, fewshot_block)
        all_preds[strategy] = preds
        # save predictions FIRST so expensive API work is never lost if a later
        # metric/print step fails (e.g. all-unparseable edge cases).
        preds.to_csv(RESULTS_DIR / f"predictions_{strategy}.csv", index=False)
        m = compute_metrics(preds)
        all_metrics[strategy] = m
        print(f"\n  [{strategy}] macro-F1={m['macro_f1']:.3f}  macro-P={m['macro_precision']:.3f}  "
              f"macro-R={m['macro_recall']:.3f}  acc={m['accuracy']:.3f}  "
              f"(n_pred={m['n']}, unparseable={m.get('n_unparseable', 0)})")

        # per-strategy plots
        if not args.no_figures:
            confusion_plot(preds, strategy)
            per_category_f1_plot(m, strategy)
            precision_recall_scatter(m, strategy)

    # 7. Cross-strategy plots
    if not args.no_figures and len(all_metrics) > 1:
        print("\n--- Cross-strategy figures ---")
        strategy_comparison_plot(all_metrics)
        macro_metrics_grouped_bar(all_metrics)
        per_cat_f1_all_strategies(all_metrics)
        delta_f1_baseline_vs_best(all_metrics, baseline="baseline")
        if "baseline" in all_metrics:
            enriched = [s for s in all_metrics if s != "baseline"]
            if enriched:
                best = max(enriched, key=lambda s: all_metrics[s]["macro_f1"])
                confusion_diff_plot(all_preds, baseline="baseline", best=best)

    # 8. Save consolidated results JSON
    consolidated = {
        "n_train": len(train_df),
        "n_test": len(test_df),
        "n_test_run": len(test_df_run),
        "categories": CATEGORIES,
        "class_distribution": counts.to_dict() if counts is not None else None,
        "doc_length_stats": length_stats,
        "metrics": all_metrics,
        "term_banks": {
            "es": es_clean,
            "spacy": spacy_clean,
            "hybrid": hybrid_clean,
        },
    }
    with open(RESULTS_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(consolidated, f, ensure_ascii=False, indent=2)
    print(f"\nFinal results JSON saved to {RESULTS_DIR / 'results.json'}")

    # 9. Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'strategy':<10} {'macro-F1':>10} {'macro-P':>10} {'macro-R':>10} {'accuracy':>10} {'meets 0.70':>12}")
    print("-" * 80)
    for s in args.strategies:
        m = all_metrics[s]
        meets = "yes" if m["macro_f1"] >= 0.70 else "no"
        print(f"{s:<10} {m['macro_f1']:>10.3f} {m['macro_precision']:>10.3f} {m['macro_recall']:>10.3f} {m['accuracy']:>10.3f} {meets:>12}")
    print("=" * 80)


if __name__ == "__main__":
    main()
