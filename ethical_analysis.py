# Phase 7: Ethical Considerations & Bias
import re
import math
import hashlib
from collections import Counter
from typing import List, Dict
from prepare_data.preprocessing import Preprocessor 

# ===== Compact internal helpers ==============================================
_TOKEN_RE = re.compile(r"[a-z']+")

def _tokenize(x: str) -> List[str]:
    return _TOKEN_RE.findall(x.lower())

def _normalize01(labels: List[int]) -> List[int]:
    return [1 if y in (1, "1", True) else 0 for y in labels]

def _per_1k(cnt: int, tot_tokens: int) -> float:
    return (cnt / max(tot_tokens, 1)) * 1000.0

def _hash16(s: str) -> str:
    return hashlib.md5(s.strip().lower().encode("utf-8")).hexdigest()[:16]

# Gender lexicon (includes neutral/non-binary)
_GENDER = {
    "male":   ["he","him","his","man","men","boy","male","husband","boyfriend"],
    "female": ["she","her","hers","woman","women","girl","female","wife","girlfriend"],
    "neutral":["they","them","their","theirs","nonbinary","non-binary"],
}

def _gender_per_1k(texts: List[str]) -> Dict[str, float]:
    total, c = 0, Counter()
    for t in texts:
        toks = _tokenize(t)
        total += len(toks)
        for k, terms in _GENDER.items():
            c[k] += sum(1 for w in toks if w in terms)
    return {k: _per_1k(v, total) for k, v in c.items()}

def _pmi_terms(reviews: List[str], labels: List[int], min_df=20, topk=12):
    """Return (top_pos, top_neg) by shifted PMI."""
    y = _normalize01(labels)
    N, Np = len(reviews), sum(y)
    Nn = N - Np
    df_all, df_pos, df_neg = Counter(), Counter(), Counter()
    for txt, lab in zip(reviews, y):
        toks = set(_tokenize(txt))
        for w in toks:
            df_all[w] += 1
            (df_pos if lab == 1 else df_neg)[w] += 1
    scores = []
    for w, df in df_all.items():
        if df < min_df: 
            continue
        p_w = df / N
        p_w_pos = df_pos[w] / max(Np, 1)
        p_w_neg = df_neg[w] / max(Nn, 1)
        pmi_pos = math.log((p_w_pos / max(p_w, 1e-12)) + 1e-12)
        pmi_neg = math.log((p_w_neg / max(p_w, 1e-12)) + 1e-12)
        scores.append((w, pmi_pos - pmi_neg, df))
    top_pos = sorted(scores, key=lambda x: x[1], reverse=True)[:topk]
    top_neg = sorted(scores, key=lambda x: x[1])[:topk]
    return top_pos, top_neg

def _near_dups(texts: List[str]) -> int:
    seen, dup = set(), 0
    for t in texts:
        h = _hash16(t)
        if h in seen: dup += 1
        else: seen.add(h)
    return dup

# ===== (1) Analysis Bias =================================
def analyze_bias():
    """Section (1): Bias analysis (fairness across domains, gendered language, spurious correlations)."""
    preprocessor = Preprocessor()
    reviews, labels = preprocessor.load_reviews()
    y = _normalize01(labels)

    # (1.1) Gendered Language — per-1k (includes neutral)
    g_perk = _gender_per_1k(reviews)
    print("[Gendered Language] per 1k tokens:", {k: round(v, 3) for k, v in g_perk.items()})

    # (1.2) Fairness Across Domains — distribution-only by keywords (dataset-only)
    domains = ['electronics', 'books', 'movies', 'kitchen', 'toys']
    dom_stats = {}
    for d in domains:
        total = sum(d in r.lower() for r in reviews)
        pos = sum(d in r.lower() and yy == 1 for r, yy in zip(reviews, y))
        neg = sum(d in r.lower() and yy == 0 for r, yy in zip(reviews, y))
        dom_stats[d] = {"total": total, "positive": pos, "negative": neg, "pos_rate": round(pos / max(total,1), 3)}
    print("[Fairness Across Domains] distribution:", dom_stats)

    # (1.3) Spurious Correlations — PMI + heuristics
    pos_top, neg_top = _pmi_terms(reviews, y, min_df=20, topk=12)
    print("[Spurious/PMI] Positive-leaning terms (top):", pos_top)
    print("[Spurious/PMI] Negative-leaning terms (top):", neg_top)

    # Additional heuristics: review length and brand mentions
    lengths = [len(r.split()) for r in reviews]
    avg_len = sum(lengths)/len(lengths) if lengths else 0.0
    brands = ['sony', 'samsung', 'harry', 'potter', 'apple', 'lg', 'panasonic']
    brand_hits = sum(any(b in r.lower() for b in brands) for r in reviews)
    print(f"[Heuristics] Avg length: {avg_len:.2f} | Brand/name mentions: {brand_hits}")

    # Data quality: near-duplicates
    dups = _near_dups(reviews)
    print(f"[Data Quality] Near-duplicate indicator (hash16): {dups}")

# ===== (2) Sentiment Polarity =================================
def analyze_sentiment_polarity():
    """Section (2): Sentiment binarization & label distribution."""
    preprocessor = Preprocessor()
    reviews, labels = preprocessor.load_reviews()
    y = _normalize01(labels)
    n = len(y)
    pos = sum(y)
    neg = n - pos
    print(f"[Sentiment Polarity] Total: {n} | Positive: {pos} ({pos/n:.2%}) | Negative: {neg} ({neg/n:.2%})")
    if set(labels) - {0, 1}:
        print("⚠️ Detected labels outside {0,1} (e.g. 3-star). Consider tri-class model or keeping 'neutral'.")

# ===== (3) Context Sensitivity =================================
def analyze_context_sensitivity():
    """Section (3): Context sensitivity (sarcasm, double negation) — estimation."""
    preprocessor = Preprocessor()
    reviews, _ = preprocessor.load_reviews()
    neg1 = ['not', "n't", 'never', 'no']
    neg2_patterns = ["not.*not", "never.*no", "no.*never"]
    sarcasm = ['yeah right', 'as if', 'sure', 'great... not', 'what a joke']
    neg1_cnt = sum(any(t in r.lower() for t in neg1) for r in reviews)
    neg2_cnt = sum(any(re.search(p, r.lower()) for p in neg2_patterns) for r in reviews)
    sarcasm_cnt = sum(any(t in r.lower() for t in sarcasm) for r in reviews)
    print(f"[Context] Negation: {neg1_cnt} | Double-negation (est.): {neg2_cnt} | Sarcasm cues: {sarcasm_cnt}")

# ===== Report ===============================================================
def create_bias_mitigation_report():
    """Create report on bias mitigation strategies"""
    report = """## Ethical Considerations and Bias Mitigation

### 1. Bias Analysis
- Issue: The dataset may contain various types of bias:
    - **Fairness Across Domains**: Uneven distribution among Books/DVDs/Electronics/Kitchen/Toys.
    - **Gendered Language**: Use of gendered terms (male/female/neutral/non-binary).
    - **Spurious Correlations**: Shortcut features such as brand names, proper nouns, review length.
- Mitigation:
    - Report domain distribution, monitor positive/negative rates.
    - Track gendered language density (per-1k tokens), consider anonymization/normalization.
    - Audit with PMI/log-odds, remove/regularize shortcut features, add neutral data.
    - Monitor data quality (near-duplicates).

### 2. Sentiment Polarity (Limitation)
- Issue: Sentiment binarization (pos/neg) may be too simplistic, ignoring “neutral/3-star”.
- Mitigation: Add confidence; consider tri-class model (neg/neu/pos) or keep “neutral” in the pipeline; document trade-offs.

### 3. Context Sensitivity (Limitation)
- Issue: Sarcasm, double negation, and cultural context may mislead the model.
- Mitigation: Stress tests/perturbations; add counterfactual examples; clearly state limitations in documentation.

### 4. Transparency
- Issue: Users need clarity on label mapping, preprocessing, and known risks.
- Mitigation: Maintain a short “datasheet” for dataset assumptions and coverage; describe pipeline and limitations.
"""
    with open("ethical_considerations.md", "w", encoding="utf-8") as f:
        f.write(report)    
    
if __name__ == "__main__":
    print("--- Test analyze_sentiment_polarity ---")
    analyze_sentiment_polarity()
    print("\n--- Test analyze_context_sensitivity ---")
    analyze_context_sensitivity()
    print("\n--- Test analyze_bias ---")
    analyze_bias()
    print("\n--- Test create_bias_mitigation_report ---")
    create_bias_mitigation_report()
    print("ethical_considerations.md file created!")
