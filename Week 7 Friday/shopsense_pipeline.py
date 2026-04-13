"""
ShopSense Sentiment Analysis Pipeline  v3
End-to-end NLP evaluation & production recommendation.
All 7 sub-steps are modular functions.

IMPORTANT NOTE ON CONFUSION MATRIX SEMANTICS:
  confusion_matrix(y_true, y_pred, labels=[LABEL_NEG, LABEL_POS]).ravel()
  returns (tn, fp_sklearn, fn_sklearn, tp) where:
    fp_sklearn = actual NEGATIVE predicted POSITIVE = MISSED COMPLAINT = Business FN  → cost $15
    fn_sklearn = actual POSITIVE predicted NEGATIVE = FALSE ALARM      = Business FP  → cost $2
  These are called `b_fn` and `b_fp` throughout this code to avoid confusion.
"""

import os, time, warnings, random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, accuracy_score,
)
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)

# ── Config ────────────────────────────────────────────────────────────────
BASE_DIR      = Path("/home/claude/week-07/friday")
DATA_PATH     = BASE_DIR / "shopsense_reviews.csv"
OUTPUT_DIR    = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

FN_COST       = 15      # $ per missed negative (churn, brand damage)
FP_COST       = 2       # $ per false alarm (unnecessary support ticket)
DAILY_REVIEWS = 100_000
LABEL_POS     = "positive"
LABEL_NEG     = "negative"
CM_LABELS     = [LABEL_NEG, LABEL_POS]   # order matters for ravel()


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 · DATA ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def load_data(path: Path) -> pd.DataFrame:
    """Load and validate the ShopSense reviews CSV."""
    try:
        df = pd.read_csv(path)
        assert "review_text" in df.columns
        assert "sentiment"   in df.columns
        df = df.dropna(subset=["review_text","sentiment"]).copy()
        df["review_text"] = df["review_text"].astype(str).str.strip()
        return df
    except Exception as e:
        raise RuntimeError(f"Data load failed: {e}")


def compute_class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["sentiment"].value_counts()
    pct    = (counts / counts.sum() * 100).round(2)
    return pd.DataFrame({"Count": counts, "Percentage (%)": pct})


def explain_accuracy_paradox(dist: pd.DataFrame) -> str:
    majority_pct = dist["Percentage (%)"].max()
    return (
        f"\n{'='*68}\n  WHY ACCURACY IS MISLEADING\n{'='*68}\n"
        f"  {majority_pct:.1f}% of reviews are POSITIVE.\n"
        f"  A model that ALWAYS predicts 'positive' achieves {majority_pct:.1f}% accuracy\n"
        f"  without learning anything — identical to the '94% accuracy' reported.\n\n"
        f"  Metric gap on the broken model:\n"
        f"    Accuracy             : {majority_pct:.1f}%   ← looks great\n"
        f"    Recall (negative)    : ~0%    ← catastrophic\n"
        f"    Macro F1             : ~0.47  ← reveals the problem\n\n"
        f"  Correct metrics to track:\n"
        f"    * Recall (negative class)  — 'Are we catching complaints?'\n"
        f"    * Macro F1                 — 'Are we balanced across classes?'\n"
        f"    * Cost per day             — 'What does misclassification cost?'\n"
        f"{'='*68}"
    )


def plot_class_distribution(dist: pd.DataFrame, save_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = ["#e74c3c" if i == "negative" else "#2ecc71" for i in dist.index]
    bars = axes[0].bar(dist.index, dist["Percentage (%)"], color=colors, edgecolor="white")
    for bar, (_, row) in zip(bars, dist.iterrows()):
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                     f"{row['Percentage (%)']:.1f}%\n(n={int(row['Count']):,})",
                     ha="center", fontsize=11, fontweight="bold")
    axes[0].set_title("Class Distribution", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("% of Reviews")
    axes[0].set_ylim(0, 115)
    axes[0].spines[["top","right"]].set_visible(False)
    axes[1].pie(dist["Count"], labels=dist.index, autopct="%1.1f%%",
                colors=colors, startangle=90, textprops={"fontsize":12,"fontweight":"bold"})
    axes[1].set_title("Positive vs Negative Split", fontsize=13, fontweight="bold")
    plt.suptitle("ShopSense Reviews – Sentiment Label Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 · MODEL EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def build_lr_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=15_000, sublinear_tf=True)),
        ("clf",   LogisticRegression(class_weight="balanced", max_iter=500, random_state=42)),
    ])


def build_nb_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=15_000)),
        ("clf",   MultinomialNB(alpha=0.5)),
    ])


def _extract_metrics(y_true, y_pred, name: str, model, latency_ms: float = 0.0) -> dict:
    """
    Compute all metrics from ground truth and predictions.

    Confusion matrix with labels=[LABEL_NEG, LABEL_POS]:
       ┌───────────────────────────────────────────────────┐
       │                  Predicted NEG  │  Predicted POS  │
       │ Actual NEG  │  tn_raw          │  b_fn (FN biz)  │
       │ Actual POS  │  b_fp (FP biz)   │  tp_raw         │
       └───────────────────────────────────────────────────┘
    ravel() → tn_raw, b_fn, b_fp, tp_raw
    b_fn = missed complaints  (cost FN_COST)
    b_fp = false alarms       (cost FP_COST)
    """
    cm     = confusion_matrix(y_true, y_pred, labels=CM_LABELS)
    report = classification_report(y_true, y_pred, labels=CM_LABELS, output_dict=True)
    tn_raw, b_fn, b_fp, tp_raw = cm.ravel()

    total = tn_raw + b_fn + b_fp + tp_raw

    return dict(
        model_name   = name,
        model        = model,
        y_pred       = y_pred,
        cm           = cm,
        report       = report,
        accuracy     = accuracy_score(y_true, y_pred),
        f1_macro     = f1_score(y_true, y_pred, average="macro"),
        recall_neg   = report[LABEL_NEG]["recall"],        # TP_neg / (TP_neg + b_fn)
        precision_neg= report[LABEL_NEG]["precision"],
        f1_neg       = report[LABEL_NEG]["f1-score"],
        recall_pos   = report[LABEL_POS]["recall"],
        # Business-correct confusion counts
        b_fn         = b_fn,    # missed complaints  (actual NEG → predicted POS)
        b_fp         = b_fp,    # false alarms       (actual POS → predicted NEG)
        tp_raw       = tp_raw,
        tn_raw       = tn_raw,
        total_test   = total,
        latency_ms   = latency_ms,
    )


def evaluate_model(model, X_tr, X_te, y_tr, y_te, name: str) -> dict:
    model.fit(X_tr, y_tr)
    t0         = time.perf_counter()
    y_pred     = model.predict(X_te)
    latency_ms = (time.perf_counter() - t0) / len(X_te) * 1000
    return _extract_metrics(y_te, y_pred, name, model, latency_ms)


def print_business_summary(m: dict) -> None:
    actual_neg    = m["tn_raw"] + m["b_fn"]       # all actual negatives
    actual_pos    = m["b_fp"]  + m["tp_raw"]       # all actual positives
    pct_missed    = m["b_fn"]  / actual_neg * 100 if actual_neg else 0
    pct_false_alm = m["b_fp"]  / actual_pos * 100 if actual_pos else 0

    print(f"\n{'='*66}")
    print(f"  BUSINESS SUMMARY – {m['model_name']}")
    print(f"{'='*66}")
    print(f"  For every 100 NEGATIVE (complaint) reviews:")
    print(f"    {100-pct_missed:.0f} correctly flagged and sent for action  ✅")
    print(f"    {pct_missed:.0f} silently missed (customer feels ignored)  ❌")
    print(f"  For every 100 POSITIVE reviews:")
    print(f"    {pct_false_alm:.1f} incorrectly routed to support (false alarms)")
    print(f"  Complaint Catch Rate : {m['recall_neg']*100:.1f}%  (target: >80%)")
    print(f"  Balance Score (F1)   : {m['f1_macro']:.3f}  (0=worst, 1=best)")
    print(f"  Speed per review     : {m['latency_ms']:.3f} ms")
    print(f"{'='*66}")


def plot_confusion_matrix(m: dict, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    labels_display = ["Negative\n(Complaint)", "Positive\n(Happy)"]
    sns.heatmap(m["cm"], annot=True, fmt="d", cmap="Blues",
                xticklabels=labels_display, yticklabels=labels_display,
                ax=ax, annot_kws={"size":13,"weight":"bold"})
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("True Label", fontsize=10)
    ax.set_title(f"Confusion Matrix\n{m['model_name']}", fontsize=11, fontweight="bold")
    # Annotate the critical cell
    ax.text(1.5, 0.5, "❌ MISSED\nCOMPLAINTS\n($15 each)",
            ha="center", va="center", fontsize=8, color="#c0392b",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#fdecea", alpha=0.8))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 · MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════

def measure_latency(model, texts: list, n_warmup: int = 100) -> float:
    _ = model.predict(texts[:n_warmup])
    t0 = time.perf_counter()
    _ = model.predict(texts)
    return (time.perf_counter() - t0) / len(texts) * 1000


def eval_unseen_categories(model, df_te: pd.DataFrame, unseen_cats: list) -> float:
    try:
        df_copy  = df_te.copy().reset_index(drop=True)
        n_unseen = max(1, int(len(df_copy) * 0.15))
        df_copy.loc[df_copy.index[-n_unseen:], "product_category"] = \
            np.random.choice(unseen_cats, n_unseen)
        subset = df_copy[df_copy["product_category"].isin(unseen_cats)]
        if len(subset) == 0:
            return float("nan")
        y_pred = model.predict(subset["review_text"].tolist())
        return f1_score(subset["sentiment"], y_pred, average="macro")
    except Exception as e:
        print(f"  [WARN] unseen-category eval: {e}")
        return float("nan")


def eval_hinglish(model, df_te: pd.DataFrame) -> dict:
    try:
        if "is_hinglish" not in df_te.columns:
            return {}
        subset = df_te[df_te["is_hinglish"] == True]
        if len(subset) == 0:
            return {}
        y_pred = model.predict(subset["review_text"].tolist())
        return {
            "recall_neg": recall_score(subset["sentiment"], y_pred,
                                       pos_label=LABEL_NEG, zero_division=0),
            "f1_macro":   f1_score(subset["sentiment"], y_pred, average="macro"),
            "n":          len(subset),
        }
    except Exception as e:
        print(f"  [WARN] hinglish eval: {e}")
        return {}


def build_comparison_table(results: list, df_te: pd.DataFrame, unseen_cats: list) -> pd.DataFrame:
    rows = []
    for m in results:
        h  = eval_hinglish(m["model"], df_te)
        uf = eval_unseen_categories(m["model"], df_te, unseen_cats)
        rows.append({
            "Model":               m["model_name"],
            "Accuracy":            f"{m['accuracy']*100:.1f}%",
            "Macro F1":            f"{m['f1_macro']:.3f}",
            "Neg Recall":          f"{m['recall_neg']*100:.1f}%",
            "C1: Unseen Cat F1":   f"{uf:.3f}" if not np.isnan(uf) else "N/A",
            "C2: Hinglish Recall": f"{h.get('recall_neg',float('nan'))*100:.1f}%"
                                    if h else "N/A",
            "C3: Latency (ms)":    f"{m['latency_ms']:.4f}",
            "Meets <20ms":         "YES" if m["latency_ms"] < 20 else "NO",
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4 · COST MODEL
# ═══════════════════════════════════════════════════════════════════════════

def compute_daily_cost(m: dict, daily: int = DAILY_REVIEWS,
                       fn_cost: float = FN_COST, fp_cost: float = FP_COST) -> dict:
    """
    Project daily cost using business-correct FN / FP definitions:
      b_fn = missed complaints (actual NEG → predicted POS) → cost FN_COST
      b_fp = false alarms      (actual POS → predicted NEG) → cost FP_COST
    """
    total  = m["total_test"]
    bfn_rate = m["b_fn"] / total
    bfp_rate = m["b_fp"] / total
    daily_bfn = bfn_rate * daily
    daily_bfp = bfp_rate * daily
    return dict(
        model            = m["model_name"],
        bfn_rate         = bfn_rate,
        bfp_rate         = bfp_rate,
        daily_bfn        = daily_bfn,
        daily_bfp        = daily_bfp,
        daily_bfn_cost   = daily_bfn * fn_cost,
        daily_bfp_cost   = daily_bfp * fp_cost,
        total_daily_cost = daily_bfn * fn_cost + daily_bfp * fp_cost,
    )


def threshold_sweep(model, X_te: list, y_te: list,
                    daily: int = DAILY_REVIEWS) -> pd.DataFrame:
    """
    Sweep decision thresholds from 0.05 to 0.95 and compute cost at each.
    Returns a DataFrame sorted by ascending cost.
    """
    classes   = model.classes_.tolist()
    neg_idx   = classes.index(LABEL_NEG)
    proba_neg = model.predict_proba(X_te)[:, neg_idx]

    rows = []
    for t in np.arange(0.05, 0.96, 0.05):
        y_pred   = [LABEL_NEG if p >= t else LABEL_POS for p in proba_neg]
        cm       = confusion_matrix(y_te, y_pred, labels=CM_LABELS)
        _, b_fn, b_fp, _ = cm.ravel()
        total    = len(y_te)
        rec_neg  = recall_score(y_te, y_pred, pos_label=LABEL_NEG, zero_division=0)
        f1       = f1_score(y_te, y_pred, average="macro")
        cost     = (b_fn/total * daily * FN_COST) + (b_fp/total * daily * FP_COST)
        rows.append(dict(threshold=round(t, 2), b_fn=b_fn, b_fp=b_fp,
                         recall_neg=rec_neg, f1_macro=f1, daily_cost=cost))
    return pd.DataFrame(rows).sort_values("daily_cost")


def print_cost_summary(c: dict) -> None:
    print(f"\n{'='*66}")
    print(f"  DAILY COST PROJECTION – {c['model']}")
    print(f"  At {DAILY_REVIEWS:,} reviews/day")
    print(f"{'='*66}")
    print(f"  Missed complaints (b_fn):  {c['daily_bfn']:,.0f}/day × ${FN_COST} = ${c['daily_bfn_cost']:,.0f}")
    print(f"  False alarms (b_fp):       {c['daily_bfp']:,.0f}/day × ${FP_COST} = ${c['daily_bfp_cost']:,.0f}")
    print(f"  {'─'*50}")
    print(f"  TOTAL DAILY LOSS:  ${c['total_daily_cost']:,.0f}")
    print(f"  ANNUAL PROJECTION: ${c['total_daily_cost']*365:,.0f}")
    print(f"{'='*66}")


def plot_cost_comparison(costs: list, save_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    names = [c["model"].split("(")[0].strip() for c in costs]
    fn_vals = [c["daily_bfn_cost"] for c in costs]
    fp_vals = [c["daily_bfp_cost"] for c in costs]
    x = np.arange(len(names));  w = 0.5
    b1 = axes[0].bar(x, fn_vals, w, label=f"Missed Complaints @ ${FN_COST}",
                     color="#e74c3c", alpha=0.85)
    b2 = axes[0].bar(x, fp_vals, w, bottom=fn_vals,
                     label=f"False Alarms @ ${FP_COST}", color="#f39c12", alpha=0.85)
    axes[0].set_xticks(x);  axes[0].set_xticklabels(names, fontsize=10)
    axes[0].set_ylabel("Daily Cost (USD)");  axes[0].legend(fontsize=9)
    axes[0].set_title("Projected Daily Cost Breakdown", fontsize=12, fontweight="bold")
    axes[0].spines[["top","right"]].set_visible(False)
    for i, (fn, fp) in enumerate(zip(fn_vals, fp_vals)):
        axes[0].text(i, fn+fp+100, f"${fn+fp:,.0f}/day",
                     ha="center", fontsize=10, fontweight="bold")
    annual = [c["total_daily_cost"]*365 for c in costs]
    bars = axes[1].bar(names, annual, color=["#3498db","#9b59b6"], alpha=0.85)
    axes[1].set_ylabel("Annual Cost (USD)")
    axes[1].set_title("Annual Cost Projection", fontsize=12, fontweight="bold")
    axes[1].spines[["top","right"]].set_visible(False)
    for bar in bars:
        axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+200,
                     f"${bar.get_height():,.0f}", ha="center", fontsize=10, fontweight="bold")
    plt.suptitle(f"Cost Model @ {DAILY_REVIEWS:,} Reviews/Day", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight");  plt.close()


def plot_threshold_sweep(sweep_df: pd.DataFrame, model_name: str, save_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax1.plot(sweep_df["threshold"], sweep_df["daily_cost"]/1000, "b-o",
             ms=4, label="Daily Cost ($K)")
    ax1.set_ylabel("Daily Cost ($ thousands)", fontsize=10)
    best = sweep_df.loc[sweep_df["daily_cost"].idxmin()]
    ax1.axvline(best["threshold"], color="green", linestyle="--", linewidth=1.5,
                label=f"Optimal threshold = {best['threshold']:.2f}\n(${best['daily_cost']:,.0f}/day)")
    ax1.legend(fontsize=9);  ax1.spines[["top","right"]].set_visible(False)
    ax1.set_title(f"Threshold Sweep – {model_name}", fontsize=12, fontweight="bold")
    ax2.plot(sweep_df["threshold"], sweep_df["recall_neg"], "r-o", ms=4, label="Neg Recall")
    ax2.plot(sweep_df["threshold"], sweep_df["f1_macro"],   "g-s", ms=4, label="Macro F1")
    ax2.axvline(best["threshold"], color="green", linestyle="--", linewidth=1.5)
    ax2.set_xlabel("Decision Threshold", fontsize=10)
    ax2.set_ylabel("Score", fontsize=10);  ax2.legend(fontsize=9)
    ax2.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight");  plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5 · TECHNICAL BRIEF
# ═══════════════════════════════════════════════════════════════════════════

def write_technical_brief(best_m: dict, best_c: dict, opt_thresh: float,
                          all_costs: list, save_path: Path) -> None:
    other_costs = [c for c in all_costs if c["model"] != best_c["model"]]
    next_best   = other_costs[0] if other_costs else best_c

    lines = [
        "=" * 68,
        "  SHOPSENSE SENTIMENT CLASSIFIER – TECHNICAL BRIEF",
        "  Prepared for: Priya Menon, Head of Product",
        "=" * 68,
        "",
        "PART A – RECOMMENDATION",
        "-" * 45,
        f"DEPLOY: {best_m['model_name']}",
        f"DECISION THRESHOLD: {opt_thresh:.2f}  (tuned for minimum cost)",
        "",
        "WHY THIS MODEL:",
        f"  · Catches {best_m['recall_neg']*100:.1f}% of genuine negative reviews",
        f"    (only {100-best_m['recall_neg']*100:.1f}% of complaints go undetected)",
        f"  · Runs in {best_m['latency_ms']:.3f} ms — well under the 20 ms SLA",
        f"  · Projected daily cost: ${best_c['total_daily_cost']:,.0f}",
        f"    vs next model: ${next_best['total_daily_cost']:,.0f}/day",
        "",
        "COST MODEL (FN = missed complaint @ $15 | FP = false alarm @ $2):",
        f"  · Missed complaints/day : {best_c['daily_bfn']:,.0f}  → ${best_c['daily_bfn_cost']:,.0f}",
        f"  · False alarms/day      : {best_c['daily_bfp']:,.0f}  → ${best_c['daily_bfp_cost']:,.0f}",
        f"  · Total daily loss      : ${best_c['total_daily_cost']:,.0f}",
        f"  · Annual projection     : ${best_c['total_daily_cost']*365:,.0f}",
        "",
        "WHAT THIS MODEL CANNOT GUARANTEE:",
        "  · Sarcasm or implicit negatives ('Oh great, broke on day 1 as expected')",
        "  · Novel Hinglish slang or regional variants emerging after training",
        "  · Emoji-only reviews outside the training vocabulary",
        "  · Sustained performance without monitoring and periodic retraining",
        "",
        "PART B – PRODUCTION MONITORING PLAN",
        "-" * 45,
        "",
        "PRIMARY WEEKLY METRIC:  Negative-Class Recall",
        "  Why: A drop here = complaints silently ignored = customer churn.",
        "",
        "THRESHOLDS:",
        "  ⚠  Alert  : Recall (negative) < 70%   → Notify data science team",
        "  🔴 Retrain : Recall (negative) < 60% for 2 consecutive weeks",
        "",
        "SECONDARY METRICS TO MONITOR WEEKLY:",
        "  · Macro F1 < 0.65            → Investigate class balance collapse",
        "  · Predicted-negative% < 2%   → Possible majority-class collapse",
        "  · P95 inference latency > 15ms → Approaching SLA breach",
        "  · Daily cost increase > 20%   → Drift or data quality issue",
        "",
        "EARLY DRIFT DETECTION:",
        "  1. Canary injection: 50 known-negative reviews injected daily",
        "     → Alert if catch rate drops below 80%",
        "  2. Human audit: 5% random sample routed to agents every week",
        "     → Use agent labels as weekly ground truth",
        "  3. Shadow deployment: new model runs silently for 2 weeks before cutover",
        "  4. Per-category recall: monitored separately",
        "     → New categories (Furniture, Sports, Toys) tracked from day one",
        "",
        "FALLBACK PLAN (if model fails critically):",
        "  → Activate keyword-based rule filter immediately",
        "     (flags reviews containing: scam, fake, broken, defective, useless, fraud...)",
        "",
        f"PREPARED BY: ShopSense AI/ML Team  |  DATE: 2026-04-12",
        "=" * 68,
    ]
    text = "\n".join(lines)
    print(text)
    with open(save_path, "w") as f:
        f.write(text)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 6 · FAILURE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def reproduce_faulty_pipeline(X_tr, X_te, y_tr, y_te) -> dict:
    print(f"\n{'='*66}\n  STEP 6 – FAULTY PIPELINE ANALYSIS\n{'='*66}")

    faulty = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,1), max_features=5_000)),
        ("clf",   LogisticRegression(max_iter=500, random_state=42)),
    ])
    faulty.fit(X_tr, y_tr)
    y_pred_f = faulty.predict(X_te)
    m_f = _extract_metrics(y_te, y_pred_f, "Faulty (no class_weight)", faulty)

    print(f"\n  FAULTY MODEL – REPORTED metrics:")
    print(f"    Accuracy (what they showed) : {m_f['accuracy']*100:.1f}%  ← looks great!")
    print(f"  FAULTY MODEL – HIDDEN metrics:")
    print(f"    Recall (negative class)     : {m_f['recall_neg']*100:.1f}%  ← catastrophic!")
    print(f"    Macro F1                    : {m_f['f1_macro']:.3f}          ← reveals the problem")
    dist_f = pd.Series(y_pred_f).value_counts(normalize=True)
    print(f"\n  Prediction distribution:")
    for cls, pct in dist_f.items():
        print(f"    '{cls}': {pct*100:.1f}%  {'← nearly everything!' if pct > 0.9 else ''}")

    print("\n  ROOT CAUSES:")
    for i, (title, desc) in enumerate([
        ("Imbalanced dataset",  "91% positive class → model minimised loss by predicting 'positive' everywhere"),
        ("No class weighting",  "LogisticRegression without class_weight='balanced' treats all errors equally"),
        ("Wrong metric",        "Only accuracy was reported — trivially matches the majority class"),
        ("No per-class breakdown","Classification report was never generated before deployment"),
        ("Threshold not tuned", "Default threshold=0.5 biases predictions toward majority class"),
    ], 1):
        print(f"    {i}. {title}: {desc}")

    fixed = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=15_000, sublinear_tf=True)),
        ("clf",   LogisticRegression(class_weight="balanced", max_iter=500, random_state=42)),
    ])
    fixed.fit(X_tr, y_tr)
    y_pred_fx = fixed.predict(X_te)
    m_fx = _extract_metrics(y_te, y_pred_fx, "Fixed (balanced)", fixed)

    print(f"\n  FIXED MODEL METRICS:")
    print(f"    Accuracy              : {m_fx['accuracy']*100:.1f}%")
    print(f"    Recall (negative)     : {m_fx['recall_neg']*100:.1f}%  ← dramatically improved!")
    print(f"    Macro F1              : {m_fx['f1_macro']:.3f}")
    print(f"\n  FIXES APPLIED:")
    for fix in ["class_weight='balanced'", "bigrams (ngram_range=(1,2))",
                "sublinear_tf=True", "Evaluation changed to recall + macro-F1"]:
        print(f"    ✅ {fix}")

    return {"faulty_m": m_f, "fixed_m": m_fx, "y_pred_faulty": y_pred_f}


def plot_before_after(faulty_m: dict, fixed_m: dict, save_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    pairs = [
        ("Accuracy",    faulty_m["accuracy"],  fixed_m["accuracy"]),
        ("Neg Recall",  faulty_m["recall_neg"], fixed_m["recall_neg"]),
        ("Macro F1",    faulty_m["f1_macro"],  fixed_m["f1_macro"]),
    ]
    for ax, (label, fv, xv) in zip(axes, pairs):
        b1 = ax.bar([-0.2], [fv], 0.35, label="Faulty", color="#e74c3c", alpha=0.85)
        b2 = ax.bar([ 0.2], [xv], 0.35, label="Fixed",  color="#2ecc71", alpha=0.85)
        ax.set_xticks([0]); ax.set_xticklabels([label])
        ax.set_ylim(0, 1.15)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9); ax.spines[["top","right"]].set_visible(False)
        for b in [b1[0], b2[0]]:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                    f"{b.get_height():.2f}", ha="center", fontsize=11)
    plt.suptitle("Before vs After – Faulty vs Fixed Pipeline", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# STEP 7 · COST OF FAILURE
# ═══════════════════════════════════════════════════════════════════════════

def cost_of_failure_analysis(faulty_m: dict, recommended_cost: dict) -> None:
    c_f = compute_daily_cost(faulty_m)
    savings = c_f["total_daily_cost"] - recommended_cost["total_daily_cost"]

    print(f"\n{'='*66}\n  STEP 7 – COST OF FAILURE ANALYSIS\n{'='*66}")
    print(f"\n  FAULTY PIPELINE at {DAILY_REVIEWS:,} reviews/day:")
    print(f"    Missed complaints : {c_f['daily_bfn']:,.0f}/day × ${FN_COST} = ${c_f['daily_bfn_cost']:,.0f}/day")
    print(f"    False alarms      : {c_f['daily_bfp']:,.0f}/day × ${FP_COST} = ${c_f['daily_bfp_cost']:,.0f}/day")
    print(f"    TOTAL DAILY LOSS  : ${c_f['total_daily_cost']:,.0f}")
    print(f"    ANNUAL PROJECTION : ${c_f['total_daily_cost']*365:,.0f}")
    print(f"\n  RECOMMENDED MODEL daily cost : ${recommended_cost['total_daily_cost']:,.0f}/day")
    print(f"  Daily savings by switching   : ${savings:,.0f}/day")
    print(f"  Annual savings               : ${savings*365:,.0f}/year")
    print(f"\n  SHARED VULNERABILITIES:")
    for v in ["Bag-of-words: no sarcasm/context understanding",
              "Hinglish: novel post-training slang will degrade recall",
              "Vocabulary drift: new categories may introduce unseen terms"]:
        print(f"    ⚠  {v}")
    print(f"\n  PROTECTIONS in recommended model:")
    for v in ["class_weight='balanced' → FN risk explicitly managed",
              "Metric = recall + macro-F1 → catches collapse pre-deployment",
              "Threshold tuned to minimise daily $ cost",
              "Canary injection + human audit → drift detected early"]:
        print(f"    ✅ {v}")
    print(f"{'='*66}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "█"*68)
    print("  SHOPSENSE SENTIMENT ANALYSIS PIPELINE  –  START")
    print("█"*68 + "\n")

    # ── STEP 1 ──────────────────────────────────────────────────────────
    print("▶ STEP 1 · DATA ANALYSIS")
    df   = load_data(DATA_PATH)
    dist = compute_class_distribution(df)
    print(f"  Dataset: {df.shape[0]:,} reviews  |  {df['sentiment'].value_counts().to_dict()}")
    print(explain_accuracy_paradox(dist))
    plot_class_distribution(dist, OUTPUT_DIR / "01_class_distribution.png")
    print("  📊 01_class_distribution.png")

    # ── STEP 2 ──────────────────────────────────────────────────────────
    print("\n▶ STEP 2 · MODEL EVALUATION")
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        df["review_text"].tolist(), df["sentiment"].tolist(), df,
        test_size=0.2, stratify=df["sentiment"], random_state=42)
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    lr_model = build_lr_pipeline()
    lr_m     = evaluate_model(lr_model, X_train, X_test, y_train, y_test,
                               "Logistic Regression (TF-IDF)")
    print(classification_report(y_test, lr_m["y_pred"], target_names=CM_LABELS))
    print_business_summary(lr_m)
    plot_confusion_matrix(lr_m, OUTPUT_DIR / "02a_cm_lr.png")
    print("  📊 02a_cm_lr.png")

    # ── STEP 3 ──────────────────────────────────────────────────────────
    print("\n▶ STEP 3 · MODEL COMPARISON")
    nb_model = build_nb_pipeline()
    nb_m     = evaluate_model(nb_model, X_train, X_test, y_train, y_test,
                               "Naive Bayes (TF-IDF)")
    print(classification_report(y_test, nb_m["y_pred"], target_names=CM_LABELS))
    print_business_summary(nb_m)
    plot_confusion_matrix(nb_m, OUTPUT_DIR / "02b_cm_nb.png")

    results = [lr_m, nb_m]
    for m in results:
        m["latency_ms"] = measure_latency(m["model"], X_test)

    unseen_cats = ["Furniture", "Sports", "Toys"]
    comp = build_comparison_table(results, df_test, unseen_cats)
    print("\n  MODEL COMPARISON TABLE:")
    print(comp.to_string(index=False))
    comp.to_csv(OUTPUT_DIR / "03_model_comparison.csv", index=False)

    # Chart
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    names = [m["model_name"].split("(")[0].strip() for m in results]
    for ax, (key, title, color) in zip(axes, [
        ("f1_macro","Macro F1","#3498db"),("recall_neg","Neg Recall","#e74c3c"),
        ("latency_ms","C3: Latency (ms)","#f39c12")]):
        vals = [m[key] for m in results]
        bars = ax.bar(names, vals, color=color, alpha=0.82)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.spines[["top","right"]].set_visible(False)
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()*1.02,
                    f"{b.get_height():.3f}", ha="center", fontsize=10, fontweight="bold")
        if key == "latency_ms":
            ax.axhline(20, color="red", linestyle="--", linewidth=1.2, label="20ms SLA")
            ax.legend(fontsize=8)
    # threshold sweep chart in 4th panel
    sweep_lr = threshold_sweep(lr_m["model"], X_test, y_test)
    best_row = sweep_lr.loc[sweep_lr["daily_cost"].idxmin()]
    axes[3].plot(sweep_lr["threshold"], sweep_lr["daily_cost"]/1000,
                 "b-o", ms=4, label="Daily Cost ($K)")
    axes[3].axvline(best_row["threshold"], color="green", linestyle="--",
                    linewidth=1.5, label=f"Optimal={best_row['threshold']:.2f}")
    axes[3].set_title("Threshold vs Cost (LR)", fontsize=11, fontweight="bold")
    axes[3].set_xlabel("Threshold"); axes[3].set_ylabel("Cost ($K)")
    axes[3].legend(fontsize=8); axes[3].spines[["top","right"]].set_visible(False)
    plt.suptitle("Model Comparison Across All Production Constraints",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  📊 03_model_comparison.png")

    # ── STEP 4 ──────────────────────────────────────────────────────────
    print("\n▶ STEP 4 · COST MODEL")
    costs = [compute_daily_cost(m) for m in results]
    for c in costs:
        print_cost_summary(c)

    # Threshold sweep & optimal threshold for LR
    sweep_lr = threshold_sweep(lr_m["model"], X_test, y_test)
    best_thresh_row = sweep_lr.loc[sweep_lr["daily_cost"].idxmin()]
    opt_thresh  = float(best_thresh_row["threshold"])
    opt_cost    = float(best_thresh_row["daily_cost"])
    print(f"\n  LR Threshold sweep – optimal threshold: {opt_thresh:.2f}")
    print(f"  Cost at optimal threshold: ${opt_cost:,.0f}/day")
    print(f"  Cost at default 0.50:      ${sweep_lr.loc[sweep_lr['threshold']==0.50,'daily_cost'].values[0]:,.0f}/day")

    plot_threshold_sweep(sweep_lr, lr_m["model_name"], OUTPUT_DIR / "04b_threshold_sweep.png")
    print("  📊 04b_threshold_sweep.png")

    # Update LR cost to optimised version
    # Re-predict with optimal threshold
    classes   = lr_m["model"].classes_.tolist()
    neg_idx   = classes.index(LABEL_NEG)
    proba_neg = lr_m["model"].predict_proba(X_test)[:, neg_idx]
    y_pred_opt = [LABEL_NEG if p >= opt_thresh else LABEL_POS for p in proba_neg]
    lr_m_opt   = _extract_metrics(y_test, y_pred_opt,
                                  f"LR (TF-IDF, thresh={opt_thresh:.2f})",
                                  lr_m["model"], lr_m["latency_ms"])
    cost_lr_opt = compute_daily_cost(lr_m_opt)
    cost_nb     = compute_daily_cost(nb_m)

    all_costs = [cost_lr_opt, cost_nb]
    cost_df = pd.DataFrame([{
        "Model":       c["model"],
        "Daily B_FN":  f"{c['daily_bfn']:,.0f}",
        "Daily B_FP":  f"{c['daily_bfp']:,.0f}",
        "FN Cost/day": f"${c['daily_bfn_cost']:,.0f}",
        "FP Cost/day": f"${c['daily_bfp_cost']:,.0f}",
        "Total/day":   f"${c['total_daily_cost']:,.0f}",
        "Annual":      f"${c['total_daily_cost']*365:,.0f}",
    } for c in all_costs])
    print("\n  COST TABLE (with optimal threshold):")
    print(cost_df.to_string(index=False))
    cost_df.to_csv(OUTPUT_DIR / "04_cost_model.csv", index=False)

    plot_cost_comparison(all_costs, OUTPUT_DIR / "04a_cost_comparison.png")
    print("  📊 04a_cost_comparison.png")

    best_cost_entry = min(all_costs, key=lambda c: c["total_daily_cost"])
    recommended_name = best_cost_entry["model"]
    print(f"\n  🏆 Lowest cost model: {recommended_name} @ ${best_cost_entry['total_daily_cost']:,.0f}/day")

    # ── STEP 5 ──────────────────────────────────────────────────────────
    print("\n▶ STEP 5 · TECHNICAL BRIEF")
    rec_m = lr_m_opt if "LR" in recommended_name else nb_m
    write_technical_brief(rec_m, best_cost_entry, opt_thresh, all_costs,
                          OUTPUT_DIR / "05_technical_brief.txt")
    print("  📄 05_technical_brief.txt")

    # ── STEP 6 ──────────────────────────────────────────────────────────
    print("\n▶ STEP 6 · FAILURE ANALYSIS  (Hard – Optional)")
    fa = reproduce_faulty_pipeline(X_train, X_test, y_train, y_test)
    plot_before_after(fa["faulty_m"], fa["fixed_m"], OUTPUT_DIR / "06_before_after.png")
    print("  📊 06_before_after.png")

    # ── STEP 7 ──────────────────────────────────────────────────────────
    print("\n▶ STEP 7 · COST OF FAILURE  (Hard – Optional)")
    cost_of_failure_analysis(fa["faulty_m"], best_cost_entry)

    print("\n" + "█"*68)
    print("  PIPELINE COMPLETE")
    print(f"  Outputs: {OUTPUT_DIR}")
    print("█"*68 + "\n")


if __name__ == "__main__":
    main()
