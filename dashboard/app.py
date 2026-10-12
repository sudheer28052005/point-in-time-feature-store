"""Customer Retention Engine - Streamlit dashboard.   Run:  streamlit run dashboard/app.py

All figures are read from SQL Server (read-only) or from files written by the evaluation scripts.
The data is SYNTHETIC; business values are illustrative assumptions, not measured outcomes.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(os.path.abspath(__file__)).parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard import data as D  # noqa: E402

st.set_page_config(page_title="Customer Retention Engine", page_icon="📊", layout="wide")

# cached, parameterised read-only queries (10 minutes)
cache = st.cache_data(ttl=600, show_spinner=False)
bounds, kpis, monthly, channels = cache(D.data_bounds), cache(D.kpis), cache(D.monthly_activity), cache(D.customers_by_channel)
cohort_matrix, cohort_long, snapshots, rfm = cache(D.cohort_matrix), cache(D.cohort_long), cache(D.snapshots), cache(D.rfm_distribution)
model_versions, risk_dist, top_risk, deciles = cache(D.model_versions), cache(D.risk_distribution), cache(D.top_risk_customers), cache(D.decile_outcomes)
model_metrics, bench, bench_env = cache(D.model_metrics), cache(D.benchmark_table), cache(D.benchmark_environment)

st.title("Customer Retention Engine")
st.warning("**Synthetic data.** Every customer, order, web event and outcome here is simulated. Revenue and campaign values are "
           "illustrative; model metrics come from `reports/metrics_*.json` written by `scripts/evaluate_model.py`.", icon="⚠️")

try:
    b = bounds()
except Exception as exc:  # noqa: BLE001 - show an actionable message instead of a stack trace
    st.error(f"Cannot read the database: {exc}\n\nCheck `.env` (SQL_SERVER_HOST, trust-certificate setting) and that "
             "`scripts/run_pipeline.py` has been run.")
    st.stop()

with st.sidebar:
    st.header("Filters")
    period = st.date_input("Period (orders, revenue, activity)", value=(b["min"], b["max"]), min_value=b["min"], max_value=b["max"])
    d_from, d_to = (period if isinstance(period, (list, tuple)) and len(period) == 2 else (b["min"], b["max"]))
    st.caption(f"Data covers {b['min']} to {b['max']} (UTC).")
    activity_def = st.radio("Retention activity definition", ["purchase", "web"], help="purchase = a qualifying order in the month (primary); "
                            "web = product view / cart / checkout / purchase event (secondary)")
    snaps = snapshots()
    snapshot = st.selectbox("Feature snapshot (RFM)", snaps, index=len(snaps) - 2 if len(snaps) > 1 else 0)
    versions = model_versions()
    version = st.selectbox("Model version", versions) if versions else None
    top_n = st.slider("Top high-risk customers", 10, 100, 25, step=5)

tab_over, tab_ret, tab_seg, tab_model, tab_perf = st.tabs(["Overview", "Retention cohorts", "Segments & churn risk", "Model quality", "SQL performance"])

# --------------------------------------------------------------------------- overview
with tab_over:
    k = kpis(d_from, d_to)
    c = st.columns(4)
    c[0].metric("Registered customers (to period end)", f"{k['customers']:,}")
    c[1].metric("Qualifying orders in period", f"{k['orders']:,}")
    c[2].metric("Net revenue in period", f"${k['revenue'] / 1e6:,.2f}M" if k["revenue"] >= 1e6 else f"${k['revenue']:,.0f}",
                help=f"Exact: \\${k['revenue']:,.2f}")
    c[3].metric("Average order value", "n/a" if np.isnan(k["aov"]) else f"${k['aov']:,.2f}")
    st.caption("Qualifying = completed orders, plus refunded orders only while the refund had not yet happened at the period end. "
               "Net = gross - discount.")
    m = monthly(d_from, d_to)
    if m.empty:
        st.info("No orders in this period.")
    else:
        m = m.set_index("month")
        left, right = st.columns(2)
        left.subheader("Monthly net revenue")
        left.line_chart(m[["revenue"]])
        right.subheader("Monthly orders")
        right.line_chart(m[["orders"]])
        left.subheader("Customers active (bought) per month vs new sign-ups")
        left.line_chart(m[["active_customers", "new_customers"]])
        right.subheader("Customers by acquisition channel")
        ch = channels().set_index("acquisition_channel")
        right.bar_chart(ch)
        st.caption("Partial first/last months in the selected period have fewer days and look lower.")

# --------------------------------------------------------------------------- retention
with tab_ret:
    st.subheader(f"Monthly cohort retention - activity = {activity_def}")
    mat = cohort_matrix(activity_def)
    if mat.empty:
        st.info("No cohort data: run `scripts/build_features.py`.")
    else:
        fig, ax = plt.subplots(figsize=(11, 5.5))
        data = np.ma.masked_invalid(mat.to_numpy(float))
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("#eeeeee")                       # unobserved future months: grey, NOT zero
        im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=0)
        ax.set_xlabel("months since signup (0 = signup month)")
        ax.set_yticks(range(0, len(mat), 3))
        ax.set_yticklabels([d.strftime("%Y-%m") for d in mat.index[::3]])
        ax.set_ylabel("signup cohort")
        fig.colorbar(im, ax=ax, label="% of cohort active")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Grey = month not yet observed (future or partial). Dark/purple = observed with little or no activity - a true low, "
                   "not missing data. Month 0 is a partial month for most customers.")
        long = cohort_long(activity_def)
        avg = long[long.is_observed.astype(bool)].groupby("months_since_acquisition").apply(
            lambda g: 100 * g.active_customers.sum() / g.cohort_size.sum(), include_groups=False)
        st.subheader("Pooled retention curve")
        st.line_chart(avg.rename("% of customers active"))
        with st.expander("Retention matrix (percent)"):
            st.dataframe(mat.round(1), use_container_width=True)

# --------------------------------------------------------------------------- segments and risk
with tab_seg:
    left, right = st.columns(2)
    left.subheader(f"RFM segments - snapshot {snapshot}")
    r = rfm(snapshot)
    left.bar_chart(r.set_index("rfm_segment")[["customers"]], horizontal=True)
    left.dataframe(r.round({"avg_lifetime_spend": 2, "avg_orders": 2, "share_pct": 1}), use_container_width=True, hide_index=True)
    left.caption("Quintiles are computed over customers who have purchased as of the snapshot; 'No Purchase' customers have no scores.")
    right.subheader("Predicted churn risk")
    if version is None:
        right.info("No predictions yet: run `scripts/evaluate_model.py`.")
    else:
        rd = risk_dist(version, "scoring")
        if rd.empty:
            right.info("No scoring-snapshot predictions for this model version.")
        else:
            right.caption(f"Model `{version}`; latest snapshot (features only; the outcome is not observable yet). "
                          "Population = customers who bought in the previous 90 days.")
            # zero-padded labels: Streamlit sorts categories as text, so "5%" would otherwise land after "45%"
            right.bar_chart(rd.assign(band=lambda d: [f"{int(round(b * 100)):02d}-{int(round(b * 100)) + 5:02d}%" for b in d.band_start])
                            .set_index("band")[["customers"]])
            right.caption("x-axis: predicted churn probability band; y-axis: customers.")
            tot, flagged = int(rd.customers.sum()), int(rd.flagged.sum())
            right.metric("Customers flagged for outreach (validation-chosen threshold)", f"{flagged:,} of {tot:,}", f"{100 * flagged / tot:.1f}%", delta_color="off")
    if version is not None:
        st.subheader(f"Top {top_n} highest-risk customers (synthetic ids)")
        st.dataframe(top_risk(version, top_n).round({"churn_probability": 3, "days_since_last_purchase": 1, "spend_90d": 2}),
                     use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------- model
with tab_model:
    mm = model_metrics("active_90d")
    if mm is None:
        st.info("No metrics file: run `scripts/train_model.py` then `scripts/evaluate_model.py`.")
    else:
        champ = mm["champion"]
        st.markdown(f"**Champion: `{champ}`**, selected on validation PR-AUC only. Evaluated once on the untouched test partition: "
                    f"{mm['test_rows']:,} customer-snapshots, {len(mm['test_snapshots'])} monthly cutoffs "
                    f"({mm['test_snapshots'][0]} to {mm['test_snapshots'][-1]}), churn rate {mm['test_churn_rate']:.1%}.")
        rows = []
        for name, v in mm["models"].items():
            r_ = v["ranking"]
            rows.append({"model": name, "ROC-AUC": r_["roc_auc"], "PR-AUC": r_["pr_auc"], "recall @ top 10%": r_["recall_at_top10pct"],
                         "precision @ top 10%": r_["precision_at_top10pct"], "lift @ top 10%": r_["lift_at_top10pct"]})
        st.dataframe(pd.DataFrame(rows).sort_values("PR-AUC", ascending=False).round(4), use_container_width=True, hide_index=True)
        st.caption(f"Prevalence {mm['test_churn_rate']:.3f}: a model with no skill has PR-AUC equal to it. Recall @ top 10% is capped at "
                   f"{0.1 / mm['test_churn_rate']:.1%} when 10% of customers are inspected and {mm['test_churn_rate']:.0%} churn - read it with precision and lift.")
        ch = mm["models"][champ]
        t = ch["at_threshold"]
        c = st.columns(5)
        c[0].metric("Threshold", f"{t['threshold']:.3f}", help="Chosen on validation data to maximise illustrative campaign value; applied to the raw predicted probability.")
        c[1].metric("Precision", f"{t['precision']:.3f}")
        c[2].metric("Recall", f"{t['recall']:.3f}")
        c[3].metric("F1", f"{t['f1']:.3f}")
        c[4].metric("Flagged share", f"{t['flagged_share']:.1%}")
        cm = t["confusion_matrix"]
        st.dataframe(pd.DataFrame([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]], index=["actual: stays", "actual: churns"],
                                  columns=["predicted: stays", "predicted: churns"]), use_container_width=True)
        biz = mm["business"]
        # "\$" because Streamlit markdown treats a pair of bare dollar signs as LaTeX
        st.info(f"Illustrative campaign value on test at the validation-chosen threshold: **\\${biz['test_campaign_value_at_threshold']:,.0f}** vs "
                f"\\${biz['test_value_contact_everyone']:,.0f} contacting everyone. Assumptions (not measured): {biz['assumptions']}.")
        figs = ROOT / "reports" / "figures" / "active_90d"
        pairs = [("roc_pr.png", "ROC and precision-recall"), ("calibration.png", "Calibration"), ("deciles.png", "Churn rate by risk decile"),
                 ("feature_importance.png", "Permutation importance"), ("stability.png", "Stability across test months")]
        cols = st.columns(2)
        for i, (fn, cap) in enumerate(pairs):
            if (figs / fn).exists():
                cols[i % 2].image(str(figs / fn), caption=cap)
        if version is not None:
            dd = deciles(version)
            if not dd.empty:
                st.subheader("Observed churn rate by predicted-risk decile (test, from SQL)")
                st.bar_chart(dd.set_index("risk_decile")[["observed_churn_rate", "mean_predicted"]])

# --------------------------------------------------------------------------- performance
with tab_perf:
    bt = bench("warm")
    if bt.empty:
        st.info("No benchmark results: run `scripts/run_benchmarks.py`.")
    else:
        env = bench_env() or {}
        st.caption(f"{env.get('sql_server_edition', '')} {env.get('sql_server_version', '')}, {env.get('logical_cpus', '?')} logical CPUs, "
                   f"{env.get('physical_memory_mb', '?')} MB RAM. Logical reads are exact; elapsed times carry laptop noise (see reports/performance_report.md).")
        names = {"A": "A baseline", "B": "B user-time clustered", "C": "C + time index", "D": "D + columnstore", "E": "E + batch mode", "F": "F deployed"}
        metric = st.radio("Metric", ["logical_reads", "elapsed_ms"], horizontal=True)
        piv = bt.pivot_table(index="benchmark_id", columns="design_letter", values=metric)
        piv = piv[[c for c in names if c in piv.columns]]
        fig, ax = plt.subplots(figsize=(11, 0.45 * len(piv) + 1.5))
        h = 0.8 / len(piv.columns)
        for j, col in enumerate(piv.columns):
            ax.barh(np.arange(len(piv)) + j * h, piv[col].fillna(0), h, label=names[col])
        ax.set_yticks(np.arange(len(piv)) + 0.4 - h / 2)
        ax.set_yticklabels(piv.index, fontsize=8)
        ax.set_xscale("log")
        ax.invert_yaxis()
        ax.set_xlabel(("median logical reads (pages)" if metric == "logical_reads" else "median warm elapsed ms") + ", log scale - lower is better")
        ax.legend(fontsize=7, ncol=3, loc="lower right")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.dataframe(piv.round(1), use_container_width=True)
        st.caption("A = surrogate-key clustered PK; B = clustered by (user, time); C = B + covering time indexes; D = C + columnstore; "
                   "E = B + batch-mode enabler; F = deployed layout. Not every design was run for every query.")
