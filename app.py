"""Streamlit demo for the Quanta multimodal brain-age held-out validation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DATA = Path(__file__).parent / "data"

st.set_page_config(
    page_title="Brain-age held-out validation",
    page_icon="🧠",
    layout="wide",
)


@st.cache_data
def load_predictions() -> pd.DataFrame:
    npz = np.load(DATA / "holdout_predictions_clean.npz", allow_pickle=True)
    df = pd.DataFrame(
        {
            "true_age": npz["y"],
            "cv_oof": npz["cv_oof"],
            "test_pred": npz["test_pred"],
            "is_test": npz["is_test"],
            "fold": npz["fold_ids"],
            "sex": npz["sex"],
        }
    )
    df["split"] = np.where(df["is_test"], "Held-out test (n=86)", "Train CV OOF (n=341)")
    df["pred"] = np.where(df["is_test"], df["test_pred"], df["cv_oof"])
    df["abs_error"] = (df["pred"] - df["true_age"]).abs()
    return df


@st.cache_data
def load_summary() -> dict:
    return json.loads((DATA / "holdout_clean_summary.json").read_text())


@st.cache_data
def load_ablation() -> pd.DataFrame:
    return pd.read_csv(DATA / "base_learner_block_ablation.tsv", sep="\t")


preds = load_predictions()
summary = load_summary()
ablation = load_ablation()
S = summary["summary"]

st.title("Quanta multimodal brain-age — held-out validation")
st.caption(
    "Independent 80/20 frozen split (n_train=341, n_test=86). Stacked Ridge meta-learner "
    "over six modality blocks (BEH · EEG · MRI · COMP · DL · DTI). Click any tab to drill in."
)

# ── headline metrics ─────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Train-CV MAE", f"{S['cv_mae']:.3f} yr", f"± {S['cv_mae_std']:.3f}")
c2.metric("Held-out MAE", f"{S['test_mae']:.3f} yr", f"Δ {S['test_mae']-S['cv_mae']:+.3f}")
c3.metric("Held-out R²", f"{S['test_r2']:.3f}")
c4.metric("Held-out Pearson r", f"{S['test_pearson']:.3f}")

tabs = st.tabs(["Scatter", "Fold metrics", "Modality ablation", "About"])

# ── scatter ──────────────────────────────────────────────────────────────────
with tabs[0]:
    st.subheader("Predicted vs chronological age")
    lo = float(min(preds["true_age"].min(), preds["pred"].min())) - 2
    hi = float(max(preds["true_age"].max(), preds["pred"].max())) + 2
    fig = px.scatter(
        preds,
        x="true_age",
        y="pred",
        color="split",
        symbol="split",
        hover_data={
            "fold": True,
            "abs_error": ":.2f",
            "is_test": False,
            "split": False,
            "true_age": ":.1f",
            "pred": ":.1f",
        },
        labels={"true_age": "Chronological age (years)", "pred": "Predicted age (years)"},
        color_discrete_map={
            "Train CV OOF (n=341)": "#7aa6c2",
            "Held-out test (n=86)": "#d2766b",
        },
        opacity=0.75,
    )
    fig.add_trace(
        go.Scatter(
            x=[lo, hi], y=[lo, hi], mode="lines",
            line=dict(color="#888", dash="dash"), name="identity", showlegend=False,
        )
    )
    fig.update_layout(
        height=540,
        xaxis=dict(range=[lo, hi], scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[lo, hi]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, width="stretch")

    sub1, sub2 = st.columns(2)
    with sub1:
        st.markdown("**Train CV OOF (n=341)**")
        st.write(
            {
                "MAE": f"{preds.loc[~preds.is_test, 'abs_error'].mean():.3f} yr",
                "Pearson r": f"{preds.loc[~preds.is_test].corr(numeric_only=True).loc['true_age','pred']:.3f}",
            }
        )
    with sub2:
        st.markdown("**Held-out test (n=86)**")
        st.write(
            {
                "MAE": f"{preds.loc[preds.is_test, 'abs_error'].mean():.3f} yr",
                "Pearson r": f"{preds.loc[preds.is_test].corr(numeric_only=True).loc['true_age','pred']:.3f}",
            }
        )

# ── fold metrics ─────────────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("Cross-validation fold metrics (5-fold inner CV on the 341 train subjects)")
    fm = pd.DataFrame(summary["fold_metrics"])
    fm.insert(0, "fold", [f"fold {i}" for i in range(len(fm))])
    fm_disp = fm.copy()
    fm_disp["mae"] = fm_disp["mae"].map(lambda v: f"{v:.3f}")
    fm_disp["rmse"] = fm_disp["rmse"].map(lambda v: f"{v:.3f}")
    fm_disp["r2"] = fm_disp["r2"].map(lambda v: f"{v:.3f}")
    fm_disp["pearson"] = fm_disp["pearson"].map(lambda v: f"{v:.3f}")
    st.dataframe(fm_disp, hide_index=True, width="stretch")

    fig = px.bar(
        fm, x="fold", y="mae",
        labels={"mae": "MAE (years)", "fold": ""},
        color_discrete_sequence=["#7aa6c2"],
    )
    fig.add_hline(
        y=S["cv_mae"], line_dash="dash", line_color="#444",
        annotation_text=f"CV mean = {S['cv_mae']:.3f}", annotation_position="top right",
    )
    fig.add_hline(
        y=S["test_mae"], line_dash="dot", line_color="#d2766b",
        annotation_text=f"Held-out = {S['test_mae']:.3f}", annotation_position="bottom right",
    )
    fig.update_layout(height=360)
    st.plotly_chart(fig, width="stretch")

# ── modality ablation ────────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("Modality block ablation on the held-out test set")
    st.caption(
        "Δ test MAE relative to the full 6-modality model. Positive bars mean removing "
        "that block hurts the test fit (block was informative); negative bars mean the "
        "model is slightly better without it on the 86 held-out subjects."
    )
    drop = ablation[ablation["mode"] == "drop_block"].copy()
    drop["block"] = drop["label"].str.replace("drop__", "", regex=False)
    drop = drop.sort_values("delta_test_mae_vs_full", ascending=True)
    fig = px.bar(
        drop, x="delta_test_mae_vs_full", y="block", orientation="h",
        text=drop["delta_test_mae_vs_full"].map(lambda v: f"{v:+.3f}"),
        labels={"delta_test_mae_vs_full": "Δ Test MAE vs full (years)", "block": ""},
        color="delta_test_mae_vs_full",
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=420, coloraxis_showscale=False)
    st.plotly_chart(fig, width="stretch")

    alone = ablation[ablation["mode"] == "block_alone"].copy()
    alone["block"] = alone["label"].str.replace("alone__", "", regex=False)
    alone = alone.sort_values("test_mae", ascending=True)
    st.markdown("**Each block alone** — MAE when only that block's base learners feed the meta:")
    fig2 = px.bar(
        alone, x="block", y="test_mae",
        text=alone["test_mae"].map(lambda v: f"{v:.2f}"),
        labels={"test_mae": "Held-out MAE (years)", "block": ""},
        color_discrete_sequence=["#7aa6c2"],
    )
    fig2.add_hline(
        y=S["test_mae"], line_dash="dash", line_color="#d2766b",
        annotation_text=f"Full stack = {S['test_mae']:.3f}", annotation_position="top right",
    )
    fig2.update_traces(textposition="outside")
    fig2.update_layout(height=380)
    st.plotly_chart(fig2, width="stretch")

    with st.expander("Full ablation table"):
        st.dataframe(ablation, hide_index=True, width="stretch")

# ── about ────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.subheader("About this demo")
    st.markdown(
        """
        **Cohort.** 427 healthy adults (age 19–80) from the TCNL Quanta multimodal
        ageing study. Phenotype includes behavioural batteries, resting EEG, T1-w
        MRI (cortical/sub-cortical morphometry), DTI (JHU-ICBM ROIs + voxel-level),
        and resting-state fMRI Schaefer-400 functional connectivity.

        **Model.** Six modality blocks × five base learners (Ridge, KRR-RBF,
        LGBM-L2, SVR-RBF, LGBM-quantile) feeding a RidgeCV meta-learner with a
        residual Ridge correction and an OOF-selected percentile clip on the
        meta-features.

        **Held-out split.** 80/20 age × sex stratified split frozen ahead of the
        final run: 341 training subjects + 86 held-out subjects. Auxiliary scalar
        predictors (DTI ROI / DTI voxel ridge / rs-fMRI FC ridge) were regenerated
        under the frozen split — train rows received inner-fold OOF predictions
        and held-out rows received predictions from a model fit only on the 341
        train subjects. Missing values were imputed from train-only statistics.

        **What this demo is not.** No raw subject features ship with the repo.
        Only de-identified position-indexed prediction vectors, the run summary
        JSON, and the modality ablation TSV are included.
        """
    )
    st.markdown(
        "**Source repository:** internal `Quanta_heldout` "
        "(commit `bea3ee6`). "
        "**Distinctness audit:** see `logs/heldout_distinctness_audit.md` in that repo."
    )
