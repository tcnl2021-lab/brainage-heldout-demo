# Brain-age held-out demo

Streamlit app that visualises the held-out validation of the Quanta multimodal
brain-age stacking model.

- **Cohort:** 427 healthy adults (TCNL Quanta multimodal ageing study)
- **Split:** 341 train / 86 held-out, age × sex stratified, frozen
- **Result:** train-CV MAE = 2.848 yr  ·  held-out MAE = 3.045 yr  ·  R² = 0.960

The app reads three artifacts in `data/` produced by the internal
`Quanta_heldout` pipeline (no raw subject features ship in this repo).

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud

1. Push this repo to GitHub (suggested name: `tcnl2021-lab/brainage-heldout-demo`).
2. At <https://share.streamlit.io>, **New app** → pick the repo, branch `main`,
   entry file `app.py`. Free tier is sufficient.
3. The artifacts in `data/` are committed — no secrets, no env vars, no DB.

## Refreshing the artifacts

The three files in `data/` are produced by the internal pipeline. To regenerate
them, in the `Quanta_heldout` working repo:

```bash
micromamba run -n base python codes/brain_age_2026/tuning/holdout_clean/train_holdout_clean.py
micromamba run -n base python codes/brain_age_2026/tuning/holdout_clean/base_learner_block_ablation.py
```

Then copy these three files into `data/` here and commit:

- `holdout_predictions_clean.npz`
- `holdout_clean_summary.json`
- `base_learner_block_ablation.tsv`

## Provenance

- Source repo: internal `Quanta_heldout`, commit `bea3ee6`
  ("Add independent held-out brain-age modelling repo")
- Distinctness audit: `logs/heldout_distinctness_audit.md` in that repo verifies
  the held-out 86 subjects are fully disjoint from the 341 training subjects
  and that no leaky precomputed OOF inputs reach the test rows.
