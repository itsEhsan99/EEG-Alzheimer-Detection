# EEG-based Alzheimer's Detection

An end-to-end tool for detecting Alzheimer's disease from resting-state EEG,
built on the ds004504 dataset. Work in progress.

## Approach
Recordings are band-pass filtered to the alpha band (8–13 Hz), split into
5-second epochs, and each epoch is converted to a 19×19 Phase Locking Value
(PLV) connectivity matrix. A small CNN classifies these matrices, evaluated
with Leave-One-Subject-Out (LOSO) cross-validation.

## Status
- [x] Project setup
- [x] Data loading
- [x] Preprocessing (bandpass, epoching, normalization)
- [x] PLV feature extraction
- [x] Baseline model & LOSO evaluation
- [ ] Application
- [ ] GNN model (planned)

## Results

### Baseline — alpha-band PLV + 2-layer CNN (LOSO)
Small subset: 10 AD + 10 HC subjects, 15 training epochs per fold.

| Metric | Value |
|--------|-------|
| Pooled accuracy | 0.647 |
| Pooled F1 | 0.689 |
| Pooled AUC | 0.662 |
| Mean per-subject accuracy | 0.646 (±0.342) |

The high per-subject variance reflects the core cross-subject
generalization challenge in EEG dementia detection: with few training
subjects the model captures individual connectivity signatures more than a
general disease pattern. This is an honest baseline to improve on, not a
deployable result.

## Known limitations / future work
- Epoching does not yet respect `boundary` annotations from the
  preprocessed (derivatives) recordings, so some windows may straddle data
  discontinuities.
- Subject selection uses simple filename sorting; extend and verify when
  scaling to all subjects.