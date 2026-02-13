# Analysis Scripts (ICLR)

This folder contains the three core analysis scripts used for the ICLR paper:

- Paper link: https://arxiv.org/abs/2509.22391

## Scripts

1. `grounded_reason.py`
- Purpose: grounded reasoning quality and search behavior analysis (RQI/SEI style figures).
- Input: `--input` (default: `final_annotated_traces.jsonl`)
- Output dir: `--outdir` (default: `figs_cohort`)
- Main outputs: cohort grounding/search figures and `reporting_checklist.txt`.

2. `recovery.py`
- Purpose: low-quality evidence recovery dynamics.
- Input: positional `input_file` (default: `final_annotated_traces.jsonl`)
- Output dir: positional `output_dir` (default: `recovery_figs`)
- Main outputs: recovery plots and CSVs (`R1_turn_level_records.csv`, `R1_A3_recovery_trajectory_stats.csv`).

3. `calibration.py`
- Purpose: evidence-strength calibration and answer-timing analysis.
- Input: `--input` (default: `final_annotated_traces.jsonl`)
- Output dir: `--outdir` (default: `figs_evidence`)
- Main outputs: `fig_R3_AB_combined.png`, `fig_R3_D_calibration_rates.png`, `fig_R3_E_timing.png`.

## Quick Run

```bash
python grounded_reason.py --input final_annotated_traces.jsonl --outdir figs_cohort
python recovery.py final_annotated_traces.jsonl recovery_figs
python calibration.py --input final_annotated_traces.jsonl --outdir figs_evidence
```

