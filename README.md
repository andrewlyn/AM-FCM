
## Purpose

The package addresses the PLOS ONE revision request for a minimal data set: values behind reported averages and error metrics, source values used to make figures and tables, related metadata, method parameters, and author-generated analysis code.

## Data conventions

- Synthetic sample indices are zero-based in algorithm result fields unless a column name ends in `_1` or `_1based`.
- Field manual picks are provided both as one-based labels (`reference_arrival_1based`) and zero-based array indices (`reference_index_0based`).
- Sampling rate is 1000 Hz and the sample interval is 1 ms for the packaged synthetic and field records.
- `error_ms` and `abs_error_ms` are absolute errors; `signed_error_ms` preserves the direction of the error.
- CSV files are UTF-8 with a byte-order mark for compatibility with Excel and common statistical software.

## Directory map

- `01_Synthetic_Data`: metadata, arrival references, frequency/SNR performance, feature correlation and combination results, cluster comparison, HOS ablation, and robustness summaries.
- `02_Field_Data`: 816 field waveforms, manual picks, trace-level AM-FCM and baseline results, and field method comparison.
- `03_Figure_Source_Data`: flat CSV source tables corresponding to Figures 2-13.
- `04_Table_Source_Data`: flat CSV source tables corresponding to Tables 1-6 in the working revision outputs.
- `05_Parameters`: parsed settings and the deep-learning experiment configuration.
- `06_Code`: author-generated scripts, figure notebooks, requirements, and requested convenience filenames.
- `07_Supporting_Information`: supplementary parameter workbook.

## Reproduction entry points

The authoritative scripts are under `06_Code/source`. Their expected input/output locations are documented in `program_file_descriptions.txt` in the original workspace and in the source-file docstrings. The figure notebooks are under `06_Code/notebooks`.

## Items that still require completion before public deposition

The workspace now contains replicate-level raw output for the noise-robustness experiment (Figure 8) and waveform-robustness experiment (Figure 9); these files have been placed in `01_Synthetic_Data` and `03_Figure_Source_Data`. The HOS ablation folder still contains only its macro-average summary, so paired raw records should be added if the manuscript reports replicate-level uncertainty from that experiment. The workspace also did not contain field PhaseNet/EQTransformer result CSVs; the two files in `02_Field_Data` are status records, not completed results. Figure 2's CWT example arrays should still be exported from the plotting notebook before submission.

Do not upload the package as a final supporting-information archive until these explicitly marked gaps are resolved or explained in the Data Availability Statement.
