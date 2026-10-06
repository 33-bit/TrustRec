# Reports and Figures

Put metric tables in `reports/tables/`. Put figures in `reports/figures/`. Git ignores generated files in these paths. Each report artifact must have an experiment manifest and a source citation.

Before you add a chart, name its snapshot, candidate protocol, metric, seed summary, and uncertainty method.

The T5.3 report bundle uses `configs/report_bundle.json` and
`scripts/package_report.py`. Write the generated bundle to
`reports/generated/t5-3/`. Run the saved bundle verifier before you publish the
report. Keep full metrics and figures outside Git.
