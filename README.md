# GPU Decision Engine (GDE)

Learned, cost-aware GPU scheduling decision model (candidate scoring), validated in simulation on public GPU-cluster traces. Spec: `PLAN.MD`. Agent rules: `AGENTS.md`.

## Setup

Requires Python 3.11 or 3.12. No GNU make; all tasks go through `tasks.py`.

```text
python tasks.py setup     # create .venv, install pinned requirements.txt + gde (editable)
python tasks.py hwcheck   # write docs/env.md (GPU / torch check)
python tasks.py data      # download traces per their READMEs -> data/raw/, docs/data_dictionary.md
python tasks.py eda       # per-dataset summary stats -> reports/eda/p1_summary.md
python tasks.py test      # CI-style check: venv version, pip check, pytest
python tasks.py help      # list all targets
```
