# Paper reproduction

Prepare the price data first:

```bash
python scripts/reproduce/prepare_finsaber.py
```

The runner reads the exact public model IDs and experiment constants from
`configs/paper/`. Commands can be inspected without loading a model:

```bash
python scripts/reproduce/run_paper.py benchmarks --model phi4 --dry-run
python scripts/reproduce/run_paper.py backtest-is --model phi4 --dry-run
python scripts/reproduce/run_paper.py ranking --model phi4 --dry-run
```

Remove `--dry-run` to execute. Run the `ranking` stage once for each of the
eleven model keys, then aggregate the 330 seven-model subsets:

```bash
python scripts/reproduce/summarize_alignment.py
```

Outputs are written under `results/reproduction/` and are ignored by Git.
HumanEval executes model-generated Python and must only be run in an isolated
environment.
