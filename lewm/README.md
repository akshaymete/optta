# OpTTA on LeWM

Adds test-time adaptation to the released LeWM checkpoints, using
[stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) for environments, data and checkpoint loading.

## Install

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130
```

Keep `transformers` at 4.x: under version 5 the release checkpoints fail to load.

## Checkpoints and data

Download the checkpoints and datasets from the
[LeWM Hugging Face collection](https://huggingface.co/collections/quentinll/lewm) and lay them out as:

```
$STABLEWM_HOME/
├── checkpoints/{pusht,reacher,tworoom}/lewm/{weights.pt,config.json}
└── datasets/
    ├── pusht_expert_train.h5
    ├── tworoom.h5
    └── dmc/reacher_random.h5
```

## Run

```bash
export STABLEWM_HOME=/path/to/stablewm_home
export OPTTA_ROOT=/path/to/outputs
export PYTHONPATH=/path/to/optta/lewm/code:$PYTHONPATH
export MUJOCO_GL=egl

# OpTTA, predictor only, PushT blurred with sigma 4, seed 1
python -m lewm_optta.eval task=pusht arm=optta seed=1 \
    corruption.name=blur corruption.level=4
```

Settings live in `code/lewm_optta/config.yaml` and can be overridden as `key=value`; unknown keys are rejected.

| Setting | Override |
|---|---|
| Task | `task=pusht`, `task=reacher` or `task=tworoom` |
| Method | `arm=frozen`, `arm=adajepa` or `arm=optta` |
| Corruption | `corruption.name=<name> corruption.level=<level>`; none for clean episodes |
| Salt-and-pepper | also `corruption.goal_mode=episode` (one noise pattern on the goal image per episode) |
| Predictor and encoder | `adapt.finetune_encoder=true` |

Each run writes `summary.json` with the success rate and the cumulative success at each replan to
`$OPTTA_ROOT/results/<run name>/`. The run name comes from the task, corruption, method, encoder setting and seed
(`blur4_optta_s1` above); for other changes add `tag=<suffix>` or set `run_name=<name>`.
