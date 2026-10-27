# OpTTA on temporal straightening

Plans with the released temporal-straightening world models, using CEM or a gradient-based planner. Keep
`third_party/` next to `code/`.

## Install

```bash
cd code
conda env create -f environment.yaml
conda activate ts
```

PointMaze also needs MuJoCo 2.1 and OSMesa rendering for mujoco-py; PushT and PushObj do not.

## Checkpoints and data

Download the [checkpoints and evaluation data](https://drive.google.com/drive/folders/11IVDIrVDU6W47txR_ku1RkhRUj9F4ybs?usp=sharing)
released with AdaJEPA, and the medium-maze data
[here](https://drive.google.com/drive/folders/1qbPO9MK7LwX2GBQq-fP_xXoy82wYlkde?usp=sharing). The paper uses
`pusht_visual_shift` (PushT corruptions and recolorings), `pushobj_shape_shift` (held-out shapes) and
`mediummaze_dynamics_shift` (maze dynamics).

## Run

```bash
export DATASET_DIR=/path/to/data
export WANDB_MODE=disabled
export MUJOCO_GL=egl
cd code

# OpTTA, encoder and predictor adapted, PushT darkened to 0.7, seed 1
python plan.py --config-name optta_plan_cem_pushobj.yaml \
    ckpt_base_path=/path/to/checkpoints/pusht_visual_shift \
    eval_data_path=$DATASET_DIR/pushobj_eval/val_T/plan_targets.pkl \
    ood_corruption=dark ood_level=0.7 seed=1 \
    planner.adapt.optimism_ratio_target=1 \
    hydra.run.dir=/path/to/outputs/dark0.7_optta_s1 +wandb_logging=false
```

| Method | Arguments |
|---|---|
| Frozen | `planner.adapt.lr=0 planner.adapt.steps=0` |
| AdaJEPA | none |
| OpTTA | `planner.adapt.optimism_ratio_target=1` |
| Predictor only | add `planner.adapt.finetune_encoder=false` |

- **Maze:** use `optta_plan_cem_maze.yaml` with `eval_data_path=$DATASET_DIR/point_maze_medium`, set
  `MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa`, and pass corruptions as `+ood_corruption=... +ood_level=...`.
- **Gradient-based planner:** use `optta_plan_gd_pushobj.yaml` with the same arguments (20 replans per episode).

Each run writes `summary.json` with the success rate and the cumulative success at each replan to
`hydra.run.dir`; give each run its own directory, or runs overwrite each other.

Environment shifts:

| Shift | Checkpoint | Arguments |
|---|---|---|
| Red agent | `pusht_visual_shift` | `…/val_T/plan_targets.pkl '++env_kwargs_override.agent_color=Red'` |
| Red anchor | `pusht_visual_shift` | `…/val_T/plan_targets.pkl '++env_kwargs_override.goal_color=Red'` |
| Red block | `pusht_visual_shift` | `…/val_T/plan_targets.pkl '++env_kwargs_override.color=Red'` |
| Shape I | `pushobj_shape_shift` | `…/val_I/plan_targets.pkl` |
| Small tee | `pushobj_shape_shift` | `…/val_small_tee/plan_targets.pkl` |
| Square | `pushobj_shape_shift` | `…/val_square/plan_targets.pkl` |
| Low mass | `mediummaze_dynamics_shift` | maze config, `'++env_kwargs_override.density_scale=0.2'` |
| High damping | `mediummaze_dynamics_shift` | maze config, `'++env_kwargs_override.damping_scale=20'` |
