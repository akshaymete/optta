# OpTTA on DINO-WM

Built on `../temporal_straightening/code`, with faster planning switched on in the configs. The paper evaluates
DINO-WM on PushT with the predictor adapted; its DINOv2 encoder stays frozen. Keep `third_party/` next to `code/`.

## Install

Use the environment from `../temporal_straightening`. The DINOv2 weights download through `torch.hub` on the first
run, into `$TORCH_HOME`.

## Checkpoints and data

- **Checkpoint:** the PushT checkpoint from the [DINO-WM release](https://github.com/gaoyuezhou/dino_wm).
- **Evaluation episodes:** `pushobj_eval/val_T/plan_targets.pkl` from the
  [AdaJEPA data](https://drive.google.com/drive/folders/11IVDIrVDU6W47txR_ku1RkhRUj9F4ybs?usp=sharing).

The checkpoint's config refers to `$DATASET_DIR`, so set it. DINO-WM's training data is not needed.

## Run

```bash
export DATASET_DIR=/path/to/data
export WANDB_MODE=disabled
export MUJOCO_GL=egl
cd code

# OpTTA, PushT darkened to 0.1, seed 1
python plan.py --config-name optta_plan_cem_pushobj.yaml \
    ckpt_base_path=/path/to/checkpoints/pusht \
    eval_data_path=$DATASET_DIR/pushobj_eval/val_T/plan_targets.pkl \
    ood_corruption=dark ood_level=0.1 seed=1 \
    planner.adapt.optimism_ratio_target=1 planner.adapt.finetune_encoder=false \
    hydra.run.dir=/path/to/outputs/dark0.1_optta_s1 +wandb_logging=false
```

Each run writes `summary.json` to `hydra.run.dir`; give each run its own directory, or runs overwrite each other.

| Method | Arguments |
|---|---|
| Frozen | `planner.adapt.lr=0 planner.adapt.steps=0` |
| AdaJEPA | `planner.adapt.finetune_encoder=false` |
| OpTTA | `planner.adapt.optimism_ratio_target=1 planner.adapt.finetune_encoder=false` |

The configs evaluate CEM's costs and the planning term in bfloat16, and the planning term in chunks of 50 candidates.
Leave `OPTTA_ATTENTION` at its default `auto`: `sdpa` and `auto_cudnn` are not deterministic.
