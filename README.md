# MCMMGNN

This folder contains the clean runnable version of the MCMMGNN model for the Uniswap PyG `HeteroData` snapshot dataset.

## Files

- `model.py`: MCMMGNN model implementation.
- `train_model.py`: training script with validation, checkpoint saving, and test-after-train.
- `test_model.py`: standalone checkpoint evaluation script.

## Default Setup

The default training configuration is:

```text
dataset_path = pull_data/uniswap_pyg_dataset_precise_masks/uniswap_hetero_snapshots_h6.pt
output_dir = MCMMGNN/checkpoints/TRSM
epochs = 70
time_window = 6
hidden_dim = 64
state_steps = 1
state_grad_steps = 0
encoder_dropout = 0.1
transition_dropout = 0.2
readout_dropout = 0.3
head_dropout = 0.5
lr = 0.001
transition_lr_mult = 4.0
weight_decay = 0.0001
grad_clip = 5.0
position_loss_weight = 0.5
batch_size = 8
device = cuda:3
seed = 50
```

## Train

Run with default parameters:

```powershell
python MCMMGNN/train_model.py
```

Run with an explicit device:

```powershell
python MCMMGNN/train_model.py --device cuda:3
```

The script saves:

```text
MCMMGNN/checkpoints/TRSM/best.pt
MCMMGNN/checkpoints/TRSM/last.pt
MCMMGNN/checkpoints/TRSM/history.json
MCMMGNN/checkpoints/TRSM/test_metrics.json
```

## Test

Evaluate the default best checkpoint:

```powershell
python MCMMGNN/test_model.py
```

Evaluate a specific checkpoint:

```powershell
python MCMMGNN/test_model.py --checkpoint MCMMGNN/checkpoints/TRSM/best.pt --device cuda:3
```

## Notes

- The dataset is expected to be a saved payload containing `snapshots` and `splits`.
- Each snapshot should be a PyTorch Geometric `HeteroData` object.
- The model predicts both pool labels and position labels; loss weighting is controlled by `--position-loss-weight`.
- Training automatically evaluates on the test split after loading the best validation checkpoint.
