# CryoET Particle Detection

Deep learning model for detecting and classifying particles in CryoET (Cryo-electron tomography) data.

## Features

- 3D Vision Transformer architecture with axial attention
- Feature pyramid network for multi-scale processing
- Squeeze-excitation blocks for channel attention
- Mixed precision training
- Multi-GPU support
- Test-time augmentation
- Ensemble predictions


## Usage

### Training

```python
from models import ImprovedViT3D
from training import train_model

# Initialize model
model = ImprovedViT3D(
    patch_size=(8, 16, 16),
    embed_dim=512,
    depth=12,
    num_heads=16
)

# Train
train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=15,
    save_dir='checkpoints'
)
```

### Inference

```python
from inference import predict_ensemble, generate_submission

predictions = predict_ensemble(
    models=models,
    zarr_path=zarr_path,
    patch_size=patch_size,
    stride_3d=stride_3d
)
```

## Model Architecture

- Improved Vision Transformer 3D with axial attention
- Feature Pyramid Network for multi-scale processing
- Residual blocks with squeeze-excitation
- Multiple prediction heads:
  - Classification (particle/no particle)
  - Coordinate regression
  - Particle type classification

## Dataset

Uses the CryoET3DDataset class for loading and processing 3D tomography data:

- Supports ZARR format
- On-the-fly patch extraction
- Data augmentation
- Balanced sampling

## License

MIT
