# ConvLSTM Time Series Forecasting for Satellite Imagery

This project implements a TensorFlow-based ConvLSTM model for forecasting satellite spectral bands using spatiotemporal patterns from weather data and Sentinel-2 imagery.

## Problem Statement

Forecast spectral bands for 100 days (20 Sentinel-2 observations) using:
- **Context period**: First 10 images (50 days of data)
  - 10 Sentinel-2 observations (4 spectral bands: B02, B03, B04, B8A)
  - 10 weather observations (7 variables: humidity, precipitation, pressure, specific humidity, temperature min/avg/max)
- **Static features**: DEM (3 bands), Land Use/Land Cover, Geomorphology

**Output**: Predicted spectral band deltas for 20 future time steps (100 days)

## Dataset Structure

Data is stored as `.nc` (NetCDF) files with the following structure:

| Variable | Dimensions | Description |
|----------|------------|-------------|
| `s2` | (30, 4, 128, 128) | Sentinel-2 reflectance (B02, B03, B04, B8A) |
| `cloudmask` | (30, 128, 128) | Binary cloud mask (0=clear, 1=cloudy) |
| `weather` | (150,) | Daily weather variables (7 channels) |
| `dem` | (3, 128, 128) | Digital Elevation Models |
| `lulc` | (128, 128) | Land Use/Land Cover classes |
| `geomorph` | (128, 128) | Geomorphology classes |
| `doy` | (30,) | Day of year for each S2 observation |

### Input/Output Split

```
Context (t=0..9):     [S2(10,4,H,W) + Weather(10,7) + DEM(3,H,W) + CloudMask]
                              │
                              ▼
                        ConvLSTM Model
                              │
                              ▼
Future (t=10..29):    [ΔS2(20,4,H,W)] → Add baseline → [S2(20,4,H,W)]
```

## Implementation Plan

### Phase 1: Data Pipeline for TensorFlow

#### 1.1 Create TensorFlow Dataset Loader (`models/data_pipeline.py`)

```python
# Convert PyTorch dataset to TensorFlow tf.data pipeline
# Key transformations:
# - Load .nc files using xarray
# - Coarsen weather data (150 daily → 30 5-daily)
# - Compute cloud-free baseline from context period
# - Impute cloudy pixels with baseline values
# - Normalize using tanh normalization
# - Create context and future tensors
```

**Input features for the model:**
- `context`: (10, C_in, 128, 128) - Spatiotemporal context
  - S2 bands (4) + cloud mask (1) + DEM (3) + NDVI proxy (1) = 9 channels
- `weather_context`: (10, 7) - Weather time series for context
- `weather_future`: (20, 7) - Weather forecasts for prediction period
- `future`: (20, C_dec, 128, 128) - Decoder inputs (baseline + DEM)
- `doy_context`: (10,) - Day of year embeddings
- `doy_future`: (20,) - Day of year for forecast steps

**Targets:**
- `target_delta`: (20, 4, 128, 128) - Spectral band deltas from anchor
- `target_ndvi`: (20, 128, 128) - Raw NDVI for evaluation

#### 1.2 Data Preprocessing Considerations

- **Cloud handling**: Use cloud-free baseline to impute cloudy pixels in context
- **Normalization**: Tanh normalization with dataset statistics
- **Augmentation**: Random D4 transformations (rotations + flips) on spatial dimensions
- **Caching**: Pre-compute samples to `.pt` files for faster loading

---

### Phase 2: Model Architecture (`models/convlstm_model.py`)

#### 2.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      INPUT ENCODING                              │
├─────────────────────────────────────────────────────────────────┤
│  Context Encoder (Conv2D + ConvLSTM)                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│  │ Conv2D(64)  │───▶│ Conv2D(128) │───▶│ ConvLSTM    │          │
│  │ 3x3,ReLU    │    │ 3x3,ReLU    │    │ 128 filters │          │
│  └─────────────┘    └─────────────┘    └─────────────┘          │
│         ▲                                      │                 │
│         │ (10, 9, H, W)                        ▼                 │
│         │                              Hidden State              │
│         │                         (B, 128, H/4, W/4)             │
├─────────────────────────────────────────────────────────────────┤
│  Weather Encoder (MLP + FiLM)                                    │
│  ┌─────────────┐    ┌─────────────┐                             │
│  │ Dense(64)   │───▶│ Dense(256)  │───▶ γ, β for FiLM           │
│  │ (10, 7)     │    │ (per step)  │                             │
│  └─────────────┘    └─────────────┘                             │
├─────────────────────────────────────────────────────────────────┤
│                    SPATIOTEMPORAL FUSION                         │
│  ConvLSTM hidden state modulated by weather via FiLM layers     │
├─────────────────────────────────────────────────────────────────┤
│                   DECODER (ConvLSTM)                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│  │ ConvLSTM    │───▶│ ConvLSTM    │───▶│ Conv2D      │          │
│  │ 128 filters │    │ 64 filters  │    │ 4 channels  │          │
│  └─────────────┘    └─────────────┘    └─────────────┘          │
│       ▲                  │                    │                  │
│       │ (20 steps)       ▼                    ▼                  │
│       │            Future Weather        ΔS2(20,4,H,W)           │
│       │            Conditioning                                   │
└─────────────────────────────────────────────────────────────────┘
```

#### 2.2 Key Components

**A. Context Encoder**
```python
# Process spatiotemporal context (10 time steps)
# Uses Conv2D for spatial feature extraction at each time step
# ConvLSTM captures temporal dependencies
```

**B. Weather Integration (FiLM)**
```python
# Feature-wise Linear Modulation
# Weather MLP outputs scale (γ) and shift (β) parameters
# Applied to ConvLSTM hidden states: h' = γ * h + β
```

**C. ConvLSTM Core**
```python
# Convolutional LSTM for spatiotemporal processing
# Preserves spatial structure while modeling temporal dynamics
# Multiple stacked layers for hierarchical feature learning
```

**D. Decoder**
```python
# Autoregressive or sequence-to-sequence decoding
# Takes last encoder state as initial decoder state
# Conditions on future weather forecasts
# Outputs delta predictions for each future time step
```

#### 2.3 Alternative: Pure Conv2D + Temporal Attention

For comparison, implement a Conv2D baseline with temporal attention:
- Encode each time step independently with shared Conv2D weights
- Use temporal self-attention to aggregate context
- Decode with transposed convolutions

---

### Phase 3: Training Pipeline (`train.py`)

#### 3.1 Loss Functions

Implement multi-component loss:

```python
# Primary: Delta MSE on vegetation pixels
delta_loss = masked_delta_mse(pred_delta, target_delta, cloud_mask, veg_mask)

# Auxiliary: NDVI loss (derived from predicted bands)
pred_ndvi = compute_ndvi(pred_s2)
ndvi_loss = masked_ndvi_mse(pred_ndvi, target_ndvi, cloud_mask, veg_mask)

# Total loss
loss = delta_loss + λ * ndvi_loss
```

**Metrics:**
- **Vegetation Score**: NNSE-based metric for vegetation pixels
- **Per-pixel NSE**: Nash-Sutcliffe Efficiency
- **MSE/MAE**: Standard regression metrics

#### 3.2 Training Configuration

```yaml
optimizer: AdamW
learning_rate: 1e-4 (with cosine decay)
batch_size: 4-8 (depending on GPU memory)
epochs: 100
gradient_clipping: 1.0
mixed_precision: fp16 (for faster training)
```

#### 3.3 Curriculum Learning Strategy

1. **Stage 1** (epochs 1-30): Train on delta prediction only
2. **Stage 2** (epochs 31-60): Add NDVI auxiliary loss
3. **Stage 3** (epochs 61-100): Fine-tune with horizon weighting (weight later time steps higher)

---

### Phase 4: Evaluation & Inference (`evaluate.py`, `inference.py`)

#### 4.1 Evaluation Metrics

- **GreenEarthNet Vegetation Score**: Primary benchmark metric
- **Temporal consistency**: Smoothness of predictions across time
- **Spatial coherence**: Preservation of spatial patterns

#### 4.2 Inference Pipeline

```python
def predict(model, tile_path, stats_path):
    # 1. Load and preprocess data
    sample = load_sample(tile_path, stats_path)
    
    # 2. Run forward pass
    pred_delta = model(sample['context'], sample['weather_context'], 
                       sample['weather_future'], sample['future'])
    
    # 3. Reconstruct absolute values
    pred_s2 = sample['s2_anchor'] + pred_delta
    
    # 4. Compute derived products (NDVI, etc.)
    pred_ndvi = compute_ndvi(pred_s2)
    
    return pred_s2, pred_ndvi
```

---

## Project Structure

```
/workspace/
├── README.md                 # This file
├── dataset/
│   ├── loader.py            # NetCDF loading utilities
│   ├── dataset.py           # PyTorch Dataset implementation
│   └── cached.py            # Cached dataset for faster loading
├── models/
│   ├── __init__.py
│   ├── data_pipeline.py     # TensorFlow data loading (TO CREATE)
│   ├── convlstm_model.py    # ConvLSTM architecture (TO CREATE)
│   └── unet_baseline.py     # Conv2D baseline (OPTIONAL)
├── train.py                  # Training script (TO CREATE)
├── evaluate.py               # Evaluation script (TO CREATE)
├── inference.py              # Inference utilities (TO CREATE)
├── metrics.py                # Evaluation metrics (existing)
└── requirements.txt          # Dependencies (TO CREATE)
```

---

## Dependencies

```txt
tensorflow>=2.15.0
tensorflow-addons>=0.23.0  # For ConvLSTM
xarray>=2023.0.0
netcdf4>=1.6.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
```

---

## Usage

### Training

```bash
python train.py \
    --train_dir greenearthnet/train \
    --val_dir greenearthnet/val \
    --stats_path stats.json \
    --output_dir checkpoints/ \
    --epochs 100 \
    --batch_size 4
```

### Evaluation

```bash
python evaluate.py \
    --model_path checkpoints/best_model.h5 \
    --test_dir greenearthnet/test \
    --stats_path stats.json
```

### Inference

```bash
python inference.py \
    --model_path checkpoints/best_model.h5 \
    --tile_path path/to/tile.nc \
    --stats_path stats.json \
    --output_path predictions/
```

---

## Key Design Decisions

1. **Delta Prediction**: Model predicts changes from anchor frame rather than absolute values
   - More stable training
   - Better handles pixel value distributions
   - Baseline provides cloud-free reference

2. **ConvLSTM over 3D Conv**: ConvLSTM better captures long-term temporal dependencies
   - Memory cell preserves information across many time steps
   - Gating mechanisms control information flow

3. **Weather via FiLM**: External modulation rather than concatenation
   - Weather affects feature processing, not just adds channels
   - Allows different weather impacts at different network depths

4. **Vegetation-focused Loss**: Weight vegetation pixels higher
   - Aligns with evaluation metric
   - Natural vegetation shows stronger weather-spectrum relationships

---

## Next Steps

1. [ ] Implement `models/data_pipeline.py` - TensorFlow dataset loader
2. [ ] Implement `models/convlstm_model.py` - ConvLSTM architecture
3. [ ] Implement `train.py` - Training loop with mixed precision
4. [ ] Implement `evaluate.py` - Evaluation with Vegetation Score
5. [ ] Run ablation studies (ConvLSTM vs Conv2D, FiLM vs concatenation)
6. [ ] Hyperparameter tuning and final training

---

## References

- Shi, X., et al. (2015). "Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting." NeurIPS.
- Perez, E., et al. (2018). "FiLM: Visual Reasoning with a General Conditioning Layer." AAAI.
- GreenEarthNet Challenge Documentation
- TensorFlow ConvLSTM Implementation: https://www.tensorflow.org/addons/api_docs/python/tfa/seq2seq/ConvLSTM2D