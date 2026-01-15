# Chronos-2 Time Series Forecasting Documentation

## Overview
Chronos-2 is Amazon's latest foundation model for universal time series forecasting. It supports univariate, multivariate, and covariate-informed forecasting in a zero-shot manner. Built on the T5 architecture, Chronos-2 delivers state-of-the-art performance across multiple benchmarks.

**Model**: amazon/chronos-2  
**Repository**: https://github.com/amazon-science/chronos-forecasting  
**Paper**: https://arxiv.org/abs/2510.15821  
**Hugging Face**: https://huggingface.co/amazon/chronos-2

## Key Features
- **Zero-shot forecasting**: No training required for new datasets
- **Multivariate support**: Jointly predict multiple time series
- **Covariate-informed**: Incorporate past-only, future, and categorical covariates
- **Probabilistic forecasting**: Generate quantile predictions with uncertainty estimates
- **High efficiency**: 300+ forecasts/second on A10G GPU
- **Flexible**: Supports both GPU and CPU inference

## Model Variants

### Chronos Family Overview
- **Chronos-2** (Latest): Universal forecasting with covariates
- **Chronos-Bolt**: Fast variant (up to 250x faster)
- **Original Chronos**: Language model-based approach

### Chronos-2 Models
- `amazon/chronos-2` - Full model (recommended)
- `amazon/chronos-2-small` - Smaller variant for faster inference

## Installation

### Basic Installation
```bash
pip install "chronos-forecasting>=2.0"
```

### With Extra Dependencies
```bash
pip install "chronos-forecasting>=2.2[extras]"
```

### Additional Requirements
```bash
pip install 'pandas[pyarrow]'  # For DataFrame API
pip install torch              # PyTorch backend
```

## Quick Start

### Basic Univariate Forecasting

```python
import pandas as pd
import torch
from chronos import Chronos2Pipeline

# Load the model (GPU recommended, CPU also supported)
pipeline = Chronos2Pipeline.from_pretrained(
    "amazon/chronos-2",
    device_map="cuda",  # or "cpu" for CPU inference
    torch_dtype=torch.bfloat16
)

# Load your time series data
df = pd.read_csv("your_data.csv")

# Generate predictions
pred_df = pipeline.predict_df(
    df,
    prediction_length=24,
    quantile_levels=[0.1, 0.5, 0.9]
)
```

### Using Tensor API (Low-Level)

```python
import torch
from chronos import BaseChronosPipeline

pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-2",
    device_map="cuda"
)

# Context must be a 1D tensor or list of 1D tensors
context = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

# Get quantile predictions
quantiles, mean = pipeline.predict_quantiles(
    context=context,
    prediction_length=12,
    quantile_levels=[0.1, 0.5, 0.9]
)

# quantiles shape: (batch_size, prediction_length, num_quantiles)
# mean shape: (batch_size, prediction_length)
```

## DataFrame API (Recommended)

### Input Format

The DataFrame should have:
- **target column(s)**: Time series values to predict
- **timestamp column**: DateTime information (optional but recommended)
- **id column**: Identifier for different time series (for multiple series)
- **covariate columns**: Additional features (past-only or future covariates)

```python
# Example DataFrame structure
# | timestamp   | id    | target | past_cov1 | future_cov1 |
# |-------------|-------|--------|-----------|-------------|
# | 2024-01-01  | ts_1  | 100    | 50        | 20          |
# | 2024-01-02  | ts_1  | 105    | 52        | 22          |
```

### Basic predict_df Usage

```python
# Univariate forecasting
pred_df = pipeline.predict_df(
    context_df,                      # Historical data
    prediction_length=24,            # Number of steps to forecast
    quantile_levels=[0.1, 0.5, 0.9], # Quantiles for probabilistic forecast
    id_column="id",                  # Column identifying time series
    timestamp_column="timestamp",    # Column with datetime info
    target="target"                  # Column(s) to predict
)
```

### With Covariates

```python
# Load historical data with past covariates
context_df = pd.read_parquet("train.parquet")

# Load future covariates (known future values)
test_df = pd.read_parquet("test.parquet")
future_df = test_df.drop(columns="target")  # Remove target, keep covariates

# Generate predictions with covariates
pred_df = pipeline.predict_df(
    context_df,
    future_df=future_df,             # Future covariate values
    prediction_length=24,
    quantile_levels=[0.1, 0.5, 0.9],
    id_column="id",
    timestamp_column="timestamp",
    target="target"
)
```

### Multivariate Forecasting

```python
# Predict multiple targets jointly
pred_df = pipeline.predict_df(
    context_df,
    prediction_length=24,
    quantile_levels=[0.1, 0.5, 0.9],
    id_column="id",
    timestamp_column="timestamp",
    target=["cpu_usage", "memory_usage", "io_usage"]  # Multiple targets
)
```

## Output Format

### DataFrame Output

The output DataFrame contains:
- **timestamp**: Future timestamps
- **id**: Time series identifier
- **target_name**: Name of the predicted target
- **mean**: Point forecast (mean prediction)
- **quantile_X**: Quantile predictions (e.g., `0.1`, `0.5`, `0.9`)

```python
# Example output structure
# | timestamp   | id    | target_name | mean  | 0.1   | 0.5   | 0.9   |
# |-------------|-------|-------------|-------|-------|-------|-------|
# | 2024-02-01  | ts_1  | target      | 102.5 | 98.0  | 102.0 | 107.0 |
```

### Tensor Output

When using `predict_quantiles`:
```python
quantiles, mean = pipeline.predict_quantiles(...)

# quantiles shape: (batch_size, prediction_length, num_quantiles)
# For batch_size=1, prediction_length=12, 3 quantiles:
# quantiles[0, :, 0] -> 0.1 quantile predictions
# quantiles[0, :, 1] -> 0.5 quantile (median)
# quantiles[0, :, 2] -> 0.9 quantile predictions

# mean shape: (batch_size, prediction_length)
```

## Working with NumPy Arrays

### Converting from NumPy to Tensor

```python
import numpy as np
import torch

# Your NumPy data
observations = np.array([...])  # shape: (timesteps, features)
actions = np.array([...])

# Convert to torch tensor
obs_tensor = torch.from_numpy(observations).float()

# For single time series
context = obs_tensor  # 1D or 2D tensor

# Generate predictions
quantiles, mean = pipeline.predict_quantiles(
    context=context,
    prediction_length=10,
    quantile_levels=[0.1, 0.5, 0.9]
)

# Convert back to NumPy if needed
predictions_np = quantiles.cpu().numpy()
```

### Batch Processing Multiple Series

```python
# List of 1D tensors (different length series)
contexts = [
    torch.tensor(series1),
    torch.tensor(series2),
    torch.tensor(series3)
]

# Or left-padded 2D tensor (same length required)
contexts = torch.stack([
    torch.nn.functional.pad(s, (max_len - len(s), 0))
    for s in series_list
])

quantiles, mean = pipeline.predict_quantiles(
    context=contexts,
    prediction_length=10,
    quantile_levels=[0.1, 0.5, 0.9]
)
```

## Covariate Support

Chronos-2 natively supports three types of covariates:

### 1. Past-Only Covariates
Historical features that are NOT known in the future (e.g., past traffic volume).

```python
# Include in context_df
context_df['past_traffic'] = [...]  # Historical values only
```

### 2. Future Covariates
Known future values (e.g., weather forecasts, promotional schedules).

```python
# Provide in future_df
future_df['scheduled_promotion'] = [...]  # Known future values
future_df['weather_forecast'] = [...]
```

### 3. Categorical Covariates
Discrete features (e.g., day of week, holiday indicator).

```python
# Include in context and/or future dataframes
context_df['day_of_week'] = [1, 2, 3, 4, 5, 6, 7, ...]
context_df['is_holiday'] = [0, 0, 1, 0, 0, ...]
```

## Advanced Usage

### Extracting Embeddings

```python
# Get embeddings from the encoder
context = torch.tensor(your_time_series)
embeddings, tokenizer_state = pipeline.embed(context)

# embeddings: High-dimensional representation from encoder
```

### Custom Quantile Levels

```python
# Specify any quantile levels you need
pred_df = pipeline.predict_df(
    context_df,
    prediction_length=24,
    quantile_levels=[0.05, 0.25, 0.5, 0.75, 0.95],  # Custom levels
    # ... other parameters
)
```

### Handling Long Context

```python
# Chronos-2 supports long context lengths
# For MBRL data with long trajectories
long_context = torch.tensor(trajectory_data)  # e.g., 1000+ timesteps

predictions = pipeline.predict_quantiles(
    context=long_context,
    prediction_length=50,  # Multi-step forecast
    quantile_levels=[0.1, 0.5, 0.9]
)
```

## Probabilistic Forecasting Metrics

### Computing CRPS (Continuous Ranked Probability Score)

```python
from scipy.stats import norm
import numpy as np

def compute_crps(quantiles, quantile_levels, actuals):
    """
    Compute CRPS from quantile predictions.
    
    Args:
        quantiles: (n_timesteps, n_quantiles) array
        quantile_levels: list of quantile levels
        actuals: (n_timesteps,) array of actual values
    """
    crps_values = []
    for t in range(len(actuals)):
        pred_quantiles = quantiles[t]
        actual = actuals[t]
        
        # Approximate CRPS using quantiles
        crps = 0
        for i, q_level in enumerate(quantile_levels):
            pred = pred_quantiles[i]
            crps += 2 * (q_level - (actual < pred)) * (pred - actual)
        
        crps_values.append(crps / len(quantile_levels))
    
    return np.mean(crps_values)
```

### Calibration Analysis

```python
def check_calibration(quantiles, quantile_levels, actuals):
    """
    Check if predicted quantiles are well-calibrated.
    
    Returns coverage for each quantile level.
    """
    coverage = {}
    for i, q_level in enumerate(quantile_levels):
        predictions = quantiles[:, i]
        below = (actuals < predictions).mean()
        coverage[q_level] = below
    
    return coverage

# Expected: coverage[0.1] ≈ 0.1, coverage[0.5] ≈ 0.5, etc.
```

## MBRL-Specific Use Case

### Dynamics Modeling with Actions as Covariates

```python
import pandas as pd
import numpy as np

def prepare_mbrl_data(observations, actions, lookback=20):
    """
    Prepare MBRL data for Chronos-2.
    
    Args:
        observations: (n_steps, obs_dim) array of observations
        actions: (n_steps, act_dim) array of actions
        lookback: context length for forecasting
    
    Returns:
        DataFrame suitable for Chronos-2
    """
    data_list = []
    
    # Create rolling windows
    for i in range(lookback, len(observations) - 1):
        # Context window
        ctx_obs = observations[i-lookback:i]
        ctx_actions = actions[i-lookback:i]
        
        # Target (next observation)
        target = observations[i]
        
        # Future action (known covariate)
        future_action = actions[i]
        
        # Create row for each dimension
        for dim in range(observations.shape[1]):
            row = {
                'timestamp': i,
                'id': f'obs_dim_{dim}',
                'target': target[dim],
            }
            
            # Add past actions as features
            for t in range(lookback):
                for act_dim in range(actions.shape[1]):
                    row[f'past_action_{act_dim}_t{t}'] = ctx_actions[t, act_dim]
            
            # Add future action as covariate
            for act_dim in range(actions.shape[1]):
                row[f'future_action_{act_dim}'] = future_action[act_dim]
            
            data_list.append(row)
    
    return pd.DataFrame(data_list)

# Usage
context_df = prepare_mbrl_data(observations, actions)

# Split into context and future
split_idx = int(len(context_df) * 0.8)
train_df = context_df[:split_idx]
test_df = context_df[split_idx:]

# Prepare future covariates
future_cols = [c for c in test_df.columns if c.startswith('future_')]
future_df = test_df[['timestamp', 'id'] + future_cols]

# Forecast
pred_df = pipeline.predict_df(
    train_df,
    future_df=future_df,
    prediction_length=10,
    quantile_levels=[0.1, 0.5, 0.9],
    id_column='id',
    timestamp_column='timestamp',
    target='target'
)
```

### Single-Step vs Multi-Step Forecasting

```python
# Single-step forecasting
single_step_pred = pipeline.predict_quantiles(
    context=observations_tensor,
    prediction_length=1,  # Only next step
    quantile_levels=[0.1, 0.5, 0.9]
)

# Multi-step forecasting
multi_step_pred = pipeline.predict_quantiles(
    context=observations_tensor,
    prediction_length=10,  # 10 steps ahead
    quantile_levels=[0.1, 0.5, 0.9]
)

# Compare MSE
single_step_mse = ((single_step_pred[1][0, 0] - actual[0]) ** 2).item()
multi_step_mse = ((multi_step_pred[1][0] - actual[:10]) ** 2).mean().item()
```

## Performance Tips

### 1. Use GPU When Possible
```python
# GPU is significantly faster
pipeline = Chronos2Pipeline.from_pretrained(
    "amazon/chronos-2",
    device_map="cuda"  # ~300 forecasts/second on A10G
)
```

### 2. Batch Processing
```python
# Process multiple time series at once
contexts = [series1, series2, series3]  # List of tensors
quantiles, mean = pipeline.predict_quantiles(
    context=contexts,
    prediction_length=10,
    quantile_levels=[0.1, 0.5, 0.9]
)
```

### 3. Use bfloat16 for Memory Efficiency
```python
pipeline = Chronos2Pipeline.from_pretrained(
    "amazon/chronos-2",
    device_map="cuda",
    torch_dtype=torch.bfloat16  # Reduces memory usage
)
```

## Model Comparison

### Chronos-2 vs Chronos-Bolt vs Original Chronos

| Feature | Chronos-2 | Chronos-Bolt | Original Chronos |
|---------|-----------|--------------|------------------|
| Multivariate | ✅ Yes | ❌ No | ❌ No |
| Past Covariates | ✅ Native | ⚠️ External | ⚠️ External |
| Future Covariates | ✅ Native | ❌ No | ❌ No |
| Speed | Fast | Fastest (250x) | Moderate |
| Accuracy | Best | Better | Good |
| Use Case | Universal | Speed-critical | Univariate |

## Common Issues and Solutions

### Issue: Out of Memory
```python
# Solution 1: Use CPU
pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")

# Solution 2: Use smaller model
pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2-small")

# Solution 3: Reduce batch size
# Process series one at a time instead of batching
```

### Issue: Slow Inference
```python
# Solution: Use Chronos-Bolt for speed
from chronos import ChronosBoltPipeline
pipeline = ChronosBoltPipeline.from_pretrained("amazon/chronos-bolt-base")
```

### Issue: Poor Uncertainty Estimates
```python
# Ensure you're using appropriate quantile levels
quantile_levels = [0.1, 0.25, 0.5, 0.75, 0.9]  # More quantiles for better uncertainty

# Check calibration on validation set
coverage = check_calibration(quantiles_val, quantile_levels, actuals_val)
print(coverage)  # Should match quantile levels approximately
```

## Production Deployment

### AWS SageMaker
```python
from sagemaker.jumpstart.model import JumpStartModel

# Deploy to SageMaker
model = JumpStartModel(
    model_id="pytorch-forecasting-chronos-2",
    instance_type="ml.g5.2xlarge"
)
predictor = model.deploy()

# Make predictions
payload = {
    "inputs": [{"target": your_series.tolist()}],
    "parameters": {"prediction_length": 24}
}
forecast = predictor.predict(payload)["predictions"]
```
