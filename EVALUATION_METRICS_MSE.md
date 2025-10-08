# Evaluation Metrics: Mean Squared Error (MSE)

## Overview

The training pipeline now uses **Mean Squared Error (MSE)** as the primary evaluation metric, computed using **PyTorch** for consistency with the rest of the codebase.

## Why MSE?

Your model outputs **continuous similarity scores** between 0 and 1 (scaled cosine similarity), and labels are also 0 or 1. This is effectively a **regression problem**, where MSE directly measures how close predictions are to the true labels.

## Metrics Computed

All metrics are computed using PyTorch (`torch.nn.functional`):

### 1. MSE (Mean Squared Error)
```python
mse = F.mse_loss(predictions, labels)
```
- **Range**: 0 to 1 (in your case)
- **Lower is better**
- Measures average squared difference between predictions and labels
- **Interpretation**:
  - MSE < 0.05: Excellent (predictions very close to labels)
  - MSE < 0.10: Good
  - MSE > 0.10: Needs improvement

### 2. RMSE (Root Mean Squared Error)
```python
rmse = torch.sqrt(mse)
```
- **Range**: 0 to 1 (in your case)
- **Lower is better**
- Same units as the original predictions (similarity scores)
- More interpretable than MSE
- **Interpretation**: Average prediction error in similarity units

### 3. MAE (Mean Absolute Error)
```python
mae = F.l1_loss(predictions, labels)
```
- **Range**: 0 to 1 (in your case)
- **Lower is better**
- Average absolute difference (less sensitive to outliers than MSE)
- **Interpretation**: Average magnitude of prediction errors

### 4. R² Score (Coefficient of Determination)
```python
r2 = 1 - (sum_squared_residuals / total_sum_of_squares)
```
- **Range**: -∞ to 1 (typically 0 to 1 for good models)
- **Higher is better**
- Measures how well the model explains variance in the data
- **Interpretation**:
  - R² > 0.9: Excellent (explains >90% of variance)
  - R² > 0.7: Good
  - R² > 0.5: Moderate
  - R² < 0.5: Poor

## Usage

### Training with MSE Evaluation

```bash
# Basic training - automatically evaluates with MSE
python src/quantized_finetuning_v2.py \
    --run_name my_experiment \
    --dataset_path datasets/training_set_v2.json \
    --num_of_epochs 5
```

### Output Example

```
==================================================
Training Summary:
==================================================
Final training loss: 0.0834
Final validation loss: 0.0912

Validation Metrics (PyTorch-based):
  MSE:  0.0456
  RMSE: 0.2135
  MAE:  0.1823
  R² Score: 0.8734

Overfitting Analysis:
  ✅ Model appears to generalize well!
     Validation loss (0.0912) is close to training loss (0.0834)
  ✓ Good MSE: 0.0456
  ✅ Excellent R²: 0.8734 (model explains variance well)
==================================================
```

## Overfitting Detection

The pipeline checks for overfitting using multiple indicators:

### 1. Loss Ratio
Compares training loss vs validation loss:
- **eval_loss / train_loss < 1.1**: ✅ Good generalization
- **1.1 < ratio < 1.2**: ✓ Reasonable
- **ratio > 1.2**: ⚠️ Potential overfitting

### 2. MSE Magnitude
Checks if predictions are accurate:
- **MSE < 0.05**: ✅ Excellent
- **MSE < 0.10**: ✓ Good
- **MSE > 0.10**: ⚠️ Needs improvement

### 3. R² Score
Checks if model explains variance:
- **R² > 0.9**: ✅ Excellent
- **R² > 0.7**: ✓ Good
- **R² > 0.5**: ⚠️ Moderate
- **R² < 0.5**: ⚠️ Poor

## Comparing Models

Use the updated `compare_models.py` script:

```bash
# Compare all trained models
python compare_models.py

# Export to CSV for analysis
python compare_models.py --export_csv results.csv
```

### Comparison Output

```
============================================================
MODEL COMPARISON
============================================================
checkpoint      train_loss  eval_loss  eval_mse  eval_rmse  eval_mae  eval_r2_score  overfitting_ratio
baseline_model  0.0834      0.0912     0.0456    0.2135     0.1823    0.8734         1.093
high_lr_model   0.0723      0.0998     0.0523    0.2287     0.1956    0.8421         1.380
small_model     0.0912      0.0945     0.0478    0.2186     0.1867    0.8612         1.036
============================================================

🏆 Best MSE: baseline_model
   MSE: 0.0456

🏆 Best R² Score: baseline_model
   R²: 0.8734

🏆 Best Eval Loss: baseline_model
   Loss: 0.0912

📊 Best Generalization (lowest overfitting ratio):
   ✅ small_model: 1.036
   ✅ baseline_model: 1.093
   ⚠️ high_lr_model: 1.380
```

## Best Model Selection

The pipeline automatically saves the best model based on **eval_mse** (lowest MSE on validation set).

Configuration in `TrainingArguments`:
```python
metric_for_best_model="eval_mse",
greater_is_better=False,  # Lower MSE is better
```

## TensorBoard Visualization

View all metrics in real-time:

```bash
tensorboard --logdir=peft_lab_outputs/logs
```

TensorBoard shows:
- Training loss curves
- Validation loss curves
- MSE, RMSE, MAE, R² over time
- Comparison across different runs

## Technical Details

### Why PyTorch instead of NumPy/sklearn?

1. **Consistency**: Rest of the codebase uses PyTorch
2. **GPU acceleration**: Metrics computed on GPU if available
3. **Automatic differentiation**: Can be extended for custom training
4. **Native integration**: Works seamlessly with HuggingFace Trainer

### Metric Computation Flow

1. **During training**: Model outputs cosine similarity scores (0-1 range)
2. **During evaluation**: 
   - Predictions collected from validation set
   - Converted to PyTorch tensors
   - Metrics computed using `torch.nn.functional`
   - Results logged to TensorBoard and saved as JSON

### Saved Files

Each training run saves:
- `train_results.json` - Training metrics and final train loss
- `eval_results.json` - Evaluation metrics (MSE, RMSE, MAE, R²)
- `trainer_state.json` - Complete history of all metrics over time

## Example: Interpreting Results

### Scenario 1: Good Model
```
Training loss: 0.0523
Eval loss: 0.0567
MSE: 0.0234
RMSE: 0.1530
R²: 0.9123
```
✅ **Interpretation**: Excellent model
- Losses are close (good generalization)
- Low MSE means accurate predictions
- High R² means model explains variance well

### Scenario 2: Overfitting
```
Training loss: 0.0123
Eval loss: 0.0823
MSE: 0.0467
RMSE: 0.2161
R²: 0.6234
```
⚠️ **Interpretation**: Model is overfitting
- Eval loss much higher than train loss
- Moderate MSE
- Lower R² suggests poor generalization
- **Solution**: Reduce epochs, add regularization, or get more data

### Scenario 3: Underfitting
```
Training loss: 0.1523
Eval loss: 0.1567
MSE: 0.1234
RMSE: 0.3513
R²: 0.4567
```
⚠️ **Interpretation**: Model is underfitting
- Both losses are high
- High MSE
- Low R² means model doesn't explain variance
- **Solution**: Train longer, increase model capacity, or improve data quality

## Migration from Classification Metrics

### Old Metrics (Removed)
- Accuracy
- Precision
- Recall
- F1 Score

### Why Changed?
These metrics treated the problem as binary classification with a threshold (0.5), but:
1. Your model outputs **continuous** similarity scores
2. Labels are 0 or 1, but the task is better framed as **regression**
3. MSE directly optimizes what BCE loss is training for
4. MSE is more interpretable for similarity scoring

### If You Need Classification Metrics
You can add them back to `compute_metrics()`:

```python
# In quantized_finetuning_v2.py
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions_tensor = torch.tensor(predictions, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    
    # MSE metrics (current)
    mse = F.mse_loss(predictions_tensor, labels_tensor)
    # ... other metrics ...
    
    # Optional: Add classification metrics
    binary_preds = (predictions > 0.5).astype(int)
    binary_labels = labels.astype(int)
    accuracy = (binary_preds == binary_labels).mean()
    
    return {
        "mse": mse.item(),
        "rmse": rmse.item(),
        "mae": mae.item(),
        "r2_score": r2_score.item(),
        "accuracy": accuracy,  # Optional
    }
```

## Related Files

- `src/quantized_finetuning_v2.py` - Main training script (updated)
- `compare_models.py` - Model comparison tool (updated)
- `EVALUATION_GUIDE.md` - General evaluation guide
- `peft_lab_outputs/` - Saved models and metrics

## Quick Reference

| Metric | Range | Best Value | PyTorch Function | Purpose |
|--------|-------|------------|------------------|---------|
| MSE | 0-1 | 0 | `F.mse_loss()` | Primary metric, squared error |
| RMSE | 0-1 | 0 | `torch.sqrt(mse)` | Same units as predictions |
| MAE | 0-1 | 0 | `F.l1_loss()` | Absolute error, robust to outliers |
| R² | -∞ to 1 | 1 | `1 - (SS_res/SS_tot)` | Variance explained |
| BCE Loss | 0-∞ | 0 | `F.binary_cross_entropy()` | Training objective |

---

**Note**: All metrics are computed using PyTorch for consistency with your training pipeline.

