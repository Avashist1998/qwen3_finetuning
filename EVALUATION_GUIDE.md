# Model Evaluation and Overfitting Detection Guide

This guide explains how to use the evaluation features in the training pipeline to detect overfitting and compare model performance.

## Overview

The updated training pipeline (`src/quantized_finetuning_v2.py`) now includes:
- **Train/validation split** or separate validation dataset support
- **Evaluation metrics** computed during training (accuracy, precision, recall, F1, loss)
- **Automatic overfitting detection** by comparing train vs validation loss
- **Model comparison tool** to compare multiple trained models

## Key Features

### 1. Evaluation During Training

The pipeline automatically:
- Evaluates the model on the validation set after each epoch
- Computes multiple metrics (accuracy, precision, recall, F1, BCE loss)
- Saves the best model based on validation loss
- Logs all metrics to TensorBoard for visualization

### 2. Overfitting Detection

After training completes, the pipeline:
- Compares training loss vs validation loss
- Displays a warning if validation loss is significantly higher (>20% higher)
- Indicates good generalization if losses are similar

**Overfitting Indicators:**
- ⚠️  **High overfitting**: `eval_loss > 1.2 × train_loss`
- ✓  **Moderate fit**: `1.1 × train_loss < eval_loss < 1.2 × train_loss`
- ✅ **Good generalization**: `eval_loss < 1.1 × train_loss`

### 3. Metrics Saved

The pipeline saves metrics in JSON format:
- `train_results.json` - Training metrics
- `eval_results.json` - Evaluation metrics
- `trainer_state.json` - Complete training history

## Usage

### Basic Training with Evaluation

```bash
# Use automatic train/validation split (80/20 by default)
python src/quantized_finetuning_v2.py \
    --run_name my_experiment \
    --dataset_path datasets/training_set_v2.json \
    --num_of_epochs 5
```

### Custom Validation Split

```bash
# Use 70/30 train/validation split
python src/quantized_finetuning_v2.py \
    --run_name my_experiment \
    --dataset_path datasets/training_set_v2.json \
    --validation_split 0.3 \
    --num_of_epochs 5
```

### Separate Validation Dataset

```bash
# Load validation data from a different file
python src/quantized_finetuning_v2.py \
    --run_name my_experiment \
    --dataset_path datasets/training_set_v2.json \
    --eval_dataset_path datasets/evaluation_dataset.json \
    --num_of_epochs 5
```

### Quick Experimentation

```bash
# Limit dataset size for faster iteration
python src/quantized_finetuning_v2.py \
    --run_name quick_test \
    --dataset_path datasets/training_set_v2.json \
    --dataset_size 100 \
    --num_of_epochs 2
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--run_name` | str | timestamp | Name for this training run |
| `--model_id` | str | Qwen/Qwen3-Embedding-0.6B | Hugging Face model ID |
| `--dataset_path` | str | datasets/mini_dataset_v2.json | Path to training dataset |
| `--eval_dataset_path` | str | None | Path to separate evaluation dataset |
| `--validation_split` | float | 0.2 | Validation split ratio (0.0-1.0) |
| `--num_of_epochs` | int | 2 | Number of training epochs |
| `--dataset_size` | int | None | Limit dataset size (for testing) |

## Comparing Multiple Models

Use the `compare_models.py` script to compare different training runs:

### Compare All Models

```bash
# Automatically find and compare all models in peft_lab_outputs/
python compare_models.py
```

### Compare Specific Checkpoints

```bash
# Compare specific checkpoint directories
python compare_models.py \
    --checkpoints peft_lab_outputs/run1 peft_lab_outputs/run2 peft_lab_outputs/run3
```

### Export Comparison to CSV

```bash
# Export comparison table to CSV for further analysis
python compare_models.py --export_csv model_comparison.csv
```

### Comparison Metrics

The comparison tool shows:
- **Train Loss**: Final training loss
- **Eval Loss**: Validation loss
- **Accuracy**: Binary classification accuracy (threshold=0.5)
- **Precision**: Positive class precision
- **Recall**: Positive class recall
- **F1 Score**: Harmonic mean of precision and recall
- **Overfitting Ratio**: `eval_loss / train_loss` (closer to 1.0 is better)

## Monitoring Training with TensorBoard

View real-time training progress:

```bash
# Start TensorBoard
tensorboard --logdir=peft_lab_outputs/logs

# Open browser to http://localhost:6006
```

TensorBoard displays:
- Training and validation loss curves
- All evaluation metrics over time
- Easy comparison between different runs

## Understanding the Metrics

### Classification Metrics

Your task is **binary classification** - predicting whether two sentences are similar (1) or different (0).

- **Accuracy**: Percentage of correct predictions
  - Good baseline metric
  - Can be misleading with imbalanced datasets

- **Precision**: Of all predicted similar pairs, what % are actually similar?
  - High precision = few false positives
  - Important if you want to avoid false matches

- **Recall**: Of all actually similar pairs, what % did we find?
  - High recall = few false negatives
  - Important if you want to find all similar pairs

- **F1 Score**: Harmonic mean of precision and recall
  - Balances both precision and recall
  - Good single metric for model comparison

### Loss Metrics

- **BCE Loss** (Binary Cross-Entropy): Main training objective
  - Measures how well predicted probabilities match true labels
  - Lower is better
  - Range: 0 to ∞ (typically 0-2)

- **Training Loss vs Eval Loss**:
  - Similar values → good generalization
  - Eval >> Train → overfitting (model memorizes training data)
  - Eval < Train → unusual, may indicate data leakage

## Example Output

```
==================================================
Training Summary:
==================================================
Final training loss: 0.1234
Final validation loss: 0.1456
Validation accuracy: 0.9123
Validation precision: 0.8956
Validation recall: 0.9234
Validation F1: 0.9087

✅ Model appears to generalize well!
==================================================
```

## Best Practices

### 1. Always Use Validation Set
- Never evaluate only on training data
- Use at least 20% of data for validation
- For small datasets, consider k-fold cross-validation

### 2. Monitor for Overfitting
- Watch train/eval loss gap
- If overfitting occurs:
  - Reduce model complexity (lower LoRA rank)
  - Add more training data
  - Reduce number of epochs
  - Increase regularization (LoRA dropout)

### 3. Compare Multiple Runs
- Train with different hyperparameters
- Use `compare_models.py` to find the best configuration
- Consider ensemble methods for production

### 4. Dataset Quality
- Ensure validation set is representative of test data
- Check for data leakage between train and validation
- Balance positive and negative examples

### 5. Save Everything
- All metrics are automatically saved
- Keep checkpoints for best models
- Document experiments and results

## Troubleshooting

### "No evaluation metrics found"
- Make sure training completed successfully
- Check that `evaluation_strategy="epoch"` in TrainingArguments
- Verify validation dataset is not empty

### Validation loss is NaN
- Check for NaN values in your dataset
- Verify tokenization is correct
- Reduce learning rate

### Poor validation performance
- Model may be underfitting or overfitting
- Try different hyperparameters:
  - Learning rate: 1e-3 to 5e-2
  - LoRA rank: 2 to 64
  - LoRA alpha: 8 to 32
  - Number of epochs: 2 to 10

### Memory issues during evaluation
- Reduce `per_device_eval_batch_size` in TrainingArguments
- Use smaller validation set
- Enable gradient checkpointing

## Advanced: Custom Metrics

To add custom metrics, modify the `compute_metrics` function in `quantized_finetuning_v2.py`:

```python
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    # Add your custom metric here
    # Example: AUC-ROC score
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(labels, predictions)
    
    return {
        "accuracy": accuracy_score(...),
        "auc": auc,  # Your custom metric
        # ... other metrics
    }
```

## Related Files

- `src/quantized_finetuning_v2.py` - Main training script with evaluation
- `compare_models.py` - Model comparison tool
- `run_evaluation.py` - Detailed evaluation on test set
- `peft_lab_outputs/` - Saved models and metrics

## Next Steps

1. **Train baseline model**: Run with default settings
2. **Experiment with hyperparameters**: Try different configurations
3. **Compare results**: Use `compare_models.py` to find best model
4. **Final evaluation**: Test best model on held-out test set using `run_evaluation.py`
5. **Deploy**: Use best checkpoint for inference

---

For questions or issues, check the main README.md or examine the code in `src/quantized_finetuning_v2.py`.

