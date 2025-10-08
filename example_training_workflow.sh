#!/bin/bash
# Example workflow for training and comparing models with evaluation metrics

echo "=================================================="
echo "Model Training and Comparison Workflow"
echo "=================================================="

# 1. Train a baseline model with default settings
echo -e "\n[1/4] Training baseline model..."
python src/quantized_finetuning_v2.py \
    --run_name baseline_model \
    --dataset_path datasets/training_set_v2.json \
    --num_of_epochs 3 \
    --validation_split 0.2

# 2. Train a model with higher learning rate
echo -e "\n[2/4] Training with different hyperparameters..."
python src/quantized_finetuning_v2.py \
    --run_name high_lr_model \
    --dataset_path datasets/training_set_v2.json \
    --num_of_epochs 3 \
    --validation_split 0.2

# 3. Train with separate validation dataset
echo -e "\n[3/4] Training with separate validation set..."
python src/quantized_finetuning_v2.py \
    --run_name separate_eval_model \
    --dataset_path datasets/training_set_v2.json \
    --eval_dataset_path datasets/evaluation_dataset.json \
    --num_of_epochs 3

# 4. Compare all models
echo -e "\n[4/4] Comparing all trained models..."
python compare_models.py --export_csv model_comparison_results.csv

echo -e "\n=================================================="
echo "Workflow complete!"
echo "=================================================="
echo "View results in:"
echo "  - model_comparison_results.csv"
echo "  - peft_lab_outputs/[run_name]/"
echo ""
echo "To visualize training with TensorBoard:"
echo "  tensorboard --logdir=peft_lab_outputs/logs"
echo "=================================================="

