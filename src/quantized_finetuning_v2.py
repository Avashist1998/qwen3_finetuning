import os
import torch
from typing import Tuple
from torch import nn
from datasets import Dataset
from transformers import TrainingArguments
from transformers import AutoTokenizer
from transformers import AutoModel
from peft import LoraConfig, get_peft_model, TaskType
import argparse
from datetime import datetime
from src.dataloader import preprocess_function
from src.sentence_traininer import SentencePairTrainer, compute_metrics


R = 1 # r=2, r=4, # Change to 2 to make on mac

def model_loader(model_id: str) -> Tuple[AutoModel, AutoTokenizer]:
    """
    Load the model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_model = AutoModel.from_pretrained(model_id)
    # base_model = AutoModel.from_pretrained(model_id, attn_implementation="flash_attention_2")
    lora_config = LoraConfig(
        
        r=R,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    model = get_peft_model(base_model, lora_config)
    return model, tokenizer


def main(args):
    
    # Load training dataset
    dataset_name = args.dataset_path
    dataset = Dataset.from_json(dataset_name)

    # Load or create validation dataset
    if args.eval_dataset_path:
        print(f"Loading validation dataset from: {args.eval_dataset_path}")
        eval_dataset = Dataset.from_json(args.eval_dataset_path)
    else:
        # Split dataset into train and validation
        print(f"Splitting dataset with validation ratio: {args.validation_split}")
        dataset = dataset.shuffle(seed=42)
        split_idx = int(len(dataset) * (1 - args.validation_split))
        eval_dataset = dataset.select(range(split_idx, len(dataset)))
        dataset = dataset.select(range(split_idx))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    working_dir = './'
    output_directory = os.path.join(working_dir, "peft_lab_outputs")
    print(f"Output directory: {output_directory}/{args.run_name}")

    # Debug: print first example to see structure
    print(f"Training dataset: {dataset.shape}")
    print(f"Validation dataset: {eval_dataset.shape}")
    print("Dataset columns:", dataset.column_names)


    # model_id = "Qwen/Qwen3-Embedding-0.6B"
    model_id = args.model_id
    model, tokenizer = model_loader(model_id)


    if args.dataset_size:
        dataset = dataset.select(range(args.dataset_size))
        eval_dataset = eval_dataset.select(range(min(args.dataset_size // 5, len(eval_dataset))))

    print(f"Training dataset size: {len(dataset)}")
    print(f"Validation dataset size: {len(eval_dataset)}")

    # # Tokenize both datasets use this one the gpu memory issue
    # tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names, num_proc=16, load_from_cache_file=False, keep_in_memory=True, fn_kwargs={"tokenizer": tokenizer, "max_length": args.max_length})
    # tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True, remove_columns=eval_dataset.column_names, num_proc=16, load_from_cache_file=False, keep_in_memory=True, fn_kwargs={"tokenizer": tokenizer, "max_length": args.max_length})

    # Tokenize both datasets
    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names, fn_kwargs={"tokenizer": tokenizer, "max_length": args.max_length})
    tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True, remove_columns=eval_dataset.column_names, fn_kwargs={"tokenizer": tokenizer, "max_length": args.max_length})
    
    ## Set the format of the tokenized datasets to torch
    tokenized_dataset.set_format(type='torch', columns=['input_ids_1', 'attention_mask_1', 'input_ids_2', 'attention_mask_2', 'labels'])
    tokenized_eval_dataset.set_format(type='torch', columns=['input_ids_1', 'attention_mask_1', 'input_ids_2', 'attention_mask_2', 'labels'])

    # # Validate the tokenized datasets
    # validate_dataset(tokenized_dataset)
    # validate_dataset(tokenized_eval_dataset)
    # # Shuffle the training dataset
    # tokenized_dataset = tokenized_dataset.shuffle(seed=42)
    # print(f"Shuffled dataset size: {len(tokenized_dataset)}")

    training_args = TrainingArguments(
        output_dir=os.path.join(output_directory, args.run_name),
        auto_find_batch_size=True, # Find a correct batch size that fits the size of Data.
        learning_rate= 3e-2, # Higher learning rate than full fine-tuning.
        num_train_epochs=args.num_of_epochs,
        per_device_train_batch_size=4, # Change to 2 for mac
        per_device_eval_batch_size=4,  # Larger batch size for evaluation
        gradient_accumulation_steps=4,
        use_cpu=device == "cpu",
        bf16=device != "cpu", # Specific for the training
        # bf16=False,
        save_strategy="steps",  # Changed to match eval_strategy
        save_steps=50,  # Save at same frequency as eval
        save_total_limit=5,
        remove_unused_columns=False,
        label_names=["labels"],

        # Evaluation settings
        load_best_model_at_end=True,  # Load best model at the end
        metric_for_best_model="eval_mse",  # Use MSE to determine best model
        eval_strategy="steps",
        eval_steps=50,
        greater_is_better=False,  # Lower MSE is better

        # Logging
        logging_strategy="steps",
        logging_steps=10,
        logging_dir=os.path.join(output_directory, "logs", args.run_name),
        logging_first_step=True,

        report_to="tensorboard",
    )


    trainer = SentencePairTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Train the model
    train_result = trainer.train(resume_from_checkpoint=args.from_checkpoint)
    
    # Save the final model
    trainer.save_model(os.path.join(output_directory, args.run_name))
    
    # Save training metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(tokenized_dataset)
    metrics["eval_samples"] = len(tokenized_eval_dataset)
    
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    # Run final evaluation
    print("\n" + "="*50)
    print("Running final evaluation on validation set...")
    print("="*50)
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    
    # Print summary
    print("\n" + "="*50)
    print("Training Summary:")
    print("="*50)
    print(f"Final training loss: {metrics.get('train_loss', 'N/A'):.4f}")
    print(f"Final validation loss: {eval_metrics.get('eval_loss', 'N/A'):.4f}")
    print(f"\nValidation Metrics (PyTorch-based):")
    print(f"  MSE:  {eval_metrics.get('eval_mse', 'N/A'):.4f}")
    print(f"  RMSE: {eval_metrics.get('eval_rmse', 'N/A'):.4f}")
    print(f"  MAE:  {eval_metrics.get('eval_mae', 'N/A'):.4f}")
    print(f"  R² Score: {eval_metrics.get('eval_r2_score', 'N/A'):.4f}")
    print("="*50)


def quantization_args():
    parser = argparse.ArgumentParser(description="Fine-tune embedding model with evaluation metrics")
    
    # Model and run configuration
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-Embedding-0.6B",
                        help="Hugging Face model ID")
    
    # Dataset configuration
    parser.add_argument("--dataset_path", type=str, default="datasets/mini_dataset_v2.json",
                        help="Path to training dataset (JSON format)")
    parser.add_argument("--eval_dataset_path", type=str, default=None,
                        help="Path to separate evaluation dataset (JSON format). If not provided, will split from training data.")
    parser.add_argument("--validation_split", type=float, default=0.2,
                        help="Fraction of data to use for validation when eval_dataset_path is not provided (default: 0.2)")
    
    # Training configuration
    parser.add_argument("--num_of_epochs", type=int, default=2,
                        help="Number of training epochs")
    parser.add_argument("--dataset_size", type=int, default=None,
                        help="Limit dataset size for faster experimentation (default: use all data)")
    

    parser.add_argument("--from_checkpoint", type=bool, default=None,
                        help="Path to checkpoint to load from")


    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum length of the input tokens")

    return parser.parse_args()

if __name__ == "__main__":

    args = quantization_args()


    args.run_name = f'r{R}_max_length{args.max_length}_dataset_size{args.dataset_size}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}'
    main(args)

