import os
import torch
import torch.nn.functional as F
from typing import Optional, Union, Any, List, Dict
from torch import nn
from datasets import Dataset
from transformers import TrainingArguments
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer, Trainer
from transformers import AutoModel, AutoTokenizer, data
from peft import LoraConfig, get_peft_model, TaskType
import argparse
from datetime import datetime



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
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_model = AutoModel.from_pretrained(model_id)
    # base_model = AutoModel.from_pretrained(model_id, attn_implementation="flash_attention_2")

    lora_config = LoraConfig(
        r=4, # Change to 2 to make on mac
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    model = get_peft_model(base_model, lora_config)


    def extract_sentence_embedding_from_hidden_states(hidden_states, attention_mask):
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.sum(mask_expanded, dim=1)
        return sum_embeddings / sum_mask

    # Compute metrics function for evaluation
    def compute_metrics(eval_pred):
        """
        Compute Mean Squared Error (MSE) for evaluating model performance using PyTorch.
        Measures how close predicted similarity scores are to true labels.
        """
        predictions, labels = eval_pred
        
        # Convert numpy arrays to PyTorch tensors for consistent computation
        predictions_tensor = torch.tensor(predictions, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        
        # Calculate MSE using PyTorch
        mse = F.mse_loss(predictions_tensor, labels_tensor, reduction='mean')
        
        # Calculate RMSE (Root Mean Squared Error) for interpretability
        rmse = torch.sqrt(mse)
        
        # Calculate MAE (Mean Absolute Error) as an additional metric
        mae = F.l1_loss(predictions_tensor, labels_tensor, reduction='mean')
        
        # Calculate R² score (coefficient of determination)
        # R² = 1 - (SS_res / SS_tot)
        ss_res = torch.sum((labels_tensor - predictions_tensor) ** 2)
        ss_tot = torch.sum((labels_tensor - torch.mean(labels_tensor)) ** 2)
        r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else torch.tensor(0.0)
        
        return {
            "mse": mse.item(),
            "rmse": rmse.item(),
            "mae": mae.item(),
            "r2_score": r2_score.item(),
        }

    # Custom trainer class that handles sentence pair training
    class SentencePairTrainer(Trainer):

        def compute_loss(self, model, inputs: dict[str, Union[torch.Tensor, Any]], return_outputs: bool = False, num_items_in_batch: Optional[torch.Tensor] = None):
            """Custom loss computation for sentence pairs"""

            input_ids_1 = inputs.get("input_ids_1")
            attention_mask_1 = inputs.get("attention_mask_1")
            input_ids_2 = inputs.get("input_ids_2")
            attention_mask_2 = inputs.get("attention_mask_2")
            labels = inputs.get("labels")

            try:
                # Get embeddings for sentence 1
                outputs1 = model(input_ids=input_ids_1, attention_mask=attention_mask_1)
                hidden_states1 = outputs1.last_hidden_state
                embeddings1 = extract_sentence_embedding_from_hidden_states(hidden_states1, attention_mask_1)
            except Exception as e:
                print(f"Some error happened for sentence 1 {input_ids_1} {attention_mask_1}")
                raise e
            
            try:
                # Get embeddings for sentence 2
                outputs2 = model(input_ids=input_ids_2, attention_mask=attention_mask_2)
                hidden_states2 = outputs2.last_hidden_state
                embeddings2 = extract_sentence_embedding_from_hidden_states(hidden_states2, attention_mask_2)
            except Exception as e:
                print(f"Some error happened for sentence 2 {input_ids_2} {attention_mask_2}")
                raise e
            
            # Compute cosine similarity
            cos_sim = F.cosine_similarity(embeddings1, embeddings2)
            
            # Scale similarity to [0, 1] range
            cos_sim_scaled = (cos_sim + 1) / 2
            
            # Ensure tensors are properly shaped for loss computation
            cos_sim_scaled = cos_sim_scaled.squeeze()
            labels_float = labels.float().squeeze()
            
            # Binary cross entropy loss
            loss = F.mse_loss(cos_sim_scaled, labels_float, reduction='mean')
            
            return (loss, {"cos_sim": cos_sim_scaled}) if return_outputs else loss
        
        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            """
            Custom prediction step to return cosine similarity scores for metrics computation.
            """
            has_labels = "labels" in inputs
            inputs = self._prepare_inputs(inputs)
            
            with torch.no_grad():
                # Compute loss and get outputs
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                cos_sim_scaled = outputs["cos_sim"]
                
            if prediction_loss_only:
                return (loss, None, None)
            
            # Return predictions (cosine similarities) and labels
            labels = inputs.get("labels")
            return (loss, cos_sim_scaled, labels)

    def preprocess_function(examples):
        queries = examples["sentence1"]
        max_length = args.max_length
        sentence1_encodings = tokenizer(queries, 
            padding="max_length", 
            max_length=max_length, 
            truncation=True, 
            return_tensors="pt")

        products = examples["sentence2"]
        sentence2_encodings = tokenizer(products,
            padding="max_length", 
            max_length=max_length, 
            truncation=True, 
            return_tensors="pt")

        # Debug: Check tokenization output
        # print(f"sentence1_encodings keys: {sentence1_encodings.keys()}")
        # print(f"input_ids shape: {len(sentence1_encodings['input_ids'])} x {len(sentence1_encodings['input_ids'][0])}")
        result = {
            "input_ids_1": sentence1_encodings["input_ids"],
            "attention_mask_1": sentence1_encodings["attention_mask"],
            "input_ids_2": sentence2_encodings["input_ids"],
            "attention_mask_2": sentence2_encodings["attention_mask"],
            "labels": torch.tensor(examples["labels"], dtype=torch.float)
        }
        return result

    if args.dataset_size:
        dataset = dataset.select(range(args.dataset_size))
        eval_dataset = eval_dataset.select(range(min(args.dataset_size // 5, len(eval_dataset))))

    print(f"Training dataset size: {len(dataset)}")
    print(f"Validation dataset size: {len(eval_dataset)}")

    # # Tokenize both datasets use this one the gpu memory issue
    # tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names, num_proc=16, load_from_cache_file=False, keep_in_memory=True)
    # tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True, remove_columns=eval_dataset.column_names, num_proc=16, load_from_cache_file=False, keep_in_memory=True)

    # Tokenize both datasets
    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)
    tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True, remove_columns=eval_dataset.column_names)

    def validate_dataset(dataset: Dataset):
        for i in range(len(dataset)):
            if type(dataset[i]["labels"]) not in [int, float]:
                print(dataset[i]["labels"], type(dataset[i]["labels"]))
                raise Exception(f"Label for index {i} is not a float {dataset[i]['labels']} {type(dataset[i]['labels'])}")
            
            if len(dataset[i]["input_ids_1"]) == 0:
                raise Exception(f"Input ids 1 for index {i} is empty {dataset[i]['input_ids_1']}")
            
            if len(dataset[i]["input_ids_1"]) != len(dataset[i]["attention_mask_1"]):
                raise Exception(f"Input ids 1 and attention mask 1 for index {i} have different shapes {dataset[i]['input_ids_1'].shape} != {dataset[i]['attention_mask_1'].shape}")
            
            if len(dataset[i]["input_ids_2"]) == 0:
                raise Exception(f"Input ids 2 for index {i} is empty {dataset[i]['input_ids_2']}")
            
            if len(dataset[i]["input_ids_2"]) != len(dataset[i]["attention_mask_2"]):
                raise Exception(f"Input ids 2 and attention mask 2 for index {i} have different shapes {dataset[i]['input_ids_2'].shape} != {dataset[i]['attention_mask_2'].shape}")
            
            if len(dataset[i]["attention_mask_1"]) == 0:
                raise Exception(f"Attention mask 1 for index {i} is empty {dataset[i]['attention_mask_1']}")
            
            if len(dataset[i]["attention_mask_2"]) == 0:
                raise Exception(f"Attention mask 2 for index {i} is empty {dataset[i]['attention_mask_2']}")

    validate_dataset(tokenized_dataset)
    validate_dataset(tokenized_eval_dataset)

    # Shuffle the training dataset
    tokenized_dataset = tokenized_dataset.shuffle(seed=42)
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Fine-tune embedding model with evaluation metrics")
    
    # Model and run configuration
    parser.add_argument("--run_name", type=str, default=f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                        help="Name for this training run")
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

    args = parser.parse_args()
    
    main(args)

