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

parser = argparse.ArgumentParser()
parser.add_argument("--run_name", type=str, default=f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-Embedding-0.6B")
parser.add_argument("--dataset_path", type=str, default="datasets/mini_dataset_v2.json")
parser.add_argument("--num_of_epochs", type=int, default=2)
parser.add_argument("--dataset_size", type=int, default=-1) # -1 for all

args = parser.parse_args()

dataset_name = args.dataset_path
dataset = Dataset.from_json(dataset_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

working_dir = './'
output_directory = os.path.join(working_dir, "peft_lab_outputs")
print(f"Output directory: {output_directory}/{args.run_name}")

# Debug: print first example to see structure
print(f"Dataset: {dataset.shape}")
print("Dataset columns:", dataset.column_names)


# model_id = "Qwen/Qwen3-Embedding-0.6B"
model_id = args.model_id
tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModel.from_pretrained(model_id)
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj"],
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
        loss = F.binary_cross_entropy(cos_sim_scaled, labels_float, reduction='mean')
        
        return (loss, {"cos_sim": cos_sim_scaled}) if return_outputs else loss



def preprocess_function(examples):
    queries = examples["sentence1"]
    max_length = 1024
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

if args.dataset_size != -1:
    dataset = dataset.select(range(args.dataset_size))

tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

def validate_dataset(dataset: Dataset):
    for i in range(len(dataset)):
        if type(dataset[i]["labels"]) != int:
            print(dataset[i]["labels"])
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

# Shuffle the dataset
tokenized_dataset = tokenized_dataset.shuffle(seed=42)

training_args = TrainingArguments(
    output_dir=os.path.join(output_directory, args.run_name),
    auto_find_batch_size=True, # Find a correct bvatch size that fits the size of Data.
    learning_rate= 3e-2, # Higher learning rate than full fine-tuning.
    num_train_epochs=args.num_of_epochs,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    use_cpu=device == "cpu",
    save_strategy="epoch",
    remove_unused_columns=False,
    label_names=["labels"],

    # logging
    logging_strategy="steps",
    logging_steps=10,
    logging_dir=os.path.join(output_directory, "logs", args.run_name),
    logging_first_step=True,

    report_to="tensorboard",
    # evaluation_strategy="no",
)


trainer = SentencePairTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()
trainer.save_model(output_directory)


