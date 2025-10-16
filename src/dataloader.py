import torch
from transformers import AutoTokenizer
from datasets import Dataset

def preprocess_function(examples: dict, tokenizer: AutoTokenizer, max_length: int = 512) -> dict:
    queries = examples["sentence1"]
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