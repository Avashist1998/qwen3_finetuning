import torch
from transformers import AutoModel, AutoTokenizer, QuantoConfig
from peft import PeftModel
import numpy as np
from torch.nn.functional import cosine_similarity

def compare_tensors(tensor1, tensor2, name="tensors", tolerance=1e-6):
    """Comprehensive tensor comparison with detailed output."""
    print(f"\n=== Comparing {name} ===")
    print(f"Shape: {tensor1.shape} vs {tensor2.shape}")
    print(f"Equal shapes: {tensor1.shape == tensor2.shape}")
    
    if tensor1.shape != tensor2.shape:
        print("❌ Shapes don't match!")
        return False
    
    # Check if exactly equal
    exactly_equal = torch.equal(tensor1, tensor2)
    print(f"Exactly equal: {exactly_equal}")
    
    if exactly_equal:
        print("✅ Tensors are identical!")
        return True
    
    # Check approximate equality
    diff = torch.abs(tensor1 - tensor2)
    max_diff = torch.max(diff).item()
    mean_diff = torch.mean(diff).item()
    
    print(f"Max difference: {max_diff:.8f}")
    print(f"Mean difference: {mean_diff:.8f}")
    print(f"Within tolerance ({tolerance}): {max_diff < tolerance}")
    
    # Show some sample values
    print(f"Sample values 1: {tensor1.flatten()[:5]}")
    print(f"Sample values 2: {tensor2.flatten()[:5]}")
    
    return max_diff < tolerance

def load_finetuned_model(base_model_path="Qwen/Qwen3-Embedding-0.6B", 
                        peft_model_path="./peft_lab_outputs/checkpoint-94", quantization=True):
    """Load the fine-tuned model for inference."""
    print("Loading fine-tuned model...")
    
    # Load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    config = None
    if quantization:
        config = QuantoConfig(weights="int8")
    base_model = AutoModel.from_pretrained(base_model_path, quantization_config=config)    
    # Load fine-tuned LoRA weights
    model = PeftModel.from_pretrained(base_model, peft_model_path)
    model.eval()
    
    print("Model loaded successfully!")
    return model, tokenizer


def load_base_model(base_model_path="Qwen/Qwen3-Embedding-0.6B", quantization=True):
    """Load the base model for inference."""
    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    config = None
    if quantization:
        config = QuantoConfig(weights="int8") 
    base_model = AutoModel.from_pretrained(base_model_path, quantization_config=config)
    base_model.eval()
    return base_model, tokenizer


def extract_sentence_embedding_from_hidden_states(hidden_states, attention_mask):
    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
    sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
    sum_mask = torch.sum(mask_expanded, dim=1)
    return sum_embeddings / sum_mask

def get_similarity(model, tokenizer, text1, text2, max_length=70):
    """Calculate similarity between two texts."""

    # assert text1 != text2, "Inputs should be different"
    # Tokenize both texts
    inputs1 = tokenizer(text1, padding="max_length", max_length=max_length, 
                        truncation=True, return_tensors="pt")
    for key, value in inputs1.items():
        inputs1[key] = value.to(model.device)
    inputs2 = tokenizer(text2, padding="max_length", max_length=max_length, 
                        truncation=True, return_tensors="pt")
    for key, value in inputs2.items():
        inputs2[key] = value.to(model.device)
    # assert not torch.equal(inputs1.input_ids, inputs2.input_ids), "Inputs should be different for meaningful similarity"

    # Get embeddings
    with torch.no_grad():
        # Get embedding for text1
    
        outputs1 = model(**inputs1)
        hidden_states = outputs1.last_hidden_state
        attention_mask = inputs1["attention_mask"]
        emb1 = extract_sentence_embedding_from_hidden_states(hidden_states, attention_mask)

        # Get embedding for text2
        outputs2 = model(**inputs2)
        hidden_states = outputs2.last_hidden_state
        attention_mask = inputs2["attention_mask"]
        emb2 = extract_sentence_embedding_from_hidden_states(hidden_states, attention_mask)

    # Calculate cosine similarity
    similarity = cosine_similarity(emb1, emb2)[0]
    
    # Scale from [-1, 1] to [0, 1] to match training format
    similarity_scaled = (similarity + 1) / 2
    
    return similarity_scaled

def main():
    # Test cases
    test_cases = [
        ("Software Engineer", "Software Developer"),
        ("Data Scientist", "Machine Learning Engineer"), 
        ("Software Engineer", "Marketing Manager"),
        ("Python Developer", "Java Developer"),
        ("Frontend Developer", "Backend Developer"),
    ]
    
    print("\n" + "=" * 60)
    print("COMPARING BASE MODEL vs FINE-TUNED MODEL")
    print("=" * 60)
    
    # Load base model
    print("\n1. Testing BASE model:")
    print("-" * 30)
    base_model, tokenizer = load_base_model()
    
    for i, (text1, text2) in enumerate(test_cases):
        similarity = get_similarity(base_model, tokenizer, text1, text2)
        print(f"'{text1}' vs '{text2}': {similarity:.4f}")
    
    # Load fine-tuned model
    print("\n2. Testing FINE-TUNED model:")
    print("-" * 30)
    finetuned_model, _ = load_finetuned_model()
    
    for text1, text2 in test_cases:
        similarity = get_similarity(finetuned_model, tokenizer, text1, text2)
        print(f"'{text1}' vs '{text2}': {similarity:.4f}")
    
    # Compare improvements
    print("\n3. IMPROVEMENT ANALYSIS:")
    print("-" * 30)
    print("Comparing fine-tuned vs base model improvements:")
    
    for text1, text2 in test_cases:
        base_sim = get_similarity(base_model, tokenizer, text1, text2)
        finetuned_sim = get_similarity(finetuned_model, tokenizer, text1, text2)
        improvement = finetuned_sim - base_sim
        print(f"'{text1}' vs '{text2}': {improvement:+.4f} improvement")

if __name__ == "__main__":
    main()
