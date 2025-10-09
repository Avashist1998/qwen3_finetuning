import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.simple_inference import load_finetuned_model, get_similarity, load_base_model, extract_sentence_embedding_from_hidden_states
from src.utils import mean_intra_inter_pytorch

def pytorch_pca(data, n_components=3):
    """
    PyTorch PCA implementation using torch.pca_lowrank for efficient dimensionality reduction.
    
    Args:
        data: torch.Tensor of shape (n_samples, n_features)
        n_components: int, number of components to keep
    
    Returns:
        torch.Tensor of shape (n_samples, n_components)
    """
    # Use PyTorch's optimized PCA implementation
    U, S, V = torch.pca_lowrank(data, q=n_components, center=True)
    
    # Project data onto the principal components
    # U contains the projected data, S contains the singular values
    projected_data = U * S.unsqueeze(0)
    
    # Calculate explained variance ratio from singular values
    # For PCA, explained variance = S^2 / (m-1) where m is number of samples
    explained_variance = (S ** 2) / (data.shape[0] - 1)
    explained_variance_ratio = explained_variance / explained_variance.sum()
    
    return projected_data, explained_variance_ratio

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default="finetuned")
    parser.add_argument("--dataset_path", type=str, default="datasets/evaluation_set/evaluation_dataset.json")
    parser.add_argument("--base_model_path", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--quantization", type=bool, default=True)
    parser.add_argument("--peft_model_path", type=str, default="./peft_lab_outputs/checkpoint-1328")
    parser.add_argument("--kind", type=str, default="similarity", choices=["similarity", "visualization"])
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples for visualization (default: all)")
    return parser.parse_args()


def visualize_embeddings(batch_embeddings, roles):
    """Visualize embeddings using PyTorch PCA"""
    
    try:
        print(f"Visualizing {len(batch_embeddings)} embeddings for {len(set(roles))} unique roles")
        n_samples = len(batch_embeddings)
        
        # Ensure embeddings are on CPU and in the right format
        if hasattr(batch_embeddings, 'cpu'):
            batch_embeddings = batch_embeddings.cpu()
        elif hasattr(batch_embeddings, 'numpy'):
            batch_embeddings = torch.tensor(batch_embeddings)
        
        print(f"Embeddings shape: {batch_embeddings.shape}")
        
        # Apply PyTorch PCA for dimensionality reduction
        print("Applying PyTorch PCA...")
        embeddings_3d, explained_variance_ratio = pytorch_pca(batch_embeddings, n_components=3)
        print(f"PyTorch PCA completed. Shape: {embeddings_3d.shape}")
        print(f"Explained variance ratio: {explained_variance_ratio}")
        
        # Convert to numpy for matplotlib
        embeddings_3d_np = embeddings_3d.numpy()
        
        # Create color mapping for roles
        unique_roles = list(set(roles))
        print(f"Unique roles: {unique_roles}")
        
        # Create the 3D plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create scatter plot with different colors for each role
        for i, role in enumerate(unique_roles):
            mask = [r == role for r in roles]
            if any(mask):
                ax.scatter(embeddings_3d_np[mask, 0], embeddings_3d_np[mask, 1], embeddings_3d_np[mask, 2], 
                          label=role, alpha=0.7, s=50)
        
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title(f"PyTorch PCA Visualization of Job Role Embeddings ({n_samples} samples)")
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error during visualization: {e}")
        print(f"Embeddings shape: {batch_embeddings.shape if hasattr(batch_embeddings, 'shape') else 'Unknown'}")
        print(f"Number of roles: {len(roles)}")
        raise

def load_evaluation_dataset(evaluation_dataset_path):
    with open(evaluation_dataset_path, "r") as f:
        evaluation_dataset = json.load(f)
    return evaluation_dataset



def get_batch_embeddings(model, tokenizer, texts, batch_size=8, max_length=1024):
    """Compute embeddings for multiple texts in batches."""
    all_embeddings = []
    

    with torch.no_grad():
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize batch
            inputs = tokenizer(
                batch_texts, 
                padding="max_length", 
                max_length=max_length,
                truncation=True,
                return_tensors="pt"
            )

            # Move to device
            for key, value in inputs.items():
                inputs[key] = value.to(model.device)
            
            # Get embeddings
            outputs = model(**inputs)
            hidden_states = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"]
            embeddings = extract_sentence_embedding_from_hidden_states(hidden_states, attention_mask)
            
            # Keep on GPU as tensors (don't convert to numpy yet)
            all_embeddings.append(embeddings)
    
    # print(all_embeddings[0][:10], all_embeddings[1][:10])
    # Concatenate all embeddings into a single tensor
    return torch.cat(all_embeddings, dim=0)


def evaluate_model_ultra_optimized(model, tokenizer, evaluation_dataset):
    """Ultra-optimized version using PyTorch vectorized operations."""
    print("Computing embeddings for all texts...")
    
    # Pre-compute all embeddings (keep as tensors)
    all_texts = [item["text"] for item in evaluation_dataset]
    job_roles = [item["job_role"] for item in evaluation_dataset]
    embeddings_tensor = get_batch_embeddings(model, tokenizer, all_texts, max_length=2048)
    print("Computing similarity matrix...")
    # There is not difference when i check the embedding tensor
    diff_output = embeddings_tensor[0] != embeddings_tensor[1]
    print("Are the embeddings tensor" , embeddings_tensor[0][diff_output])
    # Normalize embeddings for cosine similarity
    embeddings_normalized = F.normalize(embeddings_tensor, p=2, dim=1)

    mean_intra, mean_inter, separation_score = mean_intra_inter_pytorch(embeddings_normalized, job_roles)
    print(f"Mean intra-class similarity: {mean_intra}")
    print(f"Mean inter-class similarity: {mean_inter}")
    print(f"Separation score: {separation_score}")

    # Compute entire similarity matrix at once using PyTorch
    similarity_matrix = torch.mm(embeddings_normalized, embeddings_normalized.t())
    
    # Scale from [-1, 1] to [0, 1] to match training format
    similarity_matrix_scaled = (similarity_matrix + 1) / 2
    
    # Build results dictionary
    scores = {}
    for i in range(len(evaluation_dataset)):
        for j in range(i+1, len(evaluation_dataset)):
            key = "_".join(sorted([evaluation_dataset[i]["job_role"], evaluation_dataset[j]["job_role"]]))
            
            similarity = similarity_matrix_scaled[i, j].item()
            
            if key not in scores:
                scores[key] = {
                    "job_role_1": evaluation_dataset[i]["job_role"], 
                    "job_role_2": evaluation_dataset[j]["job_role"], 
                    "similarity": [similarity]
                }
            else:
                scores[key]["similarity"].append(similarity)

    # Calculate statistics
    for key, value in scores.items():
        value["similarity_mean"] = np.mean(value["similarity"])
        value["similarity_std"] = np.std(value["similarity"])
    
    return scores


def evaluate_model(model, tokenizer, evaluation_dataset):
    scores = {}
    for i in range(len(evaluation_dataset)):
        for j in range(i+1, len(evaluation_dataset)):
            key = "_".join(sorted([evaluation_dataset[i]["job_role"], evaluation_dataset[j]["job_role"]]))
            similarity = get_similarity(model, tokenizer, evaluation_dataset[i]["text"], evaluation_dataset[j]["text"], max_length=1024).item()
            # if similarity > 0.95:
            #     print(evaluation_dataset[i]["text"])
            #     print(evaluation_dataset[j]["text"])
            if key not in scores:
                scores[key] = {"job_role_1": evaluation_dataset[i]["job_role"], "job_role_2": evaluation_dataset[j]["job_role"], "similarity": [similarity]}
            else:
                scores[key]["similarity"].append(similarity)

    for key, value in scores.items():
        value["similarity_mean"] = np.mean(value["similarity"])
        value["similarity_std"] = np.std(value["similarity"])
    return scores


def print_score_in_table(scores: dict):
    """Print a covariance matrix where row is job role and column is job role
    Make it output in a nice markdown table
    present the average similarity score in the table
    |           | Job Role 1 | Job Role 2 | Job Role 3 | ... |
    | Job Role 1 |           |           |           | ... |
    | Job Role 2 |           |           |           | ... |
    | Job Role 3 |           |           |           | ... |
    | ...       |           |           |           | ... |

    if score is empty, print a dash
    """
    
    # Extract all unique job roles
    job_roles = set()
    for key, value in scores.items():
        job_roles.add(value["job_role_1"])
        job_roles.add(value["job_role_2"])
    
    job_roles = sorted(list(job_roles))
    
    # Create a matrix to store similarity scores
    matrix = {}
    for role in job_roles:
        matrix[role] = {}
        for other_role in job_roles:
            matrix[role][other_role] = "-"
    
    # Fill in the similarity scores
    for key, value in scores.items():
        job_1 = value["job_role_1"]
        job_2 = value["job_role_2"]
        similarity = value["similarity_mean"]
        
        # Set both directions (since it's symmetric)
        matrix[job_1][job_2] = f"{similarity:.3f}"
        matrix[job_2][job_1] = f"{similarity:.3f}"
    
    # # Set diagonal to 1.0 (self-similarity)
    # for role in job_roles:
    #     matrix[role][role] = "1.000"
    
    # Print the header
    header = "|           |" + " | ".join(job_roles) + " |"
    separator = "|------------|" + " | ".join(["------------"] * len(job_roles)) + " |"
    
    print(header)
    print(separator)
    
    # Print each row
    for role in job_roles:
        row_values = [matrix[role][other_role] for other_role in job_roles]
        row = f"| {role} |" + " | ".join(row_values) + " |"
        print(row)


def main():

    args = parse_args()
    if args.type == "finetuned":
        assert args.base_model_path is not None, "Base model path is required for finetuned model"
        assert args.peft_model_path is not None, "Peft model path is required for finetuned model"
        model, tokenizer = load_finetuned_model(base_model_path=args.base_model_path, peft_model_path=args.peft_model_path, quantization=args.quantization)
    elif args.type == "base":
        model, tokenizer = load_base_model(base_model_path=args.base_model_path, quantization=args.quantization)
    else:
        raise ValueError(f"Invalid type: {args.type}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    evaluation_dataset = load_evaluation_dataset(args.dataset_path)
    if args.max_samples:
        evaluation_dataset = evaluation_dataset[:args.max_samples]
    print(f"Using {len(evaluation_dataset)} samples for evaluation")
    if args.kind == "similarity":
        scores = evaluate_model_ultra_optimized(model, tokenizer, evaluation_dataset)
        print_score_in_table(scores)
        # print(json.dumps(scores, indent=4))
    elif args.kind == "visualization":
        roles = [item["job_role"] for item in evaluation_dataset]
        all_texts = [item["text"] for item in evaluation_dataset]
        print("Computing embeddings for visualization...")
        batch_embeddings = get_batch_embeddings(model, tokenizer, all_texts, max_length=2048)
        print("Embeddings computed, creating visualization...")
        visualize_embeddings(batch_embeddings, roles)
    else:
        raise ValueError(f"Invalid kind: {args.kind}")
    


if __name__ == "__main__":
    main()