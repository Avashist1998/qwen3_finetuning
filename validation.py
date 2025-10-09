import requests
import pandas as pd 
import json
import numpy as np
from typing import Dict, List, Any
from datetime import datetime

import argparse
from src.simple_inference import load_finetuned_model, get_similarity, load_base_model, extract_sentence_embedding_from_hidden_states

def print_results_formatted(results: List[Dict[str, Any]], top_n: int = 3) -> None:
    """
    Print the role adjacency results in a nicely formatted CLI output.
    
    Args:
        results: List of dictionaries containing role and similar roles data
        top_n: Number of top similar roles to display (default: 3)
    """
    print("\n" + "=" * 100)
    print("ROLE ADJACENCY ANALYSIS RESULTS".center(100))
    print("=" * 100 + "\n")
    
    for idx, item in enumerate(results, 1):
        # Header for each role
        print(f"\n{'─' * 100}")
        print(f"[{idx}/{len(results)}] SOURCE ROLE")
        print(f"{'─' * 100}")
        
        # Role details
        print(f"  Role:        {item['role']}")
        print(f"  Company:     {item['company']}")
        print(f"  Job Family:  {item['job_family']}")
        
        # Similar roles section
        similar_roles = item.get('similar_roles', [])
        # Sort by score in descending order
        similar_roles_sorted = sorted(similar_roles, key=lambda x: x.get('score', 0.0), reverse=True)
        total_found = len(similar_roles_sorted)
        # Limit to top N
        similar_roles_display = similar_roles_sorted[:top_n]
        
        print(f"\n  SIMILAR ROLES (showing top {len(similar_roles_display)} of {total_found} found):")
        
        if similar_roles_display:
            print(f"  {'─' * 96}")
            print(f"  {'Rank':<6} {'Similar Role':<40} {'Score':<12} {'Normalized':<12}")
            print(f"  {'─' * 96}")
            
            for rank, similar_role in enumerate(similar_roles_display, 1):
                role_name = similar_role.get('job_role', 'N/A')
                score = similar_role.get('score', 0.0)
                normalized_score = similar_role.get('normalized_score', 0.0)
                
                # Color coding based on normalized similarity score
                if normalized_score >= 0.08:
                    indicator = "🟢"  # High similarity
                elif normalized_score >= 0.04:
                    indicator = "🟡"  # Medium similarity
                else:
                    indicator = "🔴"  # Low similarity
                
                print(f"  {rank:<6} {role_name:<40} {indicator} {score:<10.4f} {normalized_score:.4f}")
        else:
            print("  No similar roles found.")
        
        print()
    
    # Summary statistics
    print("\n" + "=" * 100)
    print("SUMMARY".center(100))
    print("=" * 100)
    print(f"\n  Total roles analyzed:        {len(results)}")
    
    total_similar_roles = sum(len(item.get('similar_roles', [])) for item in results)
    avg_similar_roles = total_similar_roles / len(results) if results else 0
    
    print(f"  Total similar roles found:   {total_similar_roles}")
    print(f"  Average per role:            {avg_similar_roles:.2f}")
    
    # Find roles with highest and lowest similar roles
    if results:
        max_similar = max(results, key=lambda x: len(x.get('similar_roles', [])))
        min_similar = min(results, key=lambda x: len(x.get('similar_roles', [])))
        
        print(f"\n  Most similar roles found:    {len(max_similar.get('similar_roles', []))} for '{max_similar['role']}'")
        print(f"  Least similar roles found:   {len(min_similar.get('similar_roles', []))} for '{min_similar['role']}'")
    
    print("\n" + "=" * 100 + "\n")


def export_to_csv(results: List[Dict[str, Any]], output_file: str = None, top_n: int = 3, is_lora_model: bool = False) -> str:
    """
    Export the role adjacency results to a CSV file for Excel analysis.
    
    Args:
        results: List of dictionaries containing role and similar roles data
        output_file: Optional custom output filename
        top_n: Number of top similar roles to include (default: 3)
        
    Returns:
        The path to the generated CSV file
    """
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"role_adjacency_results_{timestamp}_{'lora' if is_lora_model else 'base'}.csv"
    
    # Flatten the nested structure for CSV
    flattened_data = []
    
    for item in results:
        source_role = item['role']
        company = item['company']
        job_family = item['job_family']
        similar_roles = item.get('similar_roles', [])
        
        # Sort by score in descending order
        similar_roles_sorted = sorted(similar_roles, key=lambda x: x.get('score', 0.0), reverse=True)
        
        # Limit to top N similar roles
        similar_roles_export = similar_roles_sorted[:top_n]
        
        if similar_roles_export:
            for rank, similar_role in enumerate(similar_roles_export, 1):
                flattened_data.append({
                    'source_role': source_role,
                    'source_company': company,
                    'source_job_family': job_family,
                    'rank': rank,
                    'similar_role': similar_role.get('job_role', 'N/A'),
                    'similarity_score': similar_role.get('score', 0.0),
                    'normalized_score': similar_role.get('normalized_score', 0.0)
                })
        else:
            # Include roles with no similar roles found
            flattened_data.append({
                'source_role': source_role,
                'source_company': company,
                'source_job_family': job_family,
                'rank': 0,
                'similar_role': 'No similar roles found',
                'similarity_score': 0.0,
                'normalized_score': 0.0
            })
    
    # Create DataFrame and export to CSV
    df = pd.DataFrame(flattened_data)
    df.to_csv(output_file, index=False)
    
    return output_file


def get_similar_roles(data: pd.DataFrame, embeddings: np.ndarray, top_n: int = 3):
    """
    Find similar roles based on mean embeddings.
    
    Args:
        data: DataFrame with role, job_family, job_description columns
        embeddings: numpy array of embeddings corresponding to each row in data
        top_n: number of top similar roles to return
    
    Returns:
        List of dictionaries containing role and similar roles data
    """
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Add embeddings to dataframe temporarily
    data_with_embeddings = data.copy()
    data_with_embeddings['embedding'] = list(embeddings)
    
    # Group by role and compute mean embedding for each unique role
    role_embeddings = {}
    role_info = {}
    
    for role_name, group in data_with_embeddings.groupby("role"):
        # Stack all embeddings for this role and compute mean
        role_embs = np.stack(group['embedding'].values)
        mean_embedding = np.mean(role_embs, axis=0)
        role_embeddings[role_name] = mean_embedding
        
        # Store role info (use first occurrence)
        role_info[role_name] = {
            'role': role_name,
            'job_family': group['job_family'].iloc[0],
            'company': 'Draup Inc.'  # Default company
        }
    
    # Convert to arrays for similarity computation
    role_names = list(role_embeddings.keys())
    embedding_matrix = np.stack([role_embeddings[name] for name in role_names])
    
    # Compute cosine similarity between all role pairs
    similarity_matrix = cosine_similarity(embedding_matrix)
    similarity_matrix = (similarity_matrix + 1)/2
    
    # For each role, find top N most similar roles (excluding itself)
    results = []
    for i, role_name in enumerate(role_names):
        # Get similarity scores for this role
        similarities = similarity_matrix[i]
        
        # Get indices of top N+1 similar roles (excluding itself)
        # We get N+1 because the most similar will be itself
        top_indices = np.argsort(similarities)[::-1][1:top_n+1]
        
        # Build similar roles list
        similar_roles = []
        for idx in top_indices:
            similar_roles.append({
                'job_role': role_names[idx],
                'score': float(similarities[idx]),
                'normalized_score': float(similarities[idx])  # Already normalized with cosine similarity
            })
        
        # Add to results
        result_item = role_info[role_name].copy()
        result_item['similar_roles'] = similar_roles
        results.append(result_item)
    
    return results

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="datasets/validation_set/roles_2.csv")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--peft_model_path", type=str, default="./peft_lab_outputs/eval_test/checkpoint-9")
    parser.add_argument("--company", type=str, default="Draup Inc.")
    parser.add_argument("--is_lora", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--top_n", type=int, default=3, help="Number of top similar roles to find")
    return parser.parse_args()

import torch


def clean_string_and_unicode(string: str):
    string = string.encode('utf-8', errors='ignore')
    string = string.decode('utf-8')
    return string

if __name__ == "__main__":
    args = parse_args()
    data = pd.read_csv(args.dataset_path)
    data.rename(columns={"Draup Family": "job_family", "Draup Role": "role", "Job Description": "job_description"}, inplace=True)
    data = data[["role", "job_family", "job_description"]]
    data["job_description"] = data["job_description"].apply(clean_string_and_unicode)

    if args.batch_size:
        data = data[:args.batch_size]
    

    if args.is_lora:
        model, tokenizer = load_finetuned_model(base_model_path=args.model_path, peft_model_path=args.peft_model_path)
    else:
        model, tokenizer = load_base_model(base_model_path=args.model_path)


    # Tokenize all the descriptions
    input_ids, attention_masks = [], []
    for index, row in data.iterrows():
        tokenized = tokenizer(row["job_description"], 
            padding="max_length", 
            max_length=512, 
            truncation=True, 
            return_tensors="pt")
        input_ids.append(tokenized["input_ids"].squeeze(0).to(model.device))
        attention_masks.append(tokenized["attention_mask"].squeeze(0).to(model.device))
    

    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)
    print(input_ids.shape, attention_masks.shape)
    with torch.no_grad():
        hidden_states = model(input_ids=input_ids, attention_mask=attention_masks).last_hidden_state
        extracted_embeddings = extract_sentence_embedding_from_hidden_states(hidden_states, attention_masks)

    extracted_embeddings = extracted_embeddings.cpu().numpy()
    
    # Get similar roles
    res = get_similar_roles(data, extracted_embeddings, top_n=args.top_n)

    # Display formatted results
    print_results_formatted(res, top_n=args.top_n)
    
    # Export to CSV
    csv_file = export_to_csv(res, top_n=args.top_n, is_lora_model=args.is_lora)
    print(f"\n✅ Results exported to: {csv_file}")
    print(f"   Showing top {args.top_n} similar roles per source role")
    print(f"   Total rows in CSV: {sum(min(len(item.get('similar_roles', [])), args.top_n) or 1 for item in res)}")
    print(f"   You can now open this file in Excel for further analysis.\n")

