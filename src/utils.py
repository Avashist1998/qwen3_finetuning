"""
Fixed version of mean_intra_inter_pytorch function that addresses the sampling bias issue.
"""

import torch
import torch.nn.functional as F
import numpy as np
from itertools import combinations

def mean_intra_inter_pytorch(embeddings, job_roles):
    """
    Calculate mean intra-class and inter-class similarities using PyTorch.
    Fixed version that addresses sampling bias in inter-class calculation.
    
    Args:
        embeddings: torch.Tensor of shape (N, D) - normalized embeddings
        job_roles: list of job role labels for each embedding
    
    Returns:
        tuple: (mean_intra_similarity, mean_inter_similarity, separation_score)
    """
    # Convert string labels to integer indices for tensor operations
    unique_roles = list(set(job_roles))
    role_to_idx = {role: idx for idx, role in enumerate(unique_roles)}
    job_role_indices = torch.tensor([role_to_idx[role] for role in job_roles], device=embeddings.device)
    
    intra_vals = []
    unique_role_indices = torch.unique(job_role_indices)
    
    # Calculate intra-class similarities (unchanged - this part was correct)
    for role_idx in unique_role_indices:
        # Get indices for this role
        idx = (job_role_indices == role_idx).nonzero(as_tuple=True)[0]
        
        if len(idx) < 2:
            continue
            
        # Get embeddings for this role
        role_embeddings = embeddings[idx]
        
        # Calculate pairwise similarities (cosine similarity since embeddings are normalized)
        sims = torch.mm(role_embeddings, role_embeddings.t())
        
        # Take upper triangular excluding diagonal
        mask = torch.triu(torch.ones_like(sims), diagonal=1).bool()
        intra_similarities = sims[mask]
        intra_vals.append(intra_similarities)
    
    # Concatenate all intra-class similarities
    if intra_vals:
        intra_vals = torch.cat(intra_vals)
        mean_intra = torch.mean(intra_vals).item()
    else:
        mean_intra = 0.0
    
    # FIXED: Calculate inter-class similarities using systematic approach
    inter_vals = []
    
    # Get all unique role pairs
    for i in range(len(unique_roles)):
        for j in range(i + 1, len(unique_roles)):
            role_i_idx = role_to_idx[unique_roles[i]]
            role_j_idx = role_to_idx[unique_roles[j]]
            
            # Get indices for each role
            idx_i = (job_role_indices == role_i_idx).nonzero(as_tuple=True)[0]
            idx_j = (job_role_indices == role_j_idx).nonzero(as_tuple=True)[0]
            
            if len(idx_i) > 0 and len(idx_j) > 0:
                # Calculate all pairwise similarities between these two roles
                embeddings_i = embeddings[idx_i]
                embeddings_j = embeddings[idx_j]
                
                # Calculate similarity matrix between role i and role j
                sims_ij = torch.mm(embeddings_i, embeddings_j.t())
                
                # Add all similarities to inter-class values
                inter_vals.append(sims_ij.flatten())
    
    # Concatenate all inter-class similarities
    if inter_vals:
        inter_vals = torch.cat(inter_vals)
        mean_inter = torch.mean(inter_vals).item()
    else:
        mean_inter = 0.0
    
    separation_score = mean_intra - mean_inter
    
    return mean_intra, mean_inter, separation_score

def mean_intra_inter_pytorch_sampled(embeddings, job_roles, max_samples=10000):
    """
    Alternative fixed version that uses balanced sampling for large datasets.
    This version ensures equal representation of all role pairs.
    
    Args:
        embeddings: torch.Tensor of shape (N, D) - normalized embeddings
        job_roles: list of job role labels for each embedding
        max_samples: maximum number of inter-class pairs to sample
    
    Returns:
        tuple: (mean_intra_similarity, mean_inter_similarity, separation_score)
    """
    # Convert string labels to integer indices for tensor operations
    unique_roles = list(set(job_roles))
    role_to_idx = {role: idx for idx, role in enumerate(unique_roles)}
    job_role_indices = torch.tensor([role_to_idx[role] for role in job_roles], device=embeddings.device)
    
    intra_vals = []
    unique_role_indices = torch.unique(job_role_indices)
    
    # Calculate intra-class similarities (unchanged)
    for role_idx in unique_role_indices:
        idx = (job_role_indices == role_idx).nonzero(as_tuple=True)[0]
        
        if len(idx) < 2:
            continue
            
        role_embeddings = embeddings[idx]
        sims = torch.mm(role_embeddings, role_embeddings.t())
        mask = torch.triu(torch.ones_like(sims), diagonal=1).bool()
        intra_similarities = sims[mask]
        intra_vals.append(intra_similarities)
    
    if intra_vals:
        intra_vals = torch.cat(intra_vals)
        mean_intra = torch.mean(intra_vals).item()
    else:
        mean_intra = 0.0
    
    # FIXED: Balanced sampling for inter-class similarities
    inter_vals = []
    
    # Calculate total possible inter-class pairs
    total_inter_pairs = 0
    role_pair_info = []
    
    for i in range(len(unique_roles)):
        for j in range(i + 1, len(unique_roles)):
            role_i_idx = role_to_idx[unique_roles[i]]
            role_j_idx = role_to_idx[unique_roles[j]]
            
            idx_i = (job_role_indices == role_i_idx).nonzero(as_tuple=True)[0]
            idx_j = (job_role_indices == role_j_idx).nonzero(as_tuple=True)[0]
            
            if len(idx_i) > 0 and len(idx_j) > 0:
                num_pairs = len(idx_i) * len(idx_j)
                total_inter_pairs += num_pairs
                role_pair_info.append({
                    'idx_i': idx_i,
                    'idx_j': idx_j,
                    'num_pairs': num_pairs
                })
    
    if total_inter_pairs == 0:
        mean_inter = 0.0
    else:
        # Sample proportionally from each role pair
        samples_per_pair = max(1, max_samples // len(role_pair_info)) if role_pair_info else 0
        
        torch.manual_seed(0)  # For reproducibility
        
        for pair_info in role_pair_info:
            idx_i = pair_info['idx_i']
            idx_j = pair_info['idx_j']
            
            # Sample pairs from this role combination
            num_samples = min(samples_per_pair, len(idx_i) * len(idx_j))
            
            if num_samples > 0:
                # Generate all possible pairs for this role combination
                all_pairs_i = idx_i.repeat(len(idx_j))
                all_pairs_j = idx_j.repeat_interleave(len(idx_i))
                
                # Randomly sample from these pairs
                if len(all_pairs_i) > num_samples:
                    sample_indices = torch.randperm(len(all_pairs_i), device=embeddings.device)[:num_samples]
                    sampled_i = all_pairs_i[sample_indices]
                    sampled_j = all_pairs_j[sample_indices]
                else:
                    sampled_i = all_pairs_i
                    sampled_j = all_pairs_j
                
                # Calculate similarities for sampled pairs
                similarities = torch.sum(embeddings[sampled_i] * embeddings[sampled_j], dim=1)
                inter_vals.append(similarities)
        
        if inter_vals:
            inter_vals = torch.cat(inter_vals)
            mean_inter = torch.mean(inter_vals).item()
        else:
            mean_inter = 0.0
    
    separation_score = mean_intra - mean_inter
    
    return mean_intra, mean_inter, separation_score