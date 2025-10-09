import argparse
import pandas as pd
from random import shuffle
from typing import TypedDict, cast
import numpy as np
import ast
import json

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, default="datasets/raw_jd_data.csv")
parser.add_argument("--dataset_size", type=int, default=100)
#parser.add_argument("--ner_based", type=bool, default=False)
parser.add_argument("--type", type=str, default="ner", choices=["ner", "org", "hybrid"])
parser.add_argument("--num_of_negative_examples", type=int, default=3)
args = parser.parse_args()
args.output_path = f"datasets/training_set/{args.dataset_size}_{args.type}.json"


dataset_size = args.dataset_size
df = pd.read_csv(args.dataset_path)
print(f"Shape of the dataset: {df.shape}")
# Parse the string representations back to lists
df["cleaned_requirements"] = df["candidate_requirements"].apply(
    lambda x: ast.literal_eval(x) if pd.notna(x) and isinstance(x, str) and x.startswith('[') else []
)
df["cleaned_responsibilities"] = df["cleaned_responsibilities"].apply(
    lambda x: ast.literal_eval(x) if pd.notna(x) and isinstance(x, str) and x.startswith('[') else []
)

df = df[~df["cleaned_requirements"].isna() | ~df["cleaned_responsibilities"].isna()]
df = df[~df["cleaned_requirements"].apply(lambda x: len(x) == 0) & ~df["cleaned_responsibilities"].apply(lambda x: len(x) == 0)]

def format_job_description(row):
    job_role = row["job_role"]
    job_requirements = row["cleaned_requirements"]
    job_responsibilities = row["cleaned_responsibilities"]
    job_responsibilities = "\n".join(job_responsibilities)
    job_requirements = "\n".join(job_requirements)

    prompt = f"""{job_responsibilities}
    {job_requirements}
    """
    return prompt



class Triplet(TypedDict):
    sentence1: str
    sentence2: str
    labels: float


def generate_triplet(df, positive_index: int, num_of_negative_examples: int) -> list[Triplet]:

    """Generate triplets for training
    - Input will be the full job description
    - So a positive example is picked from a same job role 
    - Negative is picked from a random outside that role
    """

    anchor = df.iloc[positive_index]["job_description"]
    anchor_role = df.iloc[positive_index]["job_role"]
    anchor_job_family = df.iloc[positive_index]["job_family"]
    # Get indices of rows with same role (excluding the anchor itself)
    same_role_mask = (df["job_role"] == anchor_role) & (df.index != df.iloc[positive_index].name)
    same_role_indices = df[same_role_mask].index.tolist()

    # Get indices of rows with same job family (excluding the anchor itself)
    same_family_mask = (df["job_family"] == anchor_job_family) & (df.index != df.iloc[positive_index].name)
    same_family_indices = df[same_family_mask].index.tolist()
    
    # Get indices of rows with different roles
    different_role_mask = df["job_role"] != anchor_role
    different_role_indices = df[different_role_mask].index.tolist()
    
    # Check if we have enough examples
    if len(same_role_indices) < 1:
        print(f"Warning: Only {len(same_role_indices)} examples of same role available, but 1 requested")
        num_of_negative_examples = len(same_role_indices)
        if num_of_negative_examples == 0:
            return []
    
    if len(different_role_indices) < num_of_negative_examples:
        print(f"Warning: Only {len(different_role_indices)} examples of different role available, but {num_of_negative_examples} requested")
        num_of_negative_examples = min(num_of_negative_examples, len(different_role_indices))
        if num_of_negative_examples == 0:
            return []
    
    # Pick random indices from same role (excluding anchor)
    positive_indices = np.random.choice(same_role_indices, size=1, replace=False)
    positive_jds = df.loc[positive_indices]["job_description"].tolist()
    postitive_job_role = df.loc[positive_indices]["job_role"].tolist()[0]

    # Same job family Example (different role but same family)
    if len(same_family_indices) > 0:
        job_family_index = np.random.choice(same_family_indices, size=1, replace=False)[0]
        job_family_jd = df.loc[job_family_index, "job_description"]
        job_family_job_role = df.loc[job_family_index, "job_role"]

    # Pick random indices from different roles
    negative_indices = np.random.choice(different_role_indices, size=num_of_negative_examples, replace=False)
    negative_jds = df.loc[negative_indices]["job_description"].tolist()
    negative_job_roles = df.loc[negative_indices]["job_role"].tolist()
    triplets = []
    for positive_jd in positive_jds:
        triplets.append({
            "sentence1": anchor,
            "sentence2": positive_jd,
            "labels": 0.3,
            "anchor_job_role": anchor_role,
            "postitive_job_roles": postitive_job_role,
            
        })

    if len(same_family_indices) > 0:
        triplets.append({
            "sentence1": anchor,
            "sentence2": job_family_jd,
            "labels": 0.6,
            "anchor_job_role": anchor_role,
            "job_family_job_role": job_family_job_role,
        })

    for negative_job_role, negative_jd in zip(negative_job_roles, negative_jds):
        triplets.append({
            "sentence1": anchor,
            "sentence2": negative_jd,
            "labels": 1.0,
            "anchor_job_role": anchor_role,
            "negative_job_roles": negative_job_role,
        })

    return triplets

training_set = []
stats = {
    "number_of_input_docs": args.dataset_size,
    "number_row_processed": 0,
    "number_of_triplets_generated": 0,
    "job_based_stats": {}
}

import random
index = list(range(df.shape[0]))
random.shuffle(index)
random_set = index[:args.dataset_size]

if args.type == "ner":
    df["job_description"] = df.apply(format_job_description, axis=1)
elif args.type == "hybrid":
    df["formatted_job_description"] = df.apply(format_job_description, axis=1)
    df["job_description"] = df.apply(
        lambda row: row["formatted_job_description"] if random.random() < 0.5 else row["translated_job_description"], 
        axis=1
    )
else:
    df["job_description"] = df["translated_job_description"]


for i in random_set:
    try:
        triplets = generate_triplet(df, i, args.num_of_negative_examples)
        stats["number_row_processed"] += 1
        stats["number_of_triplets_generated"] += len(triplets)
        stats["job_based_stats"][df.iloc[i]["job_role"]] = stats["job_based_stats"].get(df.iloc[i]["job_role"], 0) + 1
    except Exception as e:
        print(f"Error generating triplets for {i}th row: {e}")
        continue
    training_set.extend(triplets)

print(f"Generated {len(training_set)} triplets")
print(json.dumps(stats, indent=4))


shuffle(training_set)
with open(args.output_path, "w") as f:
    json.dump(training_set, f, indent=4)
