import argparse
import pandas as pd
import numpy as np
from collections import Counter


def analyze_dataset(file_path: str):
    """
    Analyze dataset and provide comprehensive statistics.
    
    Args:
        file_path: Path to the CSV dataset
    """
    # Load the dataset
    print(f"\n{'='*80}")
    print(f"DATASET ANALYTICS: {file_path}")
    print(f"{'='*80}\n")
    
    df = pd.read_csv(file_path)
    
    # Basic counts
    print("📊 BASIC STATISTICS")
    print("-" * 80)
    print(f"Total number of records: {len(df):,}")
    print(f"Number of columns: {len(df.columns)}")
    print(f"Column names: {', '.join(df.columns.tolist())}")
    print()
    
    # Check for the description column (could be different names)
    description_col = None
    for col in ['translated_job_description', 'job_description', 'description']:
        if col in df.columns:
            description_col = col
            break
    
    # Job Family Distribution
    if 'job_family' in df.columns:
        print("\n📈 JOB FAMILY DISTRIBUTION")
        print("-" * 80)
        
        # Count and percentage for all job families
        job_family_counts = df['job_family'].value_counts()
        total_count = len(df)
        
        print(f"Total unique job families: {len(job_family_counts)}")
        print()
        
        # Top 5 job families
        print("Top 5 Job Families:")
        print(f"{'Rank':<6} {'Job Family':<50} {'Count':<10} {'Percentage':<10}")
        print("-" * 80)
        
        for rank, (job_family, count) in enumerate(job_family_counts.head(50).items(), 1):
            percentage = (count / total_count) * 100
            job_family_name = job_family if pd.notna(job_family) else "(Missing/NaN)"
            print(f"{rank:<6} {str(job_family_name)[:48]:<50} {count:<10} {percentage:>6.2f}%")
        
        print()
        
        # Show distribution summary
        top_5_total = job_family_counts.head(5).sum()
        top_5_percentage = (top_5_total / total_count) * 100
        print(f"Top 5 job families represent: {top_5_percentage:.2f}% of the dataset")
        print(f"Remaining {len(job_family_counts) - 5} job families: {100 - top_5_percentage:.2f}%")
    else:
        print("\n⚠️  'job_family' column not found in dataset")
    
    # Job Role Distribution
    if 'job_role' in df.columns:
        print("\n📋 JOB ROLE DISTRIBUTION")
        print("-" * 80)
        job_role_counts = df['job_role'].value_counts()
        print(f"Total unique job roles: {len(job_role_counts)}")
        
        # Top 5 job roles
        print("\nTop 5 Job Roles:")
        print(f"{'Rank':<6} {'Job Role':<50} {'Count':<10}")
        print("-" * 80)
        
        for rank, (job_role, count) in enumerate(job_role_counts.head(5).items(), 1):
            job_role_name = job_role if pd.notna(job_role) else "(Missing/NaN)"
            print(f"{rank:<6} {str(job_role_name)[:48]:<50} {count:<10}")
    
    # Description Length Analysis
    if description_col:
        print(f"\n📏 DESCRIPTION LENGTH ANALYSIS (column: '{description_col}')")
        print("-" * 80)
        
        # Calculate lengths (handle NaN values)
        df['description_length'] = df[description_col].fillna('').astype(str).apply(len)
        
        lengths = df['description_length']
        
        print(f"Average length: {lengths.mean():.2f} characters")
        print(f"Median length: {lengths.median():.2f} characters")
        print(f"Standard deviation: {lengths.std():.2f} characters")
        print()
        print(f"Minimum length: {lengths.min()} characters")
        print(f"Maximum length: {lengths.max()} characters")
        print(f"Range: {lengths.max() - lengths.min()} characters")
        print()
        
        # Percentiles
        print("Length Percentiles:")
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            value = np.percentile(lengths, p)
            print(f"  {p}th percentile: {value:.0f} characters")
        
        print()
        
        # Length distribution buckets
        print("Length Distribution:")
        bins = [0, 500, 1000, 2000, 5000, 10000, float('inf')]
        labels = ['0-500', '501-1000', '1001-2000', '2001-5000', '5001-10000', '10000+']
        
        df['length_bucket'] = pd.cut(lengths, bins=bins, labels=labels, right=True)
        bucket_counts = df['length_bucket'].value_counts().sort_index()
        
        print(f"{'Range (chars)':<15} {'Count':<10} {'Percentage':<10}")
        print("-" * 40)
        for bucket, count in bucket_counts.items():
            percentage = (count / len(df)) * 100
            print(f"{bucket:<15} {count:<10} {percentage:>6.2f}%")
        
        # Count empty or very short descriptions
        empty_count = (lengths == 0).sum()
        very_short_count = ((lengths > 0) & (lengths < 50)).sum()
        
        if empty_count > 0 or very_short_count > 0:
            print()
            print("Data Quality:")
            if empty_count > 0:
                print(f"  Empty descriptions: {empty_count} ({(empty_count/len(df)*100):.2f}%)")
            if very_short_count > 0:
                print(f"  Very short descriptions (<50 chars): {very_short_count} ({(very_short_count/len(df)*100):.2f}%)")
    else:
        print("\n⚠️  Description column not found in dataset")
        print(f"   Available columns: {', '.join(df.columns.tolist())}")
    
    # Company Distribution
    if 'company' in df.columns:
        print("\n🏢 COMPANY DISTRIBUTION")
        print("-" * 80)
        company_counts = df['company'].value_counts()
        print(f"Total unique companies: {len(company_counts)}")
        
        # Top 5 companies
        print("\nTop 5 Companies by Job Postings:")
        print(f"{'Rank':<6} {'Company':<50} {'Count':<10}")
        print("-" * 80)
        
        for rank, (company, count) in enumerate(company_counts.head(5).items(), 1):
            company_name = company if pd.notna(company) else "(Missing/NaN)"
            print(f"{rank:<6} {str(company_name)[:48]:<50} {count:<10}")
    
    # Missing data analysis
    print("\n❓ MISSING DATA ANALYSIS")
    print("-" * 80)
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
    
    if len(missing_data) > 0:
        print(f"{'Column':<40} {'Missing Count':<15} {'Percentage':<10}")
        print("-" * 80)
        for col, count in missing_data.items():
            percentage = (count / len(df)) * 100
            print(f"{col:<40} {count:<15} {percentage:>6.2f}%")
    else:
        print("✅ No missing data found!")
    
    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze job dataset statistics")
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        default="datasets/8k_raw_jd_data.csv",
        help="Path to the dataset CSV file"
    )
    
    args = parser.parse_args()
    
    try:
        analyze_dataset(args.dataset_path)
    except FileNotFoundError:
        print(f"❌ Error: File not found at '{args.dataset_path}'")
    except Exception as e:
        print(f"❌ Error analyzing dataset: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

