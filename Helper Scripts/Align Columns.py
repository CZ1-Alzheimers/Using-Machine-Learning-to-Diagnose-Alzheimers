import os

import pandas as pd

def align_columns(file1_path, file2_path, output1_path, output2_path):

    try:
        df1 = pd.read_csv(file1_path)
        df2 = pd.read_csv(file2_path)
    except FileNotFoundError as e:
        print(f"Error: One or both files not found. {e}")
        return

    # Get all unique column names from both DataFrames
    all_columns = sorted(list(set(df1.columns) | set(df2.columns)))

    # Reindex both DataFrames to the match column order
    # Missing columns will be added with NaN values
    df1_harmonized = df1.reindex(columns=all_columns)
    df2_harmonized = df2.reindex(columns=all_columns)

    # Save the harmonized DataFrames to new CSV files
    df1_harmonized.to_csv(output1_path, index=False)
    df2_harmonized.to_csv(output2_path, index=False)

    print(f"Column order aligned. Updated files saved to '{output1_path}' and '{output2_path}'.")

# Main
os.makedirs(".venv/Scripts/Datasets/Aligned Datasets", exist_ok=True)
file1 = 'Datasets/ADNI_complete_dataset.csv'
file2 = 'Datasets/Oasis3_complete_dataset.csv'
output_file1 = 'Datasets/Aligned Datasets/ADNI_aligned_data.csv'
output_file2 = 'Datasets/Aligned Datasets/Oasis3_aligned_data.csv'

align_columns(file1, file2, output_file1, output_file2)