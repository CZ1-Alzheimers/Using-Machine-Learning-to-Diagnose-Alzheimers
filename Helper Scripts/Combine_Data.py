import pandas as pd

# Define the paths to input CSV files
file1_path = 'Datasets/ADNI_complete_dataset.csv'
file2_path = 'Datasets/Original Datasets/AIBL_dataset.csv'
output_file_path = 'Datasets/ADNI-AIBL_dataset.csv'

try:
    # Read the CSV files into pandas DataFrames
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)

    # Concatenate the DataFrames. Pandas will align columns by name.
    merged_df = pd.concat([df1, df2], ignore_index=True)

    # Write the merged DataFrame to a new CSV file
    merged_df.to_csv(output_file_path, index=False)

    print(f"Successfully merged '{file1_path}' and '{file2_path}' into '{output_file_path}'")

except FileNotFoundError:
    print("Error: One or both input files not found. Please check the file paths.")
except Exception as e:
    print(f"An error occurred: {e}")