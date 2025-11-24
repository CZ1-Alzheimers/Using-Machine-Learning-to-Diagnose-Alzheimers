import pandas as pd
import os


def reorder_csv_columns(reference_filepath, target_filepath, output_filepath):
    try:
        # Load the dataframes
        df_ref = pd.read_csv(reference_filepath)
        df_target = pd.read_csv(target_filepath)

        # Get the list of columns from the reference file
        reference_columns = df_ref.columns.tolist()
        print(f"Reference file columns (order to use): {reference_columns}")

        target_columns = df_target.columns.tolist()

        common_ordered_columns = [col for col in reference_columns if col in target_columns]

        if not common_ordered_columns:
            print("\nError: No common columns found between the two files. Aborting.")
            return

        print(f"Target file columns before reordering: {target_columns}")
        print(f"Columns used for reordering: {common_ordered_columns}")

        # 4. Reorder the DataFrame
        df_reordered = df_target[common_ordered_columns]

        # Save the reordered DataFrame
        df_reordered.to_csv(output_filepath, index=False)

        print("\n--- Success ---")
        print(f"Columns of '{target_filepath}' have been reordered and saved to '{output_filepath}'.")
        print(f"Final columns in '{output_filepath}': {df_reordered.columns.tolist()}")

        # Print dropped columns if any
        dropped_columns = [col for col in target_columns if col not in reference_columns]
        if dropped_columns:
            print(
                f"Note: The following columns were dropped because they were not present in the reference file: {dropped_columns}")

    except FileNotFoundError as e:
        print(f"\nError: One of the required files was not found. {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")


if __name__ == "__main__":

    ref_file = 'Datasets/ADNI-Oasis_dataset.csv'  # File with the desired column order (File 1)
    target_file = 'Datasets/Original Datasets/AIBL_dataset.csv'  # File to be reordered (File 2)
    output_file = 'Datasets/Aligned Datasets/AIBL_aligned_data.csv'  # The resulting reordered file

    # Run the column reordering function
    reorder_csv_columns(ref_file, target_file, output_file)