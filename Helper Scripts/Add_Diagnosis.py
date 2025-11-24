import pandas as pd

# Define the file names
csv_with_image_ids = 'Datasets/Label CSVs/Sample_Test_Data.csv'
csv_with_subject_ids = 'Datasets/Label CSVs/mri_common.csv'
output_csv_file = 'Datasets/Sample_Test_Data.csv'

# Read the CSV files into pandas DataFrames
df_images = pd.read_csv(csv_with_image_ids)
df_subjects = pd.read_csv(csv_with_subject_ids)

# Merge on Image_ID
merged_df = pd.merge(
    df_images,
    df_subjects[['Image_ID', 'Diagnosis']],
    on='Image_ID',
    how='left'
)

# Convert CN / MCI / AD → 1 / 2 / 3
group_map = {'CN': 1, 'MCI': 2, 'AD': 3}
merged_df['Diagnosis'] = merged_df['Diagnosis'].map(group_map)

# Save
merged_df.to_csv(output_csv_file, index=False)

print(f"\nSuccessfully merged data and saved to {output_csv_file}")
