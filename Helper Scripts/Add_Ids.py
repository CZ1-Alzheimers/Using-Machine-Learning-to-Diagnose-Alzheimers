import pandas as pd
import os

# UNCOMMENT SECTION YOU WANT TO RUN

# # ------THIS SECTON FOR ADNI -----------
# Define the file names
csv_with_image_ids = 'Datasets/Label CSVs/Sample_Test_Data.csv'
csv_with_subject_ids = 'Datasets/Label CSVs/mri_common.csv'
output_csv_file = 'Datasets/Sample_Test_Data.csv'

# Read the CSV files into pandas DataFrames
df_images = pd.read_csv(csv_with_image_ids)
df_subjects = pd.read_csv(csv_with_subject_ids)

# Merge the two DataFrames on the common column 'Image_ID'
merged_df = pd.merge(df_images, df_subjects[['Image_ID', 'Subject']], on='Image_ID', how='left')

# Save the merged DataFrame to a new CSV file
merged_df.to_csv(output_csv_file, index=False)

print(f"\nSuccessfully merged data and saved to {output_csv_file}")

# ------THIS SECTON FOR OASIS3 -----------
# # Load dataset
# df = pd.read_csv("Datasets/Oasis3_dataset.csv")
#
# # Create ImageID column
# df['Image_ID'] = df.reset_index().index + 1  # simple running number starting from 1
# df['Image_ID'] = df['Image_ID'].apply(lambda x: f"MR_d{x:04d}")  # format as MR_d0001, MR_d0002, etc.
#
# # Move 'ImageID' to the front
# df.insert(0, 'Image_ID', df.pop('Image_ID'))
#
# # Save updated file
# df.to_csv("Datasets/Oasis3_complete_dataset.csv", index=False)