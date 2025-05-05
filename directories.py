import os
import csv

# Specify the dataset folder path
dataset_folder = 'dataset'

# Collect file directories
file_directories = []

for root, dirs, files in os.walk(dataset_folder):
    for file in files:
        file_directories.append(os.path.join(root, file))

# Write the file directories to the CSV file
output_csv = 'file_directories.csv'

with open(output_csv, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['File Directory'])  # Header
    for file_path in sorted(file_directories):
        writer.writerow([file_path])

print(f"File directories have been written to {output_csv}.")
