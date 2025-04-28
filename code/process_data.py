import pandas as pd
import os
import csv

# Load parquet file
df = pd.read_parquet('./data/dataset_cropped.parquet')
'''
# Show schema (column names and types)
print(df.dtypes)
# Print just the first item in the image column
print("\nFirst item in the image column:")
print((df['image'].iloc[0]).keys()) # bytes and path
'''

# Create directory if it doesn't exist
os.makedirs('./data/DM', exist_ok=True)

# Create and write to CSV file with proper handling
csv_path = './data/DM_instructions.csv'
# Write CSV with headers first
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'scenario', 'instruction', 'image_path'])  # Write headers

    # extract the image bytes and store to local folder
    # create a csv file to store the scenario, instruction and image path
    for i in range(len(df)):
        scenario = df['scenario'].iloc[i]
        instruction = df['instruction'].iloc[i]
        image_bytes = df['image'].iloc[i]['bytes']
        image_path = f'./data/DM/{i}.jpg'
        
        # Save the image
        with open(image_path, 'wb') as img_file:
            img_file.write(image_bytes)
        
        # Write to CSV using csv writer to handle escaping
        writer.writerow([i, scenario, instruction, image_path])

print(f"Processing complete. Saved {len(df)} images and created CSV file at {csv_path}")


