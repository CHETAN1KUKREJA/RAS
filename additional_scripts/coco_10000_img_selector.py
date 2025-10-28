import json
import csv
import random
from collections import defaultdict
from tqdm import tqdm # For a helpful progress bar

# --- Configuration ---
# 1. Change this to your COCO annotation file
ANNOTATION_FILE = '/DATA/chetan/coco/annotations/captions_val2014.json' 
# 2. This will be the name of your output file
OUTPUT_CSV = '/DATA/chetan/RAS/additional_scripts/longest_captions_sample.csv'
# 3. The number of random images you want
NUM_SAMPLES = 10000 

print(f"Loading annotation file: {ANNOTATION_FILE}...")
# Load the annotation file
with open(ANNOTATION_FILE, 'r') as f:
    data = json.load(f)

# --- Step 1: Pre-process annotations for fast lookup ---
print("Pre-processing captions... (This makes things much faster)")
# Create a dictionary to map each image_id to a list of its captions
captions_by_image_id = defaultdict(list)
for ann in tqdm(data['annotations']):
    captions_by_image_id[ann['image_id']].append(ann['caption'])

# Get the list of all image objects
all_images = data['images']
total_images = len(all_images)
print(f"Found {total_images} total images.")

# --- Step 2: Randomly sample 10,000 images ---
# Ensure we don't try to sample more than we have
if NUM_SAMPLES > total_images:
    print(f"Warning: Requested {NUM_SAMPLES} but only {total_images} are available.")
    NUM_SAMPLES = total_images

print(f"Randomly selecting {NUM_SAMPLES} images...")
selected_images = random.sample(all_images, NUM_SAMPLES)

# --- Step 3: Find longest caption and write to CSV ---
print(f"Processing {NUM_SAMPLES} images to find the longest caption for each...")
processed_count = 0

with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
    # Create a CSV writer
    writer = csv.writer(f)
    
    # Write the header row
    writer.writerow(['file_name', 'longest_caption'])
    
    # Loop over our 10,000 random images with a progress bar
    for image in tqdm(selected_images):
        image_id = image['id']
        file_name = image['file_name']
        
        # Get all captions for this image_id
        captions = captions_by_image_id.get(image_id)
        
        # Skip if, for some reason, an image has no captions
        if not captions:
            continue
            
        # Find the longest caption in the list
        # max(list, key=len) is a fast way to find the longest string
        longest_caption = max(captions, key=len)
        
        # Write the image's file_name and its longest caption to the CSV
        writer.writerow([file_name, longest_caption])
        processed_count += 1

print("\n--- All Done! ---")
print(f"Successfully processed {processed_count} images.")
print(f"CSV file saved as: {OUTPUT_CSV}")