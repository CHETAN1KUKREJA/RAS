import json

annotation_file = '/DATA/chetan/coco/annotations/captions_val2014.json'

print(f"Loading annotation file: {annotation_file}...")

# --- 2. READ THE JSON FILE ---
with open(annotation_file, 'r') as f:
    data = json.load(f)

print("File loaded successfully.")
print("-" * 30)

# --- 3. PRINT THE JSON STRUCTURE ---
print("### JSON Top-Level Structure ###")
print(f"The file is a dictionary with these keys: {list(data.keys())}")
print(f"  > 'info': Contains {len(data['info'])} metadata fields.")
print(f"  > 'licenses': Contains {len(data['licenses'])} license types.")
print(f"  > 'images': Contains a list of {len(data['images'])} image entries.")
print(f"  > 'annotations': Contains a list of {len(data['annotations'])} caption entries.")

print("-" * 30)

# --- 4. PRINT CAPTIONS FOR ONE IMAGE ---
print("### Example: Captions for one image ###")

# Get the first image from the 'images' list
first_image = data['images'][0]
image_id = first_image['id']
image_filename = first_image['file_name']

print(f"Finding captions for the first image in the list:")
print(f"  > Filename: {image_filename}")
print(f"  > Image ID: {image_id}\n")

# Find all annotations (captions) that match this image_id
# We can do this efficiently with a list comprehension
image_captions = [ann['caption'] for ann in data['annotations'] if ann['image_id'] == image_id]

print(f"Found {len(image_captions)} captions:")
# Print each caption
for i, caption in enumerate(image_captions):
    print(f"  {i+1}: {caption}")