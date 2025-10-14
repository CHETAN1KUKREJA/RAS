# Calculates the CLIP score for images in a folder against the Parti Prompts dataset.
# usage: python CLIP.py path/to/my_images --batch_size 32 --model_name ViT-g-14 --pretrained laion2b_s34b_b88k

# pip install torch torchvision Pillow tqdm open-clip-torch datasets huggingface-hub
import os
import argparse
import re
from PIL import Image
import torch
from tqdm import tqdm
import open_clip
from datasets import load_dataset

def calculate_clip_score(image_folder, batch_size, model_name, pretrained):
    """
    Calculates the CLIP score for images in a folder against the Parti Prompts dataset.
    
    Args:
        image_folder (str): Path to the folder with generated images.
        batch_size (int): The batch size for processing.
        model_name (str): The CLIP model architecture (e.g., 'ViT-g-14').
        pretrained (str): The pretrained dataset name for the model (e.g., 'laion2b_s34b_b88k').
    """
    # 1. Set up device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU. This will be very slow.")
    print(f"Using device: {device}")

    # 2. Load CLIP Model, Preprocessor, and Tokenizer
    print(f"Loading CLIP model: {model_name}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval() # Set model to evaluation mode
    print("Model loaded.")

    # 3. Load Parti Prompts dataset from Hugging Face
    print("Downloading Parti Prompts dataset...")
    try:
        # We use the training set as it contains the prompts.
            parti_prompts_dataset = load_dataset("nateraw/parti-prompts", split="train")
            prompts_list = parti_prompts_dataset['Prompt']
            print("first 5 prompts are",prompts_list[0:5])
            print(f"Dataset loaded with {len(prompts_list)} prompts.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # 4. Find images and map them to prompts
    image_files = sorted(
        [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
        key=lambda x: int(re.search(r'(\d+)', x).group()) # Sort numerically by the number in the filename
    )
    
    if not image_files:
        print(f"Error: No images found in '{image_folder}'. Please check the path.")
        return

    print(f"Found {len(image_files)} images to evaluate.")
    
    all_prompts = []
    valid_image_paths = []

    for img_file in image_files:
        match = re.search(r'(\d+)', img_file)
        if match:
            img_index = int(match.group(1))
            if 0 <= img_index < len(prompts_list):
                all_prompts.append(prompts_list[img_index])
                valid_image_paths.append(os.path.join(image_folder, img_file))
            else:
                print(f"Warning: Index {img_index} from '{img_file}' is out of range for prompts list. Skipping.")
        else:
            print(f"Warning: Could not extract a number index from filename '{img_file}'. Skipping.")

    if not valid_image_paths:
        print("Error: No valid image-prompt pairs could be created. Exiting.")
        return

    # 5. Process in batches to calculate scores
    all_scores = []
    num_batches = (len(valid_image_paths) + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Calculating CLIP Scores"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(valid_image_paths))
            
            image_batch_paths = valid_image_paths[start_idx:end_idx]
            prompt_batch = all_prompts[start_idx:end_idx]
            
            # Preprocess images
            images = [Image.open(p).convert("RGB") for p in image_batch_paths]
            image_tensors = torch.stack([preprocess(img) for img in images]).to(device)
            
            # Tokenize text
            text_tokens = tokenizer(prompt_batch).to(device)
            
            # Encode images and text
            image_features = model.encode_image(image_tensors)
            text_features = model.encode_text(text_tokens)
            
            # Normalize features
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            # Calculate scores (cosine similarity * 100)
            score = 100.0 * (image_features * text_features).sum(axis=-1)
            all_scores.append(score)

    # 6. Compute and print the final average score
    final_scores_tensor = torch.cat(all_scores).cpu()
    average_score = final_scores_tensor.mean().item()

    print("\n" + "---" * 10)
    print(f"✅ CLIP Score Evaluation Complete")
    print(f"  Folder: {os.path.abspath(image_folder)}")
    print(f"  Model: {model_name} ({pretrained})")
    print(f"  Images Evaluated: {len(final_scores_tensor)}")
    print(f"  Average CLIP Score: {average_score:.4f}")
    print("---" * 10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate CLIP score for a folder of images against Parti Prompts.")
    parser.add_argument("image_folder", type=str, help="Path to the folder containing generated images (e.g., 'path/to/my_images').")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing.")
    parser.add_argument("--model_name", type=str, default="ViT-g-14", help="Name of the CLIP model from open_clip.")
    parser.add_argument("--pretrained", type=str, default="laion2b_s34b_b88k", help="Pretrained weights for the CLIP model.")
    args = parser.parse_args()
    
    calculate_clip_score(args.image_folder, args.batch_size, args.model_name, args.pretrained)