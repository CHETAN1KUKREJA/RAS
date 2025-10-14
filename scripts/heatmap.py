import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from PIL import Image
import argparse

def create_heatmap(data_folder: str, image_path: str, latent_height: int, latent_width: int, patch_size: int, alpha: float):
    """
    Creates a heatmap from token cache counts and overlays it on the final generated image,
    including a color bar legend.

    Args:
        data_folder (str): Path to the folder with 'cache_counter.pt'.
        image_path (str): Path to the final generated image.
        latent_height (int): The height of the latent space.
        latent_width (int): The width of the latent space.
        patch_size (int): The size of each patch (e.g., 2 for 2x2).
        alpha (float): The transparency of the heatmap overlay.
    """
    print("--- Starting Cache Count Heatmap Generation with Color Bar ---")

    # 1. Define and check file paths
    counter_path = os.path.join(data_folder, 'cache_counter.pt')
    if not os.path.exists(counter_path):
        print(f"Error: Could not find 'cache_counter.pt' in '{data_folder}'")
        return
    if not os.path.exists(image_path):
        print(f"Error: Could not find the source image at '{image_path}'")
        return

    # 2. Load the data
    try:
        cache_counter = torch.load(counter_path)
        generated_image = Image.open(image_path).convert("RGB")
        print(f"Successfully loaded data. Found {len(cache_counter)} tokens.")
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # 3. Reshape the 1D counter tensor into a 2D grid
    grid_height = latent_height // patch_size
    grid_width = latent_width // patch_size
    if len(cache_counter) != grid_height * grid_width:
        print(f"Error: Token count ({len(cache_counter)}) does not match latent/patch dimensions ({grid_height}x{grid_width}).")
        return
    
    data_grid = cache_counter.reshape(grid_height, grid_width).numpy()

    # --- IMAGE BLENDING ---
    min_val, max_val = np.min(data_grid), np.max(data_grid)
    normalized_grid = (data_grid - min_val) / (max_val - min_val) if (max_val - min_val) > 0 else np.zeros_like(data_grid)
    colored_heatmap_rgba = cm.viridis(normalized_grid)
    colored_heatmap_rgb = Image.fromarray((colored_heatmap_rgba[:, :, :3] * 255).astype(np.uint8))
    final_image_size = generated_image.size
    scaled_heatmap = colored_heatmap_rgb.resize(final_image_size, Image.NEAREST)
    blended_image = Image.blend(generated_image, scaled_heatmap, alpha=alpha)

    # --- PLOTTING WITH MATPLOTLIB ---
    fig = plt.figure(figsize=(12, 10), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[20, 1])
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(blended_image)
    ax_img.axis('off')
    ax_img.set_title("Token Cache Count Heatmap")
    ax_cbar = fig.add_subplot(gs[0, 1])
    norm = plt.Normalize(vmin=min_val, vmax=max_val)
    sm = cm.ScalarMappable(cmap='viridis', norm=norm)
    cbar = fig.colorbar(sm, cax=ax_cbar)
    cbar.set_label('Number of Times Cached', rotation=270, labelpad=20, fontsize=12)

    output_filename = "token_cache_heatmap_with_bar.png"
    output_path = os.path.join(data_folder, output_filename)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print(f"--- Finished: Saved heatmap with color bar to {output_path} ---")

def create_std_heatmap_at_timestamp(data_folder: str, image_path: str, timestamp: int, latent_height: int, latent_width: int, patch_size: int, alpha: float, output_dir: str = None):
    """Creates a heatmap of standard deviation values for a specific timestamp."""
    if output_dir is None:
        output_dir = data_folder
        print(f"--- Starting STD Heatmap Generation for Timestamp {timestamp} ---")

    history_path = os.path.join(data_folder, 'std_history.pt')
    if not os.path.exists(history_path) or not os.path.exists(image_path):
        if output_dir == data_folder:
             print(f"Error: Could not find required files in '{data_folder}'")
        return

    try:
        stacked_history = torch.load(history_path)
        generated_image = Image.open(image_path).convert("RGB")
    except Exception as e:
        if output_dir == data_folder:
            print(f"Error loading files: {e}")
        return

    num_timesteps = stacked_history.shape[0]
    if not (0 <= timestamp < num_timesteps):
        if output_dir == data_folder:
            print(f"Error: Timestamp {timestamp} is out of bounds (0-{num_timesteps - 1}).")
        return
    
    data_grid = stacked_history[timestamp, :].reshape(latent_height // patch_size, latent_width // patch_size).numpy()
    
    min_val, max_val = np.min(data_grid), np.max(data_grid)
    normalized_grid = (data_grid - min_val) / (max_val - min_val) if (max_val - min_val) > 0 else np.zeros_like(data_grid)
    colored_heatmap_rgba = cm.viridis(normalized_grid)
    colored_heatmap_rgb = Image.fromarray((colored_heatmap_rgba[:, :, :3] * 255).astype(np.uint8))
    
    final_image_size = generated_image.size
    scaled_heatmap = colored_heatmap_rgb.resize(final_image_size, Image.NEAREST)
    blended_image = Image.blend(generated_image, scaled_heatmap, alpha=alpha)

    fig = plt.figure(figsize=(12, 10), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[20, 1])
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(blended_image)
    ax_img.axis('off')
    ax_img.set_title(f"Standard Deviation Heatmap at Timestep {timestamp}")
    ax_cbar = fig.add_subplot(gs[0, 1])
    norm = plt.Normalize(vmin=min_val, vmax=max_val)
    sm = cm.ScalarMappable(cmap='viridis', norm=norm)
    cbar = fig.colorbar(sm, cax=ax_cbar)
    cbar.set_label('Standard Deviation Value', rotation=270, labelpad=20, fontsize=12)

    output_filename = f"std_heatmap_step_{timestamp}.png"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    if output_dir == data_folder:
        print(f"--- Finished: Saved STD heatmap to {output_path} ---")

def create_all_std_heatmaps(data_folder: str, image_path: str, latent_height: int, latent_width: int, patch_size: int, alpha: float):
    """Creates a heatmap for every timestamp and saves them in a sub-folder."""
    print("--- Starting Generation for All STD Heatmaps ---")
    heatmap_dir = os.path.join(data_folder, "std_heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)
    print(f"Saving all heatmaps to: {heatmap_dir}")

    history_path = os.path.join(data_folder, 'std_history.pt')
    if not os.path.exists(history_path):
        print(f"Error: Could not find 'std_history.pt' in '{data_folder}'")
        return
        
    stacked_history = torch.load(history_path)
    num_timesteps = stacked_history.shape[0]

    for i in range(num_timesteps):
        print(f"  Generating STD heatmap for timestamp {i+1}/{num_timesteps}...")
        create_std_heatmap_at_timestamp(data_folder, image_path, i, latent_height, latent_width, patch_size, alpha, output_dir=heatmap_dir)
    
    print("--- Finished generating all STD heatmaps. ---")

def create_mean_heatmap_at_timestamp(data_folder: str, image_path: str, timestamp: int, latent_height: int, latent_width: int, patch_size: int, alpha: float, output_dir: str = None):
    """Creates a heatmap of mean values for a specific timestamp."""
    if output_dir is None:
        output_dir = data_folder
        print(f"--- Starting Mean Heatmap Generation for Timestamp {timestamp} ---")

    history_path = os.path.join(data_folder, 'mean_history.pt')
    if not os.path.exists(history_path) or not os.path.exists(image_path):
        if output_dir == data_folder:
            print(f"Error: Could not find required files in '{data_folder}'")
        return

    try:
        stacked_history = torch.load(history_path)
        generated_image = Image.open(image_path).convert("RGB")
    except Exception as e:
        if output_dir == data_folder:
            print(f"Error loading files: {e}")
        return

    num_timesteps = stacked_history.shape[0]
    if not (0 <= timestamp < num_timesteps):
        if output_dir == data_folder:
            print(f"Error: Timestamp {timestamp} is out of bounds (0-{num_timesteps - 1}).")
        return
    
    data_grid = stacked_history[timestamp, :].reshape(latent_height // patch_size, latent_width // patch_size).numpy()
    
    min_val, max_val = np.min(data_grid), np.max(data_grid)
    normalized_grid = (data_grid - min_val) / (max_val - min_val) if (max_val - min_val) > 0 else np.zeros_like(data_grid)
    colored_heatmap_rgba = cm.viridis(normalized_grid)
    colored_heatmap_rgb = Image.fromarray((colored_heatmap_rgba[:, :, :3] * 255).astype(np.uint8))
    
    final_image_size = generated_image.size
    scaled_heatmap = colored_heatmap_rgb.resize(final_image_size, Image.NEAREST)
    blended_image = Image.blend(generated_image, scaled_heatmap, alpha=alpha)

    fig = plt.figure(figsize=(12, 10), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[20, 1])
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(blended_image)
    ax_img.axis('off')
    ax_img.set_title(f"Mean Value Heatmap at Timestep {timestamp}")
    ax_cbar = fig.add_subplot(gs[0, 1])
    norm = plt.Normalize(vmin=min_val, vmax=max_val)
    sm = cm.ScalarMappable(cmap='viridis', norm=norm)
    cbar = fig.colorbar(sm, cax=ax_cbar)
    cbar.set_label('Mean Value', rotation=270, labelpad=20, fontsize=12)

    output_filename = f"mean_heatmap_step_{timestamp}.png"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    if output_dir == data_folder:
        print(f"--- Finished: Saved Mean heatmap to {output_path} ---")

def create_all_mean_heatmaps(data_folder: str, image_path: str, latent_height: int, latent_width: int, patch_size: int, alpha: float):
    """Creates a mean heatmap for every timestamp and saves them in a sub-folder."""
    print("--- Starting Generation for All Mean Heatmaps ---")
    heatmap_dir = os.path.join(data_folder, "mean_heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)
    print(f"Saving all heatmaps to: {heatmap_dir}")

    history_path = os.path.join(data_folder, 'mean_history.pt')
    if not os.path.exists(history_path):
        print(f"Error: Could not find 'mean_history.pt' in '{data_folder}'")
        return
        
    stacked_history = torch.load(history_path)
    num_timesteps = stacked_history.shape[0]

    for i in range(num_timesteps):
        print(f"  Generating Mean heatmap for timestamp {i+1}/{num_timesteps}...")
        create_mean_heatmap_at_timestamp(data_folder, image_path, i, latent_height, latent_width, patch_size, alpha, output_dir=heatmap_dir)
    
    print("--- Finished generating all Mean heatmaps. ---")

# --- MODIFIED FUNCTIONS FOR DOT PRODUCT HEATMAP ---
def create_dot_heatmap_at_timestamp(data_folder: str, image_path: str, timestamp: int, latent_height: int, latent_width: int, patch_size: int, alpha: float, output_dir: str = None):
    """Creates a heatmap of dot product values for a specific timestamp."""
    if output_dir is None:
        output_dir = data_folder
        print(f"--- Starting Dot Product Heatmap Generation for Timestamp {timestamp} ---")

    history_path = os.path.join(data_folder, 'dot_history.pt')
    if not os.path.exists(history_path) or not os.path.exists(image_path):
        if output_dir == data_folder:
            print(f"Error: Could not find required files in '{data_folder}'")
        return

    try:
        stacked_history = torch.load(history_path)
        generated_image = Image.open(image_path).convert("RGB")
    except Exception as e:
        if output_dir == data_folder:
            print(f"Error loading files: {e}")
        return

    num_timesteps = stacked_history.shape[0]
    if not (0 <= timestamp < num_timesteps):
        if output_dir == data_folder:
            print(f"Error: Timestamp {timestamp} is out of bounds (0-{num_timesteps - 1}).")
        return
    
    data_grid = stacked_history[timestamp, :].reshape(latent_height // patch_size, latent_width // patch_size).numpy()
    
    # --- CHANGED: Use dynamic normalization based on the data's own range ---
    min_val, max_val = np.min(data_grid), np.max(data_grid)
    norm = plt.Normalize(vmin=min_val, vmax=max_val)
    # Using 'viridis' which is a good default for sequential data.
    # 'coolwarm' is also a good option if you expect both positive and negative values.
    colormap = 'viridis' 
    cbar_label = 'Dot Product Value'

    # Normalize data from 0-1 for blending, but use the real norm for the color bar
    normalized_grid_for_blending = (data_grid - min_val) / (max_val - min_val) if (max_val - min_val) > 0 else np.zeros_like(data_grid)
    colored_heatmap_rgba = cm.get_cmap(colormap)(normalized_grid_for_blending)
    colored_heatmap_rgb = Image.fromarray((colored_heatmap_rgba[:, :, :3] * 255).astype(np.uint8))
    
    final_image_size = generated_image.size
    scaled_heatmap = colored_heatmap_rgb.resize(final_image_size, Image.NEAREST)
    blended_image = Image.blend(generated_image, scaled_heatmap, alpha=alpha)

    fig = plt.figure(figsize=(12, 10), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[20, 1])
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(blended_image)
    ax_img.axis('off')
    ax_img.set_title(f"Dot Product Heatmap at Timestep {timestamp}")
    ax_cbar = fig.add_subplot(gs[0, 1])
    
    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    cbar = fig.colorbar(sm, cax=ax_cbar)
    cbar.set_label(cbar_label, rotation=270, labelpad=20, fontsize=12)

    output_filename = f"dot_heatmap_step_{timestamp}.png"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    if output_dir == data_folder:
        print(f"--- Finished: Saved Dot Product heatmap to {output_path} ---")

def create_all_dot_heatmaps(data_folder: str, image_path: str, latent_height: int, latent_width: int, patch_size: int, alpha: float):
    """Creates a dot product heatmap for every timestamp and saves them in a sub-folder."""
    print("--- Starting Generation for All Dot Product Heatmaps ---")
    heatmap_dir = os.path.join(data_folder, "dot_heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)
    print(f"Saving all heatmaps to: {heatmap_dir}")

    history_path = os.path.join(data_folder, 'dot_history.pt')
    if not os.path.exists(history_path):
        print(f"Error: Could not find 'dot_history.pt' in '{data_folder}'")
        return
        
    stacked_history = torch.load(history_path)
    num_timesteps = stacked_history.shape[0]

    for i in range(num_timesteps):
        print(f"  Generating Dot Product heatmap for timestamp {i+1}/{num_timesteps}...")
        create_dot_heatmap_at_timestamp(data_folder, image_path, i, latent_height, latent_width, patch_size, alpha, output_dir=heatmap_dir)
    
    print("--- Finished generating all Dot Product heatmaps. ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate heatmaps from analysis data.")
    parser.add_argument("data_folder", type=str, help="The path to the folder containing analysis data.")
    parser.add_argument("image_path", type=str, help="The path to the final generated image.")
    parser.add_argument("--latent_height", type=int, default=128, help="Height of the latent space.")
    parser.add_argument("--latent_width", type=int, default=128, help="Width of the latent space.")
    parser.add_argument("--patch_size", type=int, default=2, help="The size of the patches.")
    parser.add_argument("--alpha", type=float, default=0.7, help="Transparency of the heatmap overlay.")
    parser.add_argument(
        "--std_heatmap", 
        nargs='?', 
        const='all', 
        default=None, 
        metavar="TIMESTAMP", 
        help="Generate STD heatmaps. Provide a timestamp for a single map, or no value for all."
    )
    parser.add_argument(
        "--mean_heatmap", 
        nargs='?', 
        const='all', 
        default=None, 
        metavar="TIMESTAMP", 
        help="Generate Mean heatmaps. Provide a timestamp for a single map, or no value for all."
    )
    parser.add_argument(
        "--dot_heatmap", 
        nargs='?', 
        const='all', 
        default=None, 
        metavar="TIMESTAMP", 
        help="Generate Dot Product heatmaps. Provide a timestamp for a single map, or no value for all."
    )
    
    args = parser.parse_args()

    action_taken = False
    if args.std_heatmap is not None:
        action_taken = True
        if args.std_heatmap == 'all':
            create_all_std_heatmaps(args.data_folder, args.image_path, args.latent_height, args.latent_width, args.patch_size, args.alpha)
        else:
            try:
                timestamp = int(args.std_heatmap)
                create_std_heatmap_at_timestamp(args.data_folder, args.image_path, timestamp, args.latent_height, args.latent_width, args.patch_size, args.alpha)
            except ValueError:
                print(f"Error: Invalid timestamp for --std_heatmap '{args.std_heatmap}'.")

    if args.mean_heatmap is not None:
        action_taken = True
        if args.mean_heatmap == 'all':
            create_all_mean_heatmaps(args.data_folder, args.image_path, args.latent_height, args.latent_width, args.patch_size, args.alpha)
        else:
            try:
                timestamp = int(args.mean_heatmap)
                create_mean_heatmap_at_timestamp(args.data_folder, args.image_path, timestamp, args.latent_height, args.latent_width, args.patch_size, args.alpha)
            except ValueError:
                print(f"Error: Invalid timestamp for --mean_heatmap '{args.mean_heatmap}'.")
    
    if args.dot_heatmap is not None:
        action_taken = True
        if args.dot_heatmap == 'all':
            create_all_dot_heatmaps(args.data_folder, args.image_path, args.latent_height, args.latent_width, args.patch_size, args.alpha)
        else:
            try:
                timestamp = int(args.dot_heatmap)
                create_dot_heatmap_at_timestamp(args.data_folder, args.image_path, timestamp, args.latent_height, args.latent_width, args.patch_size, args.alpha)
            except ValueError:
                print(f"Error: Invalid timestamp for --dot_heatmap '{args.dot_heatmap}'.")
    
    if not action_taken:
        # Default behavior: run the cache count heatmap if no other action is specified
        create_heatmap(args.data_folder, args.image_path, args.latent_height, args.latent_width, args.patch_size, args.alpha)

# --- How to Run ---
#
# To get the DOT PRODUCT heatmap for a SINGLE step (e.g., step 10):
# python <your_script_name>.py std_analysis_... final_image.png --dot_heatmap 10
#
# To get DOT PRODUCT heatmaps for ALL steps:
# python <your_script_name>.py std_analysis_... final_image.png --dot_heatmap
#
# To get the MEAN heatmap for a SINGLE step (e.g., step 5):
# python <your_script_name>.py std_analysis_... final_image.png --mean_heatmap 5
#
# To get the STD heatmap for ALL steps:
# python <your_script_name>.py std_analysis_... final_image.png --std_heatmap
# ------------------