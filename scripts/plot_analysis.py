import os
import torch
import matplotlib.pyplot as plt
import argparse
import numpy as np

def plot_token_histograms(data_folder: str):
    """
    Loads analysis data from a specified folder and generates individual histogram plots
    for each token's standard deviation history.

    Args:
        data_folder (str): The path to the folder containing the analysis data.
    """
    print(f"--- Starting Individual Token Analysis for Folder: {data_folder} ---")

    # 1. Define file paths
    history_path = os.path.join(data_folder, 'std_history.pt')
    counter_path = os.path.join(data_folder, 'cache_counter.pt')

    # 2. Check if files exist
    if not os.path.exists(history_path) or not os.path.exists(counter_path):
        print(f"Error: Could not find 'std_history.pt' or 'cache_counter.pt' in '{data_folder}'")
        return

    # 3. Load the data
    try:
        stacked_history = torch.load(history_path)
        cache_counter = torch.load(counter_path)
        print(f"Successfully loaded data. Tensor shape: {stacked_history.shape}")
    except Exception as e:
        print(f"Error loading tensor files: {e}")
        return

    # Create a dedicated folder for the output images
    images_folder = os.path.join(data_folder, 'images')
    os.makedirs(images_folder, exist_ok=True)

    # 4. Iterate through each token, plot, and save its history
    num_tokens = stacked_history.shape[1]
    for token_idx in range(num_tokens):
        token_std_data = stacked_history[:, token_idx].numpy()
        cache_count = int(cache_counter[token_idx].item())

        # Create plot
        plt.figure(figsize=(8, 6))
        plt.hist(token_std_data, bins='auto', color='skyblue', edgecolor='black')
        plt.title(f'STD History for Token {token_idx}\n(Cached {cache_count} times)')
        plt.xlabel('Standard Deviation Value')
        plt.ylabel('Frequency')
        plt.grid(True, linestyle='--', alpha=0.6)

        # Save plot
        plot_filename = f"{cache_count}_{token_idx}.png"
        plot_path = os.path.join(images_folder, plot_filename)
        plt.savefig(plot_path)
        plt.close() # Close the figure to free up memory

    print(f"--- Finished: Saved {num_tokens} token plots to {images_folder} ---")


def plot_grouped_histograms(data_folder: str, threshold: int):
    """
    Loads analysis data, groups tokens based on a cache count threshold,
    and plots a single combined, normalized histogram comparing the two groups.

    Args:
        data_folder (str): The path to the folder containing the analysis data.
        threshold (int): The cache count to separate low-cached from high-cached tokens.
    """
    print(f"--- Starting Grouped Analysis for Folder: {data_folder} (Threshold: {threshold}) ---")

    # 1. Define and check file paths
    history_path = os.path.join(data_folder, 'std_history.pt')
    counter_path = os.path.join(data_folder, 'cache_counter.pt')
    if not os.path.exists(history_path) or not os.path.exists(counter_path):
        print(f"Error: Could not find 'std_history.pt' or 'cache_counter.pt' in '{data_folder}'")
        return

    # 2. Load the data
    try:
        stacked_history = torch.load(history_path)
        cache_counter = torch.load(counter_path)
        print(f"Successfully loaded data. Tensor shape: {stacked_history.shape}")
    except Exception as e:
        print(f"Error loading tensor files: {e}")
        return

    # 3. Group the standard deviation data based on the threshold
    foreground_stds = [] # Less cached -> More active updates -> Foreground
    background_stds = [] # More cached -> Less active updates -> Background

    num_tokens = stacked_history.shape[1]
    for token_idx in range(num_tokens):
        cache_count = int(cache_counter[token_idx].item())
        token_data = stacked_history[:, token_idx].numpy()
        if cache_count < threshold:
            foreground_stds.extend(token_data)
        else:
            background_stds.extend(token_data)
    
    print(f"Grouped tokens: {len(foreground_stds)} data points in foreground group, {len(background_stds)} in background group.")

    # 4. Create the combined plot
    plt.figure(figsize=(12, 8))
    
    # Plot normalized histograms (density plots) for both groups on the same axes
    plt.hist(foreground_stds, bins='auto', color='coral', alpha=0.7, label=f'Cached < {threshold} times (Foreground Proxy)', edgecolor='black', density=True)
    plt.hist(background_stds, bins='auto', color='royalblue', alpha=0.7, label=f'Cached >= {threshold} times (Background Proxy)', edgecolor='black', density=True)
    
    plt.title('Normalized Comparison of STD Distributions for Foreground vs. Background Tokens')
    plt.xlabel('Standard Deviation Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # 5. Save the final plot
    plot_filename = f"grouped_normalized_histogram_thresh_{threshold}.png"
    plot_path = os.path.join(data_folder, plot_filename)
    plt.savefig(plot_path)
    plt.close()

    print(f"--- Finished: Saved grouped histogram to {plot_path} ---")


def plot_cache_counter_histogram(data_folder: str):
    """
    Loads the cache counter data and plots a histogram showing the
    distribution of cache counts across all tokens.

    Args:
        data_folder (str): The path to the folder containing the analysis data.
    """
    print(f"--- Starting Cache Counter Analysis for Folder: {data_folder} ---")

    # 1. Define and check file path
    counter_path = os.path.join(data_folder, 'cache_counter.pt')
    if not os.path.exists(counter_path):
        print(f"Error: Could not find 'cache_counter.pt' in '{data_folder}'")
        return

    # 2. Load the data
    try:
        cache_counter = torch.load(counter_path).numpy()
        print(f"Successfully loaded cache counter data. Number of tokens: {len(cache_counter)}")
    except Exception as e:
        print(f"Error loading tensor file: {e}")
        return

    # 3. Create and save the plot
    plt.figure(figsize=(12, 8))
    plt.hist(cache_counter, bins=max(1, int(np.max(cache_counter)) - int(np.min(cache_counter))), color='green', alpha=0.7, edgecolor='black')
    plt.title('Distribution of Cache Counts Across All Tokens')
    plt.xlabel('Number of Times a Token was Cached')
    plt.ylabel('Number of Tokens')
    plt.grid(True, linestyle='--', alpha=0.6)

    plot_filename = "cache_counter_histogram.png"
    plot_path = os.path.join(data_folder, plot_filename)
    plt.savefig(plot_path)
    plt.close()

    print(f"--- Finished: Saved cache counter histogram to {plot_path} ---")


if __name__ == "__main__":
    # --- Command-Line Interface Setup ---
    parser = argparse.ArgumentParser(description="Generate histograms for token STD history from a data folder.")
    parser.add_argument(
        "data_folder",
        type=str,
        help="The path to the folder containing 'std_history.pt' and 'cache_counter.pt'."
    )
    parser.add_argument(
        "--grouped",
        action="store_true",
        help="Plot a single grouped histogram of STD values."
    )
    parser.add_argument(
        '--threshold',
        type=int,
        default=20,
        help='The cache count threshold for grouping tokens. Used with --grouped.'
    )
    parser.add_argument(
        "--counter_hist",
        action="store_true",
        help="Plot a histogram of the cache counts for all tokens."
    )
    args = parser.parse_args()
    
    # --- Logic to call the correct function based on arguments ---
    if args.grouped:
        plot_grouped_histograms(args.data_folder, args.threshold)
    elif args.counter_hist:
        plot_cache_counter_histogram(args.data_folder)
    else:
        plot_token_histograms(args.data_folder)

# --- How to Run ---
#
# 1. To generate individual plots for every token (default behavior):
#    python plot_analysis.py path/to/your/data_folder
#
# 2. To generate a single, grouped histogram plot with the default threshold of 20:
#    python plot_analysis.py path/to/your/data_folder --grouped
#
# 3. To generate a grouped plot with a custom threshold of 15:
#    python plot_analysis.py path/to/your/data_folder --grouped --threshold 15
# 4. To generate a histogram of the cache counts:
#    python plot_analysis.py path/to/your/data_folder --counter_hist

# ------------------

