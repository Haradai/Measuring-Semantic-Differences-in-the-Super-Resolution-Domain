import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import matplotlib.image as mpimg
import torch

def plotsave_similarity_matrix(labels, similarity_matrix, out_folder:str = "plots", show=False):
    """
    Plot a similarity matrix as a heatmap with given labels.
    
    Parameters:
    -----------
    row : pandas.Series or dict
        Row of data corresponding to the current sample
    labels : list
        List of labels for the x and y axes
    similarity_matrix : numpy.ndarray
        2D numpy array representing the similarity matrix
    
    Returns:
    --------
    None
        Displays the heatmap plot
    """
    # Create a new figure with appropriate size
    plt.figure(figsize=(10, 8))
    
    # Use seaborn to create a heatmap
    # annot=True will show the actual values in each cell
    # cmap='YlGnBu' is a color palette that works well for similarity matrices
    sns.heatmap(similarity_matrix, 
                xticklabels=labels, 
                yticklabels=labels, 
                annot=True, 
                cmap='YlGnBu', 
                cbar_kws={'label': 'Similarity'})
    
    # Set title using the row information if available
    title = "Similarity Matrix"

    plt.title(title)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout to prevent cutting off labels
    plt.tight_layout()
    
    # Show the plot
    os.makedirs(out_folder, exist_ok=True)
    plt.savefig(f"{out_folder}/sim_matrix.png", dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


def plotsave_image_matrix(img_paths, labels, out_folder:str = "plots", show=False):
    """
    Plot images in a grid format with labels.
    
    Parameters:
    -----------
    img_paths: list of str

    labels : list of str
        List of labels for each image.
    grid_shape : tuple of int, optional
        Number of rows and columns for the grid. If not provided, a square-like layout is used.
    
    Returns:
    --------
    None
        Displays the image grid.
    """
    # Load images
    images = [Image.open(pth) for pth in img_paths]

    # Ensure labels match the number of images
    if len(labels) != len(images):
        raise ValueError("Number of labels must match the number of images")

    # Determine grid shape if not provided
    num_images = len(images)
    rows = cols = int(np.ceil(np.sqrt(num_images)))


    # Create the figure and axes
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()

    # Plot each image in the grid
    for i, ax in enumerate(axes):
        if i < num_images:
            ax.imshow(images[i])
            ax.axis("off")  # Hide axes for a cleaner look
            ax.set_title(labels[i], fontsize=10, pad=5)  # Add label below the image
        else:
            ax.axis("off")  # Hide extra subplots

    plt.tight_layout()
    # Show the plot
    os.makedirs(out_folder, exist_ok=True)
    plt.savefig(f"{out_folder}/images_matrix.png", dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

def plotsave_gcam_image_grid(images, labels, figsize=(15, 15), out_folder:str = "plots", show=False):
    grid_s = len(images)
    fig = plt.figure(figsize=figsize)
    
    # Add space for labels and titles
    grid = plt.GridSpec(grid_s + 2, grid_s + 1, figure=fig)
    
    # Add x-axis title
    plt.subplot(grid[0, 1:])
    plt.text(0.5, 0.5, 'Input Images', ha='center', va='bottom', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    # Add y-axis title
    plt.subplot(grid[1:, 0])
    plt.text(0, 0.5, 'Embedding Targets', ha='right', va='center', rotation=90, fontsize=12, fontweight='bold')
    plt.axis('off')
    
    # Create axes for labels
    for i in range(grid_s):
        # Column labels (top)
        plt.subplot(grid[1, i+1])
        plt.text(0.5, 0.5, labels[i], ha='center', va='center', rotation=45)
        plt.axis('off')
        
        # Row labels (left)
        plt.subplot(grid[i+2, 1])
        plt.text(0.5, 0.5, labels[i], ha='right', va='center')
        plt.axis('off')
        
        # Plot images
        for j in range(grid_s):
            ax = plt.subplot(grid[i+2, j+1])
            ax.imshow(images[i][j])
            ax.axis('off')
    
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    
    os.makedirs(out_folder, exist_ok=True)
    plt.savefig(f"{out_folder}/gcams_matrix.png", dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def plot_grid_analysis(img_paths, labels, similarity_matrix, overlayed_cams, out_folder:str = "plots"):
    
    plotsave_similarity_matrix(labels, similarity_matrix)
    plotsave_image_matrix(img_paths, labels)
    plotsave_gcam_image_grid(overlayed_cams, labels)
    
    # Read the images
    img1 = mpimg.imread(f"{out_folder}/sim_matrix.png")
    img2 = mpimg.imread(f"{out_folder}/images_matrix.png")
    img3 = mpimg.imread(f"{out_folder}/gcams_matrix.png")
    
    # Create a figure with three subplots - two on top, one below
    fig = plt.figure(figsize=(24, 24))
    
    # Define grid layout
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
    
    # Create three axes
    ax1 = fig.add_subplot(gs[0, 0])  # top left
    ax2 = fig.add_subplot(gs[0, 1])  # top right
    ax3 = fig.add_subplot(gs[1, 0])  # bottom left
    
    # Display the images
    ax1.imshow(img1)
    ax1.axis('off')
    
    ax2.imshow(img2)
    ax2.axis('off')
    
    ax3.imshow(img3)
    ax3.axis('off')
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()
    
    # Clear the current figure to prevent overlapping
    plt.clf()


def plot_splice_word_weights(words: np.ndarray, weights: torch.Tensor):
    # Convert the weights tensor to a NumPy array
    weights = weights.cpu().numpy() if weights.is_cuda else weights.numpy()

    # Sort words and weights for better visualization
    sorted_indices = np.argsort(weights)[::-1]
    sorted_words = words[sorted_indices]
    sorted_weights = weights[sorted_indices]

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.bar(sorted_words, sorted_weights, color='skyblue')
    plt.xticks(rotation=90)
    plt.xlabel("Words")
    plt.ylabel("Weights")
    plt.title("Word Weights Bar Plot")
    plt.tight_layout()
    plt.show()
