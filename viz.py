"""
Utility functions for visualizing medical images (from files or numpy arrays)
and their associated bounding boxes in various formats (YOLO normalized, pixel coordinates).
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import measure


# --- Constants ---
DEFAULT_CMAP = "gray"
DEFAULT_BBOX_COLOR = "r"
DEFAULT_FONTSIZE = 12

# --- Helper Functions ---


def png_to_np(png_filepath: str) -> Optional[np.ndarray]:
    """
    Reads a PNG file and converts it to a NumPy array.

    Args:
        png_filepath (str): Path to the PNG file.

    Returns:
        Optional[np.ndarray]: A NumPy array representing the image (RGB or Grayscale),
                              or None if an error occurs. The array shape
                              will be (H, W) for grayscale or (H, W, 3) for RGB.
    """
    if not os.path.exists(png_filepath):
        print(f"Error: File not found at {png_filepath}")
        return None
    try:
        img = Image.open(png_filepath)
        # Convert to numpy array, keeping original channels
        return np.array(img)
    except Exception as e:
        print(f"An error occurred reading {png_filepath}: {e}")
        return None


def bbox_txt_to_list(txt_filepath: str) -> List[Tuple[float, float, float, float]]:
    """
    Reads YOLO bounding box coordinates from a TXT file.
    Each line format: class_id x_center y_center width height (space-separated).
    Coordinates are normalized to the range [0, 1].

    Args:
        txt_filepath (str): Path to the TXT file.

    Returns:
        List[Tuple[float, float, float, float]]: A list of tuples, where each tuple represents a
                                                 normalized YOLO bounding box (x_center, y_center, width, height).
                                                 Returns an empty list if the file is not found or is empty/invalid.
    """
    bboxes = []
    if not os.path.exists(txt_filepath):
        print(
            f"Warning: Label file not found at {txt_filepath}, returning empty bbox list."
        )
        return bboxes
    try:
        with open(txt_filepath, "r") as f:
            for line in f:
                try:
                    parts = line.strip().split()
                    if len(parts) == 5:  # class_id x_center y_center width height
                        # Extract only bbox coordinates
                        x_center, y_center, width, height = map(float, parts[1:])
                        bboxes.append((x_center, y_center, width, height))
                    else:
                        print(
                            f"Warning: Invalid data format in line: {line.strip()} in {txt_filepath}"
                        )
                except ValueError:
                    print(
                        f"Warning: Could not parse float values in line: {line.strip()} in {txt_filepath}"
                    )
    except Exception as e:
        print(f"An error occurred reading {txt_filepath}: {e}")
    return bboxes


def unnormalize_yolo_bboxes(
    yolo_bboxes: List[Tuple[float, float, float, float]],
    img_width: int,
    img_height: int,
) -> List[Tuple[int, int, int, int]]:
    """
    Unnormalizes a list of YOLO bounding boxes to absolute pixel coordinates.

    Converts from (x_center, y_center, width, height) [0, 1] format to
    (x_min, y_min, pixel_width, pixel_height) integer format.

    Args:
        yolo_bboxes (List[Tuple[float, float, float, float]]): List of normalized bounding boxes in YOLO format.
        img_width (int): Width of the original image in pixels.
        img_height (int): Height of the original image in pixels.

    Returns:
        List[Tuple[int, int, int, int]]: List of bounding boxes in pixel coordinates
                                         (x_min, y_min, pixel_width, pixel_height). Coordinates are integers.
    """
    pixel_bboxes = []
    if img_width <= 0 or img_height <= 0:
        print("Error: Image width and height must be positive.")
        return []

    for bbox in yolo_bboxes:
        x_center, y_center, width_norm, height_norm = bbox
        # Calculate pixel width and height
        pixel_width = int(width_norm * img_width)
        pixel_height = int(height_norm * img_height)
        # Calculate pixel top-left corner (x_min, y_min)
        x_min = int((x_center * img_width) - (pixel_width / 2))
        y_min = int((y_center * img_height) - (pixel_height / 2))

        # Ensure coordinates are within image bounds
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        pixel_width = min(img_width - x_min, pixel_width)
        pixel_height = min(img_height - y_min, pixel_height)
        pixel_bboxes.append((x_min, y_min, pixel_width, pixel_height))

    return pixel_bboxes


def get_bbox_from_mask_skimage(
    mask: np.ndarray,
) -> List[Tuple[int, int, int, int]]:
    """
    Calculates bounding boxes from a 2D segmentation mask using scikit-image.

    Args:
        mask (np.ndarray): 2D numpy array (H, W) representing the segmentation mask (binary or integer).

    Returns:
        List[Tuple[int, int, int, int]]: List of bounding boxes in pixel coordinates
                                         (x_min, y_min, width, height). Returns empty list if no objects found.
    """
    if mask.max() == 0:  # No lesions/objects in the mask
        return []

    # Ensure mask is integer type for measure.label
    int_mask = mask.astype(np.uint8)
    labels = measure.label(
        int_mask, connectivity=2
    )  # Use connectivity=2 for 8-connectivity
    properties = measure.regionprops(labels)

    if not properties:
        return []

    bounding_boxes = []
    for prop in properties:
        # regionprops bbox is (min_row, min_col, max_row, max_col)
        min_r, min_c, max_r, max_c = prop.bbox
        # Convert to (x_min, y_min, width, height)
        x_min = min_c
        y_min = min_r
        width = max_c - min_c
        height = max_r - min_r
        bounding_boxes.append((x_min, y_min, width, height))

    return bounding_boxes


# --- Plotting Functions ---


def plot_images_with_bboxes(
    images: List[np.ndarray],
    bboxes_list: List[List[Tuple[int, int, int, int]]],
    titles: Optional[List[str]] = None,
    main_title: str = "",
    cols: int = 2,
    figsize_scale: int = 5,
    figsize: Optional[Tuple[int, int]] = None,
    cmap: str = DEFAULT_CMAP,
    bbox_color: str = DEFAULT_BBOX_COLOR,
    fontsize: int = DEFAULT_FONTSIZE,
    filename_to_save: Optional[str] = None,
    wspace: float = 0.05,
    hspace: float = 0.1,
) -> None:
    """
    Plots multiple images side-by-side with their corresponding bounding boxes.

    Args:
        images (List[np.ndarray]): List of 2D numpy arrays (images). Assumes all images have same dimensions for simplicity.
        bboxes_list (List[List[Tuple[int, int, int, int]]]): List of lists of bounding boxes.
                                                              Each inner list corresponds to an image.
                                                              Format: (x_min, y_min, width, height) in pixels.
        titles (Optional[List[str]], optional): List of titles for each subplot. Defaults to None.
        main_title (str, optional): Overall title for the figure. Defaults to "".
        cols (int, optional): Number of columns in the subplot grid. Defaults to 2.
        figsize_scale (int, optional): Scaling factor for figure size. Defaults to 5.
        figsize (Optional[Tuple[int, int]], optional): Figsize scale (for example (10, 5)).
                                                       If None then figsize_scale is used. Defaults to None.
        cmap (str, optional): Colormap for displaying images. Defaults to "gray".
        bbox_color (str, optional): Color for bounding box outlines. Defaults to "r".
        fontsize (int, optional): Fontsize for titles. Defaults to 12.
        filename_to_save (Optional[str], optional): If provided, saves the figure to this path instead of showing it. Defaults to None.
        wspace (float, optional): Width space between subplots. Defaults to 0.05.
        hspace (float, optional): Height space between subplots. Defaults to 0.1.
    """
    num_images = len(images)
    if num_images == 0:
        print("No images provided to plot.")
        return
    if len(bboxes_list) != num_images:
        raise ValueError("Number of images and bounding box lists must match.")
    if titles and len(titles) != num_images:
        raise ValueError("Number of images and titles must match.")

    rows = (num_images + cols - 1) // cols

    if figsize is None:
        figsize = (figsize_scale * cols, figsize_scale * rows)

    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    # squeeze=False ensures axes is always 2D array

    if main_title:
        fig.suptitle(main_title, fontsize=fontsize + 2)

    fig.set_tight_layout(True)
    # fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to prevent title overlap

    for i, (img, bboxes) in enumerate(zip(images, bboxes_list)):
        row = i // cols
        col = i % cols
        ax = axes[row, col]

        # Display image - handle grayscale and RGB
        if img.ndim == 2:
            ax.imshow(img, cmap=cmap)
        elif img.ndim == 3:
            ax.imshow(img)
        else:
            print(
                f"Warning: Skipping image {i} due to unexpected dimensions: {img.ndim}"
            )
            continue

        # Add bounding boxes
        for x, y, width, height in bboxes:
            rect = patches.Rectangle(
                (x, y),
                width,
                height,
                linewidth=1,
                edgecolor=bbox_color,
                facecolor="none",
            )
            ax.add_patch(rect)

        # Set title
        if titles:
            ax.set_title(titles[i], fontsize=fontsize)

        ax.axis("off")

    # Hide unused subplots
    for i in range(num_images, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis("off")

    plt.subplots_adjust(
        left=0.02,
        right=0.98,
        bottom=0.02,
        top=0.92 if main_title else 0.98,
        wspace=wspace,
        hspace=hspace,
    )

    if filename_to_save:
        try:
            fig.savefig(filename_to_save, bbox_inches="tight", dpi=300)
            print(f"Figure saved to {filename_to_save}")
        except Exception as e:
            print(f"Error saving figure to {filename_to_save}: {e}")
        plt.close(fig)  # Close the figure after saving
    else:
        plt.show()


def show_prediction_with_gt(
    img_filepath: str,
    prediction_bboxes_yolo: List[Tuple[float, float, float, float]],
    gt_label_filepath: Optional[str] = None,
    comments: str = "",
    filename_to_save: Optional[str] = None,
) -> None:
    """
    Loads an image and plots it with ground truth and predicted bounding boxes.

    Args:
        img_filepath (str): Path to the image file (e.g., PNG).
        prediction_bboxes_yolo (List[Tuple[float, float, float, float]]): List of predicted bounding boxes
                                                                          in normalized YOLO format.
        gt_label_filepath (Optional[str], optional): Path to the ground truth label file (.txt).
                                                     If None, only the prediction is shown. Defaults to None.
        comments (str, optional): Additional text to display below the main title. Defaults to "".
        filename_to_save (Optional[str], optional): Path to save the plot image. If None, shows the plot.
                                                    Defaults to None.
    """
    path = Path(img_filepath)
    if not path.is_file():
        print(f"Error: Image file not found: {img_filepath}")
        return

    source_img = png_to_np(img_filepath)
    if source_img is None:
        return

    img_height, img_width = source_img.shape[:2]  # Works for grayscale and RGB

    # Prepare Ground Truth BBoxes (if path provided)
    bbox_gt_pixels = []
    if gt_label_filepath:
        if not os.path.exists(gt_label_filepath):
            print(f"Warning: GT label file not found: {gt_label_filepath}")
        else:
            bbox_gt_yolo = bbox_txt_to_list(gt_label_filepath)
            bbox_gt_pixels = unnormalize_yolo_bboxes(
                bbox_gt_yolo, img_width, img_height
            )
    elif gt_label_filepath is None:
        print("No GT label file provided, showing only prediction.")
        # Fallback: try deriving label path from image path
        derived_label_path = img_filepath.replace("images", "labels").replace(
            path.suffix, ".txt"
        )
        if os.path.exists(derived_label_path):
            print(f"Found label file at derived path: {derived_label_path}")
            bbox_gt_yolo = bbox_txt_to_list(derived_label_path)
            bbox_gt_pixels = unnormalize_yolo_bboxes(
                bbox_gt_yolo, img_width, img_height
            )
        else:
            print(f"Could not find label file at derived path: {derived_label_path}")

    # Prepare Prediction BBoxes
    bbox_predict_pixels = unnormalize_yolo_bboxes(
        prediction_bboxes_yolo, img_width, img_height
    )

    # --- Plotting ---
    images_to_plot = (
        [source_img, source_img]
        if gt_label_filepath or bbox_gt_pixels
        else [source_img]
    )
    bboxes_for_plot = (
        [bbox_gt_pixels, bbox_predict_pixels]
        if gt_label_filepath or bbox_gt_pixels
        else [bbox_predict_pixels]
    )
    plot_titles = (
        ["Ground Truth", "Prediction"]
        if gt_label_filepath or bbox_gt_pixels
        else ["Prediction"]
    )

    main_plot_title = f"File: {path.parent.name}/{path.name}\n{comments}"

    plot_images_with_bboxes(
        images=images_to_plot,
        bboxes_list=bboxes_for_plot,
        titles=plot_titles,
        main_title=main_plot_title,
        cols=len(images_to_plot),  # Show side-by-side
        filename_to_save=filename_to_save,
    )
