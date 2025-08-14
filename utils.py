"""
Utility functions for performing Exploratory Data Analysis (EDA) on
3D medical imaging datasets stored in NIfTI format and
analyzing Ultralytics training results from results.csv.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import matplotlib.pyplot as plt
import pandas as pd


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# --- Constants ---
MAP50_COL = "metrics/mAP50(B)"
MAP50_95_COL = "metrics/mAP50-95(B)"
PRECISION_COL = "metrics/precision(B)"
RECALL_COL = "metrics/recall(B)"
FITNESS_COL = "fitness"


# --- results.csv analysing functions ---


def calculate_fitness(
    mAP50: float, mAP50_95: float, w_map50: float = 0.1, w_map95: float = 0.9
) -> float:
    """
    Calculates the fitness metric based on mAP50 and mAP50-95.
    This metric is used by Ultralytics for selecting the best model checkpoint.

    Args:
        mAP50 (float): Mean Average Precision at IoU threshold 0.50.
        mAP50_95 (float): Mean Average Precision averaged over IoU thresholds [0.50:0.05:0.95].
        w_map50 (float, optional): Weight for mAP50. Defaults to 0.1.
        w_map95 (float, optional): Weight for mAP50-95. Defaults to 0.9.
    """
    return mAP50 * w_map50 + mAP50_95 * w_map95


def add_fitness_metric(
    results_df: pd.DataFrame,
    map50_col: str = MAP50_COL,
    map50_95_col: str = MAP50_95_COL,
    fitness_col: str = FITNESS_COL,
) -> pd.DataFrame:
    """
    Adds a 'fitness' column to the results DataFrame based on mAP metrics.

    Args:
        results_df (pd.DataFrame): DataFrame loaded from results.csv.
        map50_col (str): Name of the mAP50 column.
        map50_95_col (str): Name of the mAP50-95 column.
        fitness_col (str): Name of the fitness column to be added.

    Returns:
        pd.DataFrame: DataFrame with the added fitness column.
    """
    # Strip whitespace from column names just in case
    results_df.columns = results_df.columns.str.strip()
    if map50_col not in results_df.columns or map50_95_col not in results_df.columns:
        raise KeyError(
            f"Required mAP columns ('{map50_col}', '{map50_95_col}') not found in DataFrame."
        )

    results_df[fitness_col] = calculate_fitness(
        results_df[map50_col], results_df[map50_95_col]
    )
    return results_df


def get_best_epoch_metric(
    results_df: pd.DataFrame, metric_col: str
) -> Optional[Tuple[int, float]]:
    """
    Finds the epoch and value for the maximum value of a given metric column.

    Args:
        results_df (pd.DataFrame): DataFrame loaded from results.csv.
        metric_col (str): The name of the metric column to analyze.

    Returns:
        Optional[Tuple[int, float]]: A tuple containing (best_epoch, best_value),
                                     or None if the column doesn't exist or is empty.
                                     Epoch number is 1-based.
    """
    results_df.columns = results_df.columns.str.strip()
    if metric_col not in results_df.columns or results_df[metric_col].empty:
        print(f"Warning: Metric column '{metric_col}' not found or empty.")
        return None

    best_idx = results_df[metric_col].idxmax()
    best_epoch = best_idx + 1  # Epochs are 1-based
    best_value = results_df.loc[best_idx, metric_col]
    return best_epoch, best_value


def get_metrics_at_epoch(
    results_df: pd.DataFrame, epoch: int, metric_cols: List[str]
) -> Optional[Dict[str, float]]:
    """
    Retrieves the values of specified metrics at a given epoch.

    Args:
        results_df (pd.DataFrame): DataFrame loaded from results.csv.
        epoch (int): The 1-based epoch number to retrieve metrics for.
        metric_cols (List[str]): A list of metric column names to retrieve.

    Returns:
        Optional[Dict[str, float]]: A dictionary with metric names as keys and their values
                                    at the specified epoch, or None if the epoch is out of bounds
                                    or required columns are missing.
    """
    results_df.columns = results_df.columns.str.strip()
    epoch_idx = epoch - 1  # Convert 1-based epoch to 0-based index
    if epoch_idx < 0 or epoch_idx >= len(results_df):
        print(f"Warning: Epoch {epoch} is out of bounds (0-{len(results_df) - 1}).")
        return None

    metrics_at_epoch = {}
    all_cols_present = True
    for col in metric_cols:
        if col not in results_df.columns:
            print(f"Warning: Metric column '{col}' not found in DataFrame.")
            all_cols_present = False
        else:
            metrics_at_epoch[col] = results_df.loc[epoch_idx, col]

    return metrics_at_epoch if all_cols_present else None


def analyze_best_metrics(
    results_df: pd.DataFrame,
    map50_col: str = MAP50_COL,
    map50_95_col: str = MAP50_95_COL,
    precision_col: str = PRECISION_COL,
    recall_col: str = RECALL_COL,
    fitness_col: str = FITNESS_COL,
) -> Dict[str, Tuple[int, float]]:
    """
    Analyzes the results DataFrame to find the best epochs and values for key metrics.

    Calculates fitness if not present.

    Args:
        results_df (pd.DataFrame): DataFrame loaded from results.csv.
        map50_col (str): Name of the mAP50 column.
        map50_95_col (str): Name of the mAP50-95 column.
        precision_col (str): Name of the precision column.
        recall_col (str): Name of the recall column.
        fitness_col (str): Name of the fitness column.

    Returns:
        Dict[str, Tuple[int, float]]: Dictionary containing best metrics, where keys are
                                      metric names (e.g., "best_fitness") and values are
                                      tuples of (best_epoch, best_value).
    """
    results_df.columns = results_df.columns.str.strip()
    if fitness_col not in results_df.columns:
        try:
            results_df = add_fitness_metric(
                results_df, map50_col, map50_95_col, fitness_col
            )
        except KeyError as e:
            print(
                f"Error adding fitness metric: {e}. Cannot proceed with full analysis."
            )
            return {}  # Return empty if fitness can't be calculated

    metrics_summary = {}

    best_metrics_cols = {
        "best_mAP50": map50_col,
        "best_mAP50-95": map50_95_col,
        "best_precision": precision_col,
        "best_recall": recall_col,
        "best_fitness": fitness_col,
    }

    # Find best epoch for each primary metric
    for name, col in best_metrics_cols.items():
        best_result = get_best_epoch_metric(results_df, col)
        if best_result:
            metrics_summary[name] = best_result

    # Find metrics at the best fitness epoch
    if "best_fitness" in metrics_summary:
        best_fitness_epoch = metrics_summary["best_fitness"][0]
        metrics_at_best_fitness = get_metrics_at_epoch(
            results_df,
            best_fitness_epoch,
            [map50_col, map50_95_col, precision_col, recall_col],
        )
        if metrics_at_best_fitness:
            metrics_summary["best_fitness_epoch_mAP50"] = (
                best_fitness_epoch,
                metrics_at_best_fitness[map50_col],
            )
            metrics_summary["best_fitness_epoch_mAP50-95"] = (
                best_fitness_epoch,
                metrics_at_best_fitness[map50_95_col],
            )
            metrics_summary["best_fitness_epoch_precision"] = (
                best_fitness_epoch,
                metrics_at_best_fitness[precision_col],
            )
            metrics_summary["best_fitness_epoch_recall"] = (
                best_fitness_epoch,
                metrics_at_best_fitness[recall_col],
            )

    return metrics_summary


# --- EDA Functions ---


def analyze_nifti_dimensions(
    data_dir: str, file_formats: Tuple[str, ...] = (".nii", ".nii.gz")
) -> Dict[Tuple[int, ...], int]:
    """
    Walks through a directory, finds NIfTI files, and counts occurrences of different image dimensions.

    Args:
        data_dir (str): The root directory to search for NIfTI files.
        file_formats (Tuple[str,...], optional): Valid NIfTI file extensions. Defaults to (".nii", ".nii.gz").

    Returns:
        Dict[Tuple[int, ...], int]: A dictionary where keys are dimension tuples (e.g., (256, 256, 180))
                                   and values are the counts of files with those dimensions.
    """
    dimensions_count: Dict[Tuple[int, ...], int] = {}
    logger.info(f"Analyzing NIfTI dimensions in: {data_dir}")

    for dirpath, _, filenames in os.walk(data_dir):
        for file in filenames:
            if file.endswith(file_formats) and not file.startswith("."):
                file_path = os.path.join(dirpath, file)
                try:
                    img = nib.load(file_path)
                    dimensions = img.header.get_data_shape()
                    # Ensure dimensions tuple contains only spatial dimensions if 4D+
                    spatial_dimensions = tuple(dimensions[:3])

                    dimensions_count[spatial_dimensions] = (
                        dimensions_count.get(spatial_dimensions, 0) + 1
                    )
                except FileNotFoundError:
                    logger.warning(
                        f"File not found during walk (should not happen): {file_path}"
                    )
                except Exception as error:
                    logger.error(f"Error processing {file_path}: {error}")

    # Sort by count descending
    sorted_dimensions = sorted(
        dimensions_count.items(), key=lambda item: item[1], reverse=True
    )

    print("\n--- Different Dimensions Found ---")
    if not sorted_dimensions:
        print("No valid NIfTI files found.")
    else:
        for dim, count in sorted_dimensions:
            print(f"  Shape: {dim} - Count: {count}")
    print("---------------------------------\n")
    return dimensions_count


def show_slices_all_modalities(
    subject_dir: str,
    slice_idx: int,
    dim: int = 2,
    file_formats: Tuple[str, ...] = (".nii", ".nii.gz"),
) -> None:
    """
    Displays slices from all NIfTI modalities found in a subject's directory.

    Args:
        subject_dir (str): Path to the directory containing one subject's NIfTI files.
        slice_idx (int): The index of the slice to display.
        dim (int, optional): The dimension along which to slice (0, 1, or 2). Defaults to 2 (axial).
        file_formats (Tuple[str,...], optional): Valid NIfTI file extensions. Defaults to (".nii", ".nii.gz").
    """
    if not os.path.isdir(subject_dir):
        logger.error(f"Subject directory not found: {subject_dir}")
        return

    nifti_files = sorted(
        [
            f
            for f in os.listdir(subject_dir)
            if f.endswith(file_formats) and not f.startswith(".")
        ]
    )

    if not nifti_files:
        logger.warning(f"No NIfTI files found in {subject_dir}.")
        return

    num_files = len(nifti_files)
    # Simple grid layout - adjust cols as needed
    cols = min(num_files, 4)
    rows = (num_files + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    fig.suptitle(
        f"Idx: {os.path.basename(subject_dir)}, Slice: {slice_idx}, Dim: {dim}",
        fontsize=14,
    )

    fig.set_tight_layout(True)
    # fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    current_ax = 0
    img_shape = None

    for filename in nifti_files:
        filepath = os.path.join(subject_dir, filename)
        try:
            img_nii = nib.load(filepath)
            img_3d = img_nii.get_fdata()
            if img_shape is None:
                img_shape = img_3d.shape
            elif img_3d.shape[:3] != img_shape[:3]:  # Check spatial dimensions
                logger.warning(
                    f"Shape mismatch for {filename}: {img_3d.shape} vs {img_shape}. Skipping."
                )
                continue

            if not (0 <= slice_idx < img_3d.shape[dim]):
                logger.warning(
                    f"Slice index {slice_idx} out of bounds for {filename} (dim {dim}, shape {img_3d.shape}). Skipping."
                )
                continue

            if dim == 0:
                sliced_img = img_3d[slice_idx, :, :]
            elif dim == 1:
                sliced_img = img_3d[:, slice_idx, :]
            else:
                sliced_img = img_3d[:, :, slice_idx]

            row = current_ax // cols
            col = current_ax % cols

            # # Rotate axial/coronal slices for better viewing if desired
            # if dim == 2 or dim == 1:
            #     sliced_img = np.rot90(sliced_img)

            axes[row, col].imshow(sliced_img, cmap="gray")
            axes[row, col].set_title(
                filename.replace(".nii.gz", "").replace(".nii", ""), fontsize=12
            )
            axes[row, col].axis("off")
            current_ax += 1

        except Exception as e:
            logger.error(f"Error processing/plotting {filepath}: {e}")
            row = current_ax // cols
            col = current_ax % cols
            axes[row, col].set_title(
                f"Error loading\n{filename}", fontsize=8, color="red"
            )
            axes[row, col].axis("off")
            current_ax += 1

    # Hide unused subplots
    for i in range(current_ax, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis("off")

    plt.show()
