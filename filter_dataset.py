"""
Filter 2D dataset removing black or low foreground percentage images.
"""

import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from tqdm import tqdm

# --- Logging Configuration ---
DEFAULT_LOG_FILENAME = "filter_dataset_log.txt"

logger = logging.getLogger(__name__)


def setup_logging(log_filename: str = DEFAULT_LOG_FILENAME):
    """Configures logging to write to a file."""
    # Remove existing handlers if any were added previously (e.g., basicConfig)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logfmt_str = "%(asctime)s %(name)s:%(lineno)03d: %(message)s"
    # Configure logging to file and console (optional)
    logging.basicConfig(
        level=logging.INFO,  # Log INFO level and above
        format=logfmt_str,
        handlers=[logging.FileHandler(log_filename, mode="w")],
    )
    print(f"Logging configured. Log file: {log_filename}")


# --- Constants ---
IMAGE_EXT = ".png"
LABEL_EXT = ".txt"
IMAGE_DIR_NAME = "images"
LABEL_DIR_NAME = "labels"


def calculate_foreground_percentage(
    image_path: str, intensity_threshold: int = 5
) -> Optional[float]:
    """
    Calculates the percentage of foreground pixels (intensity > threshold) on image.

    Args:
        image_path: Path to image file.
        intensity_threshold: Intensity threshold for determining foreground pixel.
        Pixels with intensity <= threshold are considered background.

    Returns:
        Percentage of foreground pixels (0.0 to 100.0) or None if the image
        could not be loaded.
    """
    try:
        # Load in grayscale since R=G=B in our data
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.warning(f"Could not read image: {image_path}")
            return None

        total_pixels = img.size
        if total_pixels == 0:
            logger.warning(f"The image has zero size: {image_path}")
            return 0.0

        # Pixels with value > intensity threshold are foreground
        foreground_pixels = np.count_nonzero(img > intensity_threshold)

        return (foreground_pixels / total_pixels) * 100.0

    except Exception as e:
        logger.error(f"Error while processing image {image_path}: {e}")
        return None


def check_intensity_std_dev(
    image_path: str,
    min_intensity_threshold: Optional[float] = None,
    min_std_dev_threshold: Optional[float] = None,
) -> bool:
    """
    Checks if an image passes intensity and standard deviation filters.

    Args:
        image_path (str): Path to the image file.
        min_intensity_threshold (float): Minimum required maximum pixel value.
        min_std_dev_threshold (float): Minimum required standard deviation of pixel values.

    Returns:
        bool: True if the image passes both filters, False otherwise.
    """

    if min_intensity_threshold is None and min_std_dev_threshold is None:
        return True

    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.warning(f"Could not read image: {image_path}")
            return False  # Treat unreadable images as failing the filter

        if min_intensity_threshold is not None:
            max_val = img.max()
            if max_val < min_intensity_threshold:
                # logger.info(
                #     f"Filter fail (intensity): "
                #     f"{os.path.basename(image_path)} (max={max_val} < {min_intensity_threshold})"
                # )
                return False

        if min_std_dev_threshold is not None:
            std_dev = img.std()
            if std_dev < min_std_dev_threshold:
                # logger.info(
                #     f"Filter fail (std dev): "
                #     f"{os.path.basename(image_path)} (std={std_dev:.2f} < {min_std_dev_threshold})"
                # )
                return False
        return True  # Passes both filters
    except Exception as e:
        logger.error(f"Error checking intensity/std dev for {image_path}: {e}")
        return False  # Treat errors as failing the filter


def filter_dataset(
    dataset_root: str,
    splits: List[str],
    action: str = "move",
    foreground_threshold: Optional[float] = None,
    intensity_threshold: int = 5,
    min_max_intensity: Optional[float] = None,
    min_std_dev: Optional[float] = None,
    filtered_dir_suffix: str = "_filtered_moved",
    dry_run: bool = True,
) -> None:
    """
    Filters the YOLO dataset by removing images with a low foreground percentage
    and their corresponding labels files.

    Args:
        dataset_root: The root directory of the YOLO dataset (containing the images/ and labels/ folders).
        splits: List of subsets to filter (e.g. ['train', 'val', 'test']).
        action (str): 'delete' to remove files, 'move' to move them to a backup dir.
        foreground_threshold: Percentage threshold. Images with a foreground percentage
                              *below* this value will be removed (default None).
        intensity_threshold: Intensity threshold for determining a foreground pixel
                             (default 5). Pixels <= this value are considered background.
        min_max_intensity: Minimum allowed maximum pixel intensity. If None, the filter is skipped.
        min_std_dev: Minimum acceptable standard deviation of intensity. If None, the filter is skipped.
        filtered_dir_suffix (str): Suffix for the directory where filtered files are moved.
        dry_run: If True, only outputs information about which files would be removed,
                 but does not actually remove them. Default is True.
    """
    logger.info(f"Start filtering the dataset in {dataset_root}")
    logger.info(f"Action: {action.upper()}")
    logger.info("___Parameters:___")
    logger.info(f"splits: {splits}")
    logger.info(f"fg_thresh: {foreground_threshold}%")
    logger.info(f"intensity_thresh_for_fg: {intensity_threshold}")
    logger.info(f"min_max_intensity: {min_max_intensity}")
    logger.info(f"min_std_dev: {min_std_dev}")
    logger.info(f"dry_run: {dry_run}")

    base_path = Path(dataset_root)
    images_base_dir = base_path / IMAGE_DIR_NAME
    labels_base_dir = base_path / LABEL_DIR_NAME
    filtered_base_dir = base_path.parent / (
        base_path.name + filtered_dir_suffix
    )  # Create alongside original

    if not images_base_dir.is_dir() or not labels_base_dir.is_dir():
        logger.error(
            f"Dataset directory structure invalid. 'images' and 'labels' subdirs expected in {dataset_root}"
        )
        return

    if action == "move":
        logger.info(f"Filtered files will be moved to: {filtered_base_dir}")
        # Create corresponding filtered structure
        if not dry_run:
            os.makedirs(filtered_base_dir, exist_ok=True)

    action_past_tense = "Moved" if action == "move" else "Deleted"
    # action_gerund = "Moving..." if action == "move" else "Deleting..."

    final_statistics = {}
    filtered_images = {}

    total_files_scanned = 0
    total_kept_images_count = 0
    error_count = 0
    total_filtered_fg_count = 0
    total_filtered_intensity_std_count = 0
    total_filtered_intensity_std_count = 0
    total_filtered_labels_count = 0

    for split in splits:
        split_files_scanned = 0
        split_kept_images_count = 0
        split_filtered_fg_count = 0
        split_filtered_intensity_std_count = 0
        split_filtered_intensity_std_count = 0
        split_filtered_images = []
        split_filtered_labels_count = 0

        logger.info("--------------------------")
        logger.info(f"Subset Processing: {split}")

        image_dir_src = images_base_dir / split
        label_dir_src = labels_base_dir / split

        image_dir_dest: Optional[Path] = None
        label_dir_dest: Optional[Path] = None

        if action == "move":
            image_dir_dest = filtered_base_dir / IMAGE_DIR_NAME / split
            label_dir_dest = filtered_base_dir / LABEL_DIR_NAME / split

            if not dry_run:
                image_dir_dest.mkdir(parents=True, exist_ok=True)
                label_dir_dest.mkdir(parents=True, exist_ok=True)

        if not image_dir_src.is_dir():
            logger.warning(
                f"Image directory for subset '{split}' not found: {image_dir_src}. Skipping."
            )
            continue
        if not label_dir_src.is_dir():
            logger.warning(
                f"The folder with the labels for the subset '{split}' was not found: {label_dir_src}."
                f" The label files will not be deleted."
            )
        # Search for all image files in the subset folder
        image_files = list(image_dir_src.glob(f"*{IMAGE_EXT}"))

        if not image_files:
            logger.info(f"No images {IMAGE_EXT} found in subset '{split}'.")
            continue

        logger.info(f"Found {len(image_files)} images in subset '{split}'.")

        for image_path in tqdm(image_files, desc=f"Filtration '{split}'"):
            split_files_scanned += 1

            label_filename = image_path.stem + LABEL_EXT
            label_path = label_dir_src / label_filename

            filter_reason = None

            # 1. Filter by Foreground Percentage
            if foreground_threshold is not None:
                foreground_perc = calculate_foreground_percentage(
                    str(image_path), intensity_threshold
                )
                if foreground_perc is None:
                    error_count += 1
                    continue
                if foreground_perc < foreground_threshold:
                    filter_reason = "Foreground"
                    # filter_reason = (
                    #     f"Foreground ({foreground_perc:.2f}% < {foreground_threshold}%)"
                    # )

            # 2. Filter by intensity and standard deviation (if not already filtered)
            if filter_reason is None and (
                min_max_intensity is not None or min_std_dev is not None
            ):
                passes_intensity_std = check_intensity_std_dev(
                    str(image_path), min_max_intensity, min_std_dev
                )
                if not passes_intensity_std:
                    filter_reason = "Intensity/StdDev"

            if filter_reason:
                if filter_reason == "Intensity/StdDev":
                    split_filtered_intensity_std_count += 1
                else:
                    split_filtered_fg_count += 1

                split_filtered_images.append(image_path.name)
                # log_prefix = (
                #     f"{action_gerund}:"
                #     if not dry_run
                #     else f"DRY RUN ({action_gerund}):"
                # )
                # logger.info(
                #     f"{log_prefix} {image_path.name} (Filter Reason: {filter_reason})"
                # )

                if not dry_run:
                    try:
                        if action == "delete":
                            image_path.unlink()
                            logger.info(f"  Image removed: {image_path}")
                        elif action == "move" and image_dir_dest:
                            image_path_dest = image_dir_dest / image_path.name
                            shutil.move(str(image_path), str(image_path_dest))
                            logger.debug(f"  Image moved to: {image_path_dest}")
                    except OSError as e:
                        logger.error(f"  Failed to delete image {image_path}: {e}")
                        error_count += 1

                    # Perform an action for ground truth
                    if label_dir_src.is_dir() and label_path.exists():
                        try:
                            if action == "delete":
                                label_path.unlink()
                                logger.info(
                                    f"   Ground truth bboxes deleted: {label_path}"
                                )
                                split_filtered_labels_count += 1
                            elif action == "move" and label_dir_dest:
                                label_path_dest = label_dir_dest / label_path.name
                                shutil.move(str(label_path), str(label_path_dest))
                                logger.info(
                                    f"   Ground truth bboxes moved to: {label_path_dest}"
                                )
                                split_filtered_labels_count += 1
                        except OSError as e:
                            logger.error(
                                f"  Failed to delete ground truth bboxes {label_path}: {e}"
                            )
                            error_count += 1
                else:  # dry_run
                    if label_dir_src.is_dir() and label_path.exists():
                        split_filtered_labels_count += 1
            else:
                #  logger.info(f"Saving: {image_path}")
                split_kept_images_count += 1

        total_kept_images_count += split_kept_images_count
        total_files_scanned += split_files_scanned
        total_filtered_fg_count += split_filtered_fg_count
        total_filtered_intensity_std_count += split_filtered_intensity_std_count
        filtered_images[split] = sorted(split_filtered_images)
        total_filtered_labels_count += split_filtered_labels_count

        final_statistics[split] = {
            "files_scanned": split_files_scanned,
            "kept_images_count": split_kept_images_count,
            "filtered_images": split_filtered_fg_count
            + split_filtered_intensity_std_count,
            "filtered_fg_count": split_filtered_fg_count,
            "filtered_intensity_std_count": split_filtered_intensity_std_count,
            "filtered_labels": split_filtered_labels_count,
        }

        logger.info("--------------------------")
        logger.info(f"Curren split: ___{split}___")
        logger.info(f" Image files scanned: {split_files_scanned}")
        logger.info(f" Kept images: {split_kept_images_count}")
        logger.info(
            f" {action_past_tense} images: {split_filtered_fg_count + split_filtered_intensity_std_count}"
        )
        logger.info(
            f" - by foreground threshold (<{foreground_threshold}%): {split_filtered_fg_count}"
        )
        logger.info(
            f" - by intensity (<{min_max_intensity}) / std dev (<{min_std_dev}): {split_filtered_intensity_std_count}"
        )
        logger.info(f" {action_past_tense} labels files: {split_filtered_labels_count}")

    total_filtered_images = total_filtered_fg_count + total_filtered_intensity_std_count

    final_statistics["total"] = {
        "total_files_scanned": total_files_scanned,
        "total_kept_images_count": total_kept_images_count,
        "total_filtered_images": total_filtered_images,
        "total_filterd_fg_count": total_filtered_fg_count,
        "total_filtered_intensity_std_count": total_filtered_intensity_std_count,
        "total_filtered_labels": total_filtered_labels_count,
    }

    logger.info("--------------------------")
    logger.info("Finished dataset filtering.")
    logger.info(f"Final statistics ({'DRY RUN' if dry_run else 'REAL DELETION'}):")
    logger.info("--------------------------")
    logger.info(f" Total image files scanned: {total_files_scanned}")
    logger.info(f" Total kept images: {total_kept_images_count}")
    logger.info(f" {action_past_tense} images (total): {total_filtered_images}")
    logger.info(
        f" - by foreground threshold (<{foreground_threshold}%): {total_filtered_fg_count}"
    )
    logger.info(
        f" - by intensity (<{min_max_intensity}) / std dev (<{min_std_dev}): {total_filtered_intensity_std_count}"
    )
    logger.info(
        f" {action_past_tense} labels files (total): {total_filtered_intensity_std_count}"
    )
    logger.info(f" Errors during processing/deleting: {error_count}")
    if dry_run:
        logger.info("Dry run completed. No files were deleted.")

    with open((Path.cwd() / "filtered_files_dict.txt"), "w") as fp:
        json.dump(filtered_images, fp)


def main():
    parser = argparse.ArgumentParser(
        description="Filters the YOLO dataset by removing nearly black images and their markup."
    )
    parser.add_argument(
        "dataset_root",
        type=str,
        help="The root directory of the YOLO dataset (should contain the 'images' and 'labels' folders).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="List of subsets to process (e.g. train val test). Defaults to all three.",
    )
    parser.add_argument(
        "--action",
        type=str,
        choices=["delete", "move"],
        default="move",
        help="Action: 'delete' or 'move' (default: move).",
    )
    parser.add_argument(
        "--fg_thresh",
        type=float,
        default=None,  # 5.0 recommended
        help="Threshold in percent for foreground. Images with a lower percentage will be removed (default: 5.0).",
    )
    parser.add_argument(
        "--intensity_thresh",
        type=int,
        default=5,
        help="Pixel intensity threshold for foreground detection (default: 5).",
    )
    parser.add_argument(
        "--min_max_intensity",
        type=float,
        default=None,  # 5.0 recommended
        help="Min. allowed max intensity. Images < this value are filtered. (default: None - disabled).",
    )
    parser.add_argument(
        "--min_std_dev",
        type=float,
        default=None,  # 5.0 recommended
        help="Min. allowed standard deviation. Images < this value are filtered. (default: None - disabled).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_false",
        default=True,
        help="Run without actually deleting files, just output statistics.",
    )

    args = parser.parse_args()
    setup_logging()

    if not os.path.isdir(args.dataset_root):
        logger.error(f"Dataset root directory not found: {args.dataset_root}")
        return

    fg_thresh = args.fg_thresh  # if args.fg_thresh >= 0 else None
    min_max_intensity = (
        args.min_max_intensity
        if args.min_max_intensity is not None and args.min_max_intensity >= 0
        else None
    )
    min_std_dev = (
        args.min_std_dev
        if args.min_std_dev is not None and args.min_std_dev >= 0
        else None
    )

    if fg_thresh is None and min_max_intensity is None and min_std_dev is None:
        logger.warning(
            "All filters are disabled (fg_thresh < 0, min_max_intensity and min_std_dev are not set)."
            "No filtering will be performed."
        )
        return

    filter_dataset(
        dataset_root=args.dataset_root,
        splits=args.splits,
        action=args.action,
        foreground_threshold=fg_thresh,
        intensity_threshold=args.intensity_thresh,
        min_max_intensity=min_max_intensity,
        min_std_dev=min_std_dev,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
