import argparse
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import nibabel as nib
import numpy as np
from skimage import measure
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Constants ---
SLICE_NAMES: Dict[int, str] = {0: "saggital", 1: "frontal", 2: "axial"}
FILE_EXTENSIONS: Dict[str, str] = {"images": "png", "labels": "txt"}
YOLO_FOLDERS: List[str] = ["images", "labels"]
DEFAULT_SPLITS: List[str] = ["train", "val", "test"]
MASK_MODALITY_MSShift = "gt"
MASK_MODALITY_MSLesSeg = "MASK"
MODALITIES_MSShift = ["FLAIR", "T1CE", "T2", "PD", "T1", "gt"]
MODALITIES_MSLesSeg = ["FLAIR", "T2", "T1", "MASK"]

# --- Helper Functions ---


def make_dir(path: str) -> None:
    """Creates a directory if it doesn't exist.

    Args:
        path (str): The full path of the directory to create.
    """
    os.makedirs(path, exist_ok=True)


def normalize_image(img: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Normalizes a numpy array image to the range [0, 255].

    Args:
        img (np.ndarray): The input image as a NumPy array.
        eps (float): A small epsilon value to prevent division by zero.
                     Defaults to 1e-5.

    Returns:
        np.ndarray: The normalized image as a NumPy array of type np.uint8.
    """
    min_val = np.min(img)
    max_val = np.max(img)
    normalized = (img - min_val) / (max_val - min_val + eps)
    return (normalized * 255).astype(np.uint8)


def get_bounding_boxes(
    mask: np.ndarray, class_id: int
) -> List[Tuple[int, float, float, float, float]]:
    """Extracts bounding boxes from a 2D segmentation mask in YOLO format.

    Finds connected components in the mask, calculates the bounding box
    for each component, and formats it according to YOLO specifications
    (class_id, x_center_norm, y_center_norm, width_norm, height_norm).

    Args:
        mask (np.ndarray): A 2D NumPy array representing the segmentation mask
                           (binary or labeled).
        class_id (int): The integer class ID to assign to the bounding boxes.

    Returns:
        List[Tuple[int, float, float, float, float]]: A list of bounding boxes.
        Each bounding box is a tuple containing:
            - class_id (int)
            - x_center_norm (float): Normalized x-coordinate of the bbox center [0, 1].
            - y_center_norm (float): Normalized y-coordinate of the bbox center [0, 1].
            - width_norm (float): Normalized width of the bbox [0, 1].
            - height_norm (float): Normalized height of the bbox [0, 1].
        Returns an empty list if no objects are found in the mask.
    """
    if mask.max() == 0:  # Optimization: Check if the mask is empty
        return []

    # Ensure mask is integer type for measure.label
    mask_int = mask.astype(np.uint8)
    labels = measure.label(mask_int, connectivity=2)
    properties = measure.regionprops(labels)

    if not properties:
        return []

    height, width = mask.shape
    bboxes_yolo = []
    for prop in properties:
        # bbox format: (min_row, min_col, max_row, max_col)
        min_row, min_col, max_row, max_col = prop.bbox

        # Calculate YOLO parameters
        x_center = (min_col + max_col) / 2.0
        y_center = (min_row + max_row) / 2.0
        bbox_width = max_col - min_col
        bbox_height = max_row - min_row

        # Normalize
        x_center_norm = x_center / width
        y_center_norm = y_center / height
        width_norm = bbox_width / width
        height_norm = bbox_height / height

        bboxes_yolo.append(
            (class_id, x_center_norm, y_center_norm, width_norm, height_norm)
        )

    return bboxes_yolo


# --- Default Filename Handling Functions ---


def msshift_filename_parser(filename: str, *args: Any, **kwargs: Any) -> str:
    """Extracts modality from MS_Shift dataset filename (e.g., 'ID_MODALITY_....nii.gz').
    Assumes modality is the second part when split by '_'. Returns None if parsing fails.

    Args:
        filename (str): The input NIfTI filename (basename).
        *args(Any): Catches potential extra arguments from the caller.
        **kwargs (Any): Catches potential extra arguments from the caller.

    Returns:
         str: The extracted modality string or empty string if an error occurs.
    """
    try:
        # Example: 15_FLAIR_whatever.nii.gz -> FLAIR
        # Example: P1_T1_FLAIR.nii.gz -> T1 (Might need adjustment based on dataset)
        return filename.split("_")[1]
    except IndexError:
        logging.warning(
            f"Default parser failed to extract modality from filename '{filename}' "
            f"(expected format like 'ID_MODALITY_...')."
        )
        return None


def msshift_output_filename_formatter(
    nifti_filename_base: str,
    slice_index: int,
    slice_dim: int,
    dataset_name: str,
    split: str,
    **kwargs: Any,
) -> str:
    """Formats output filename for MSShift dataset.

    Example:
    Input: nifti_filename_base='15_FLAIR_isovox', modality='FLAIR',
           slice_index=10, slice_dim=2, dataset_name='MSShift',
           split='train'.
    Output: 'MSShift_train_15_FLAIR_idx_10_axial'

    Args:
        nifti_filename_base (str): The base name of the original NIfTI file
                                   (without '.nii.gz').
        slice_index (int): The index of the slice within its dimension.
        slice_dim (int): The dimension along which the slice was taken
                         (0: saggital, 1: frontal, 2: axial).
        dataset_name (str): Name of the dataset (e.g., 'MSLesSeg').
        split (str): The dataset split ('train', 'val' or 'test').
        **kwargs (Any): Catches potential extra arguments from the caller.

    Returns:
        str: The formatted base name for the output file (without extension).
    """
    # Assumes the first part of the nifti base name is the patient ID
    patient_id = nifti_filename_base.split("_")[0]
    modality = nifti_filename_base.split("_")[1]
    slice_name = SLICE_NAMES.get(slice_dim, f"dim{slice_dim}")
    return (
        f"{dataset_name}_{split}_{patient_id}_{modality}_idx_{slice_index}_{slice_name}"
    )


# --- Specific Parsers/Formatters for Datasets ---


def mslesseg_filename_parser(filename: str, *args: Any, **kwargs: Any) -> Optional[str]:
    """Extracts modality from MSLesSeg filenames, considering split differences.

    - Train format: P1_T1_FLAIR.nii.gz -> Extracts 'FLAIR' (3rd part)
    - Test format: P1_FLAIR.nii.gz -> Extracts 'FLAIR' (2nd part)
    - Mask format: P1_T1_MASK.nii.gz -> Extracts 'MASK'

    Args:
        filename (str): The input NIfTI filename (basename).
        *args(Any): Catches potential extra arguments from the caller.
        **kwargs (Any): Catches potential extra arguments from the caller.

    Returns:
        str: The extracted modality string or empty string if an error occurs.
    """
    parts = filename.split(".")[0].split("_")
    try:
        return parts[-1]
    except Exception as e:
        logging.warning(f"MSLesSeg parser failed for filename '{filename}' : {e}")
        return ""


def mslesseg_output_filename_formatter(
    nifti_filename_base: str,
    slice_index: int,
    slice_dim: int,
    dataset_name: str,
    split: str,
    **kwargs: Any,
) -> str:
    """Formats output filename for MSLesSeg dataset.

    Example:
    Input: nifti_filename_base='P1_T1_FLAIR', modality='FLAIR',
        slice_index=10, slice_dim=2, dataset_name='MSLesSeg',
        split='train'.
    Output: 'MSLesSeg_train_P1_T1_FLAIR_idx_10_axial'

    Args:
        nifti_filename_base (str): Base name of the NIfTI file (e.g., 'P1_T1_FLAIR').
        slice_index (int): Slice index.
        slice_dim (int): Slice dimension index.
        dataset_name (str): Name of the dataset (e.g., 'MSLesSeg').
        split (str): The dataset split ('train', 'val' or 'test').
         **kwargs (Any): Catches potential extra arguments from the caller.

    Returns:
        str: Formatted output filename base.
    """
    slice_name = SLICE_NAMES.get(slice_dim, f"dim{slice_dim}")
    # Note: The original formatter used the output_dir basename. Replicating that.
    # It also used the full nifti_filename_base which includes timestamp for train.
    return (
        f"{dataset_name}_{split}_{nifti_filename_base}_idx_{slice_index}_{slice_name}"
    )


# --- Main Converter Class ---


class NiftiToYoloConverter:
    """Converts 3D NIfTI medical images into a 2D dataset for YOLO object detection.

    This class processes NIfTI files (.nii, .nii.gz) from specified modalities,
    extracts 2D slices along chosen dimensions, normalizes the images, finds
    bounding boxes from corresponding segmentation masks, and saves them in the
    standard YOLO format (images/split/*.png, labels/split/*.txt).
    Folder in data directory should have subfolders, named like splits:
    'train', 'test' or 'val' in lower case.

    Attributes:
        output_dir (str): The root directory where the YOLO dataset will be created.
        data_dir (str): The root directory containing the input NIfTI files,
                         expected to have subdirectories for each split (e.g., 'train', 'val').
        splits (List[str]): A list of dataset splits to process (e.g., ['train', 'val']).
        modalities (List[str]): A list of modality identifiers to include in the
                                 output dataset (e.g., ['FLAIR', 'T1', 'MASK']).
        mask_modality_name (str): The specific identifier used for segmentation mask files
                                  within the `modalities` list (e.g., 'MASK', 'gt').
        class_id (int): The class index to assign to detected objects in YOLO label files.
        slice_dims (List[int]): Dimensions along which to slice the 3D volumes
                                (0: saggital, 1: frontal, 2: axial).
        max_samples_per_split (Optional[int]): If set, limits the number of slices
                                               processed per split (useful for testing).
                                               Defaults to None (process all slices).
        filename_parser (Callable): A function to extract the modality from a NIfTI
                                    filename. It receives `filename` (str) and `split` (str).
                                    Defaults to `msshift_filename_parser`.
        output_filename_formatter (Callable): A function to format the base name for
                                              output image/label files. It receives
                                              `nifti_filename_base`, `modality`,
                                              `slice_index`, `slice_dim`, and potentially
                                              other kwargs like `dataset_name`,
                                              `output_dir_basename`. Defaults to
                                              `msshift_output_filename_formatter`.
        dataset_name (Optional[str]): An optional name for the dataset, which can be
                                      used by the `output_filename_formatter`.
        walk_filter (Optional[Callable]): An optional function to filter directories
                                          during `os.walk`. It receives `dirpath`,
                                          `dirnames`, `filenames` and should return
                                          `True` to process the directory, `False` to skip.
                                          Defaults to None (process all directories).
    """

    def __init__(
        self,
        output_dir: str,
        data_dir: str,
        mask_modality_name: str,
        modalities: List[str],
        class_id: int = 0,
        splits: List[str] = DEFAULT_SPLITS,
        slice_dims: List[int] = [2],  # Default to axial
        max_samples_per_split: Optional[int] = None,
        filename_parser: Callable[..., Optional[str]] = msshift_filename_parser,
        output_filename_formatter: Callable[
            ..., str
        ] = msshift_output_filename_formatter,
        dataset_name: Optional[str] = None,
        walk_filter: Optional[Callable[[str, List[str], List[str]], bool]] = None,
    ):
        # --- Input Validation ---
        if not os.path.isdir(data_dir):
            raise ValueError(f"Input data directory not found: {data_dir}")
        if not modalities:
            raise ValueError("`modalities` list cannot be empty.")
        if mask_modality_name not in modalities:
            raise ValueError(
                f"The `mask_modality_name` ('{mask_modality_name}') "
                f"must be present in the `modalities` list: {modalities}"
            )
        if not all(0 <= dim <= 2 for dim in slice_dims):
            raise ValueError("`slice_dims` must contain integers between 0 and 2.")
        if not isinstance(max_samples_per_split, (type(None), int)):
            raise ValueError("`max_samples_per_split` must be int or None.")
        if max_samples_per_split is not None and max_samples_per_split < 0:
            raise ValueError("`max_samples_per_split` cannot be negative.")
        if not callable(filename_parser):
            raise ValueError("`filename_parser` must be a callable function.")
        if not callable(output_filename_formatter):
            raise ValueError("`output_filename_formatter` must be a callable function.")
        if walk_filter is not None and not callable(walk_filter):
            raise ValueError("`walk_filter` must be None or a callable function.")

        # --- Assign Attributes ---
        self.output_dir = output_dir
        self.data_dir = data_dir
        self.splits = splits
        self.modalities = modalities
        self.mask_modality_name = mask_modality_name
        self.class_id = class_id
        self.slice_dims = slice_dims
        self.max_samples_per_split = max_samples_per_split
        self.filename_parser = filename_parser
        self.output_filename_formatter = output_filename_formatter
        self.dataset_name = dataset_name
        self.walk_filter = walk_filter

        self._slice_counter: Dict[str, int] = {
            split: 0 for split in splits
        }  # Track samples per split

        logging.info("Initialized NiftiToYoloConverter:")
        logging.info(f"  Input Data Dir: {self.data_dir}")
        logging.info(f"  Output Dir: {self.output_dir}")
        logging.info(f"  Splits: {self.splits}")
        logging.info(f"  Modalities: {self.modalities}")
        logging.info(f"  Mask Modality: {self.mask_modality_name}")
        logging.info(f"  Class ID: {self.class_id}")
        logging.info(
            f"  Slice Dimensions: {[SLICE_NAMES.get(d, d) for d in self.slice_dims]}"
        )
        logging.info(f"  Max Samples/Split: {self.max_samples_per_split}")
        logging.info(f"  Filename Parser: {self.filename_parser.__name__}")
        logging.info(f"  Output Formatter: {self.output_filename_formatter.__name__}")
        logging.info(f"  Dataset Name: {self.dataset_name}")
        logging.info(
            f"  Walk Filter: {self.walk_filter.__name__ if self.walk_filter else 'None'}"
        )

    def _create_directories(self) -> None:
        """Creates the necessary YOLO output directory structure."""
        logging.info("Creating output directories...")
        for folder in YOLO_FOLDERS:  # 'images', 'labels'
            for split in self.splits:  # 'train', 'val', 'test'
                make_dir(os.path.join(self.output_dir, folder, split))
        logging.info("Output directories created.")

    def _get_slice(
        self, img_3d: np.ndarray, slice_dim: int, slice_index: int
    ) -> np.ndarray:
        """Extracts a 2D slice from a 3D NumPy array based on dimension and index.

        Args:
            img_3d (np.ndarray): The 3D input image.
            slice_dim (int): The dimension along which to slice (0, 1, or 2).
            slice_index (int): The index of the slice within the dimension.

        Returns:
            np.ndarray: The extracted 2D slice.

        Raises:
            IndexError: If slice_index is out of bounds for the given slice_dim.
            ValueError: If slice_dim is not 0, 1, or 2.
        """
        if not 0 <= slice_dim <= 2:
            raise ValueError(f"slice_dim must be 0, 1, or 2, got {slice_dim}")

        if slice_dim == 0:  # Saggital
            slice_2d = img_3d[slice_index, :, :]
        elif slice_dim == 1:  # Frontal (Coronal)
            slice_2d = img_3d[:, slice_index, :]
        else:  # Axial
            slice_2d = img_3d[:, :, slice_index]

        return slice_2d

    def _process_nifti(
        self, nifti_filepath: str, split: str, modality: str, is_mask: bool
    ) -> None:
        """Loads a NIfTI file, extracts/saves slices as images or processes masks for labels.

        Args:
            nifti_filepath (str): Full path to the input NIfTI file.
            split (str): The current dataset split (e.g., 'train').
            modality (str): The modality of this NIfTI file.
            is_mask (bool): True if this file is a segmentation mask, False otherwise.
        """
        try:
            logging.debug(
                f"Processing NIfTI: {nifti_filepath} (Split: {split}, Modality: {modality}, IsMask: {is_mask})"
            )
            img_3d = nib.load(nifti_filepath).get_fdata(dtype=np.float32)
            logging.debug(f"  Loaded data shape: {img_3d.shape}")

            nifti_filename_base = os.path.basename(nifti_filepath).split(".nii")[0]

            for slice_dim in self.slice_dims:
                if not 0 <= slice_dim < img_3d.ndim:
                    logging.warning(
                        f"  Slice dimension {slice_dim} is invalid for image shape {img_3d.shape}. Skipping dim."
                    )
                    continue

                num_slices_in_dim = img_3d.shape[slice_dim]
                logging.debug(
                    f"  Processing dim {slice_dim} ({SLICE_NAMES.get(slice_dim)}): {num_slices_in_dim} slices"
                )

                for slice_index in range(num_slices_in_dim):
                    # Check sample limit for the current split
                    if (
                        self.max_samples_per_split is not None
                        and self._slice_counter[split] >= self.max_samples_per_split
                    ):
                        logging.debug(
                            f"  Reached max samples ({self.max_samples_per_split}) for split '{split}'. Stopping NIfTI processing."
                        )
                        return  # Stop processing this NIfTI file if limit reached for split

                    try:
                        img_2d = self._get_slice(img_3d, slice_dim, slice_index)
                    except IndexError:
                        logging.error(
                            f"Slice index {slice_index} out of bounds for dim {slice_dim} (shape {img_3d.shape}) in {nifti_filepath}"
                        )
                        continue  # Skip this slice

                    # --- Format Output Filename ---
                    formatter_kwargs = {
                        "nifti_filename_base": nifti_filename_base,
                        "slice_index": slice_index,
                        "slice_dim": slice_dim,
                        "split": split,
                        "dataset_name": self.dataset_name,
                    }
                    output_base_name = self.output_filename_formatter(
                        **formatter_kwargs
                    )

                    if is_mask:
                        # Process mask: Get bounding boxes
                        bboxes = get_bounding_boxes(img_2d, self.class_id)
                        if bboxes:  # Only save labels if objects are found
                            self._save_bboxes(bboxes, output_base_name, split)
                        # else: # Optional: log if mask slice is empty
                        # logging.debug(f"  No bounding boxes found in mask slice {slice_index} (dim {slice_dim}) for {nifti_filepath}")
                    else:
                        # Process image: Normalize and save
                        img_normalized = normalize_image(img_2d)
                        self._save_image(img_normalized, output_base_name, split)

                    # Increment counter only if an image/label was potentially saved
                    # Note: Even if bboxes is empty, we processed the slice.
                    self._slice_counter[split] += 1

        except FileNotFoundError:
            logging.error(f"NIfTI file not found: {nifti_filepath}")
        except nib.filebasedimages.ImageFileError as e:
            logging.error(f"Error loading NIfTI file {nifti_filepath}: {e}")
        except Exception as e:
            logging.error(
                f"Unexpected error processing {nifti_filepath}: {e}", exc_info=True
            )

    def _save_image(
        self, img_2d: np.ndarray, output_base_name: str, split: str
    ) -> None:
        """Saves a normalized 2D image slice in the appropriate YOLO directory.

        Converts the 1-channel grayscale image to a 3-channel image by repeating
        the channel, as expected by many YOLO implementations.

        Args:
            img_2d (np.ndarray): The 2D normalized image slice (np.uint8).
            output_base_name (str): The base name for the output file (e.g., 'ID_MOD_idx_SLICE_DIM').
            split (str): The dataset split (e.g., 'train').
        """
        filename = f"{output_base_name}.{FILE_EXTENSIONS['images']}"
        filepath = os.path.join(self.output_dir, YOLO_FOLDERS[0], split, filename)

        # Convert to 3-channel image (required by some YOLO models)
        img_3channel = (
            cv2.cvtColor(img_2d, cv2.COLOR_GRAY2BGR) if img_2d.ndim == 2 else img_2d
        )

        try:
            success = cv2.imwrite(filepath, img_3channel)
            if not success:
                raise IOError(f"cv2.imwrite failed for {filepath}")
            logging.debug(f"    Saved image: {filepath}")
        except Exception as e:
            logging.error(f"    Failed to save image {filepath}: {e}")

    def _save_bboxes(
        self,
        bboxes: List[Tuple[int, float, float, float, float]],
        output_base_name: str,
        split: str,
    ) -> None:
        """Saves bounding box information to a .txt file in YOLO format.

        The label file is saved in the 'labels' directory corresponding to the
        image file identified by `output_base_name`.

        Args:
            bboxes (List[Tuple[int, float, float, float, float]]): List of bounding
                boxes in YOLO format (class_id, x_center, y_center, width, height).
            output_base_name (str): The base name matching the corresponding image file.
            split (str): The dataset split (e.g., 'train').
        """
        if not bboxes:
            return

        filename = f"{output_base_name}.{FILE_EXTENSIONS['labels']}"
        filepath = os.path.join(self.output_dir, YOLO_FOLDERS[1], split, filename)

        try:
            for modality in self.modalities:
                #  we don't save bboxes for mask itself
                if modality != self.mask_modality_name:
                    with open(
                        filepath.replace(self.mask_modality_name, modality), "w"
                    ) as f:  # change to the current modality
                        for bbox in bboxes:
                            # Format: class_id x_center y_center width height
                            f.write(
                                f"{bbox[0]} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} {bbox[4]:.6f}\n"
                            )
                    logging.debug(f"    Saved labels ({len(bboxes)} boxes): {filepath}")
        except Exception as e:
            logging.error(f"    Failed to save labels to {filepath}: {e}")

    def create_dataset(self) -> None:
        """Main method to generate the YOLO dataset by processing all specified NIfTI files."""
        self._create_directories()
        self._slice_counter = {split: 0 for split in self.splits}  # Reset counters

        logging.info("Starting dataset creation process...")

        # Process each split. Folder in data dir should have subfolders, named like splits:
        # 'train', 'test' or 'val' in lower case
        for split in self.splits:
            split_data_dir = os.path.join(self.data_dir, split)
            if not os.path.isdir(split_data_dir):
                logging.warning(
                    f"Split directory not found, skipping: {split_data_dir}"
                )
                continue

            logging.info(f"Processing split: '{split}' in {split_data_dir}")
            file_count = 0

            # Use tqdm for progress bar over os.walk results
            walker = os.walk(split_data_dir)
            pbar = tqdm(walker, desc=f"Scanning {split}")

            for dirpath, dirnames, filenames in pbar:
                # Apply custom directory filter if provided
                if self.walk_filter and not self.walk_filter(
                    dirpath, dirnames, filenames
                ):
                    logging.debug(f"Skipping directory due to walk_filter: {dirpath}")
                    continue

                # Check if max samples for this split has been reached
                if (
                    self.max_samples_per_split is not None
                    and self._slice_counter[split] >= self.max_samples_per_split
                ):
                    logging.info(
                        f"Reached max samples ({self.max_samples_per_split}) for split '{split}'. Stopping walk."
                    )
                    break  # Stop walking this split

                for filename in filenames:
                    if filename.endswith((".nii", ".nii.gz")):
                        # Check again if max samples reached (might happen mid-directory)
                        if (
                            self.max_samples_per_split is not None
                            and self._slice_counter[split] >= self.max_samples_per_split
                        ):
                            break  # Stop processing files in this directory

                        modality = self.filename_parser(filename)

                        if modality is None:
                            logging.debug(
                                f"Skipping file: Modality not extracted from '{filename}'"
                            )
                            continue

                        if modality not in self.modalities:
                            logging.debug(
                                f"Skipping file: Modality '{modality}' not in specified list {self.modalities}"
                            )
                            continue

                        is_mask = modality == self.mask_modality_name
                        nifti_filepath = os.path.join(dirpath, filename)
                        file_count += 1
                        pbar.set_postfix(
                            {
                                "Files Found": file_count,
                                "Slices Gen.": self._slice_counter[split],
                            }
                        )

                        # Core processing function
                        self._process_nifti(nifti_filepath, split, modality, is_mask)

            logging.info(
                f"Finished processing split: '{split}'. Total slices generated: {self._slice_counter[split]}"
            )

        logging.info("Dataset creation process finished.")
        total_generated = sum(self._slice_counter.values())
        logging.info(f"Total slices generated across all splits: {total_generated}")


# --- Command Line Interface ---
def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert NIfTI datasets to YOLO format."
    )

    # Required Arguments
    parser.add_argument(
        "--data_dir",
        help="Root directory containing input NIfTI files (with split subdirs).",
    )
    parser.add_argument(
        "--output_dir",
        help="Directory where the YOLO dataset will be created.",
    )
    parser.add_argument(
        "--mask_modality",
        help="Identifier used for mask files (e.g., 'MASK', 'gt').",
    )
    parser.add_argument(
        "--modalities",
        help="List of all modalities to process, including the mask modality.",
    )

    # Optional Arguments
    parser.add_argument(
        "--class_id",
        type=int,
        default=0,
        help="Class ID for bounding boxes (default: 0).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=DEFAULT_SPLITS,
        help=f"List of splits to process (default: {DEFAULT_SPLITS}).",
    )
    parser.add_argument(
        "--slice_dims",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Dimensions to slice (0=sag, 1=cor, 2=ax, default: [2]).",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of slices per split (default: None - process all).",
    )
    parser.add_argument(
        "--dataset_name",
        default="",
        help="Optional dataset name for output formatting.",
    )
    parser.add_argument(
        "--config",
        default="msshift",
        choices=["mslesseg", "msshift"],
        help="Use predefined config for parser/formatter (default, mslesseg, ms_shift).",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")

    args = parser.parse_args()
    return args


# --- Example Usage ---

if __name__ == "__main__":
    args = parse_arguments()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # --- Select Configuration ---

    if args.config == "mslesseg":
        selected_parser = mslesseg_filename_parser
        selected_formatter = mslesseg_output_filename_formatter
        selected_walk_filter = None
        args.dataset_name = "MSLesSeg"
        args.modalities = MODALITIES_MSLesSeg
        args.mask_modality = MASK_MODALITY_MSLesSeg
        args.splits = ["train", "test"]

        logging.info(
            "Using MSLesSeg configuration for filename parsing and formatting."
        )
        # Example check for dataset name requirement
        if not args.dataset_name:
            logging.warning(
                "MSLesSeg formatter uses dataset_name, but --dataset_name was not provided."
            )

    elif args.config == "msshift":
        # Define the MS_Shift specific walk filter
        def ms_shift_walk_filter(
            dirpath: str, dirnames: List[str], filenames: List[str]
        ) -> bool:
            """Skips directories that contain subdirs or aren't numeric patient IDs."""
            base = os.path.basename(dirpath)
            parent_base = os.path.basename(os.path.dirname(dirpath))
            # Process only if the directory name is numeric (patient ID)
            # and it's directly under a split folder (e.g., data_dir/train/15)
            is_patient_dir = base.isnumeric()
            is_under_split = (
                parent_base in args.splits
            )  # Check if parent is a split dir
            # Simple check: Skip if it has subdirectories (often indicates higher levels)
            has_subdirs = len(dirnames) > 0
            should_process = is_patient_dir and is_under_split and not has_subdirs
            # print(f"Debug walk: path={dirpath}, base={base}, parent={parent_base}, is_patient={is_patient_dir}, is_under_split={is_under_split}, has_subdirs={has_subdirs}, process={should_process}")
            return should_process

        selected_parser = msshift_filename_parser  # Default works for MS_Shift (e.g., 00001_FLAIR.nii.gz)
        selected_formatter = msshift_output_filename_formatter  # Default works
        selected_walk_filter = ms_shift_walk_filter
        args.dataset_name = "MSShift"
        args.modalities = MODALITIES_MSShift
        args.mask_modality = MASK_MODALITY_MSShift
        logging.info(
            "Using MS_Shift configuration (default parser/formatter, specific walk filter)."
        )

    # --- Initialize and Run Converter ---
    try:
        creator = NiftiToYoloConverter(
            output_dir="msshift",  # "mslesseg",  # args.output_dir,
            data_dir="/home/kenzhin/Data/Multiple_Sclerosis/MS_shift",  # "/home/kenzhin/Data/Multiple_Sclerosis/MSLesSeg",   # args.data_dir,
            splits=args.splits,
            modalities=args.modalities,
            mask_modality_name=args.mask_modality,
            class_id=args.class_id,
            slice_dims=args.slice_dims,
            max_samples_per_split=args.max_samples,
            filename_parser=selected_parser,
            output_filename_formatter=selected_formatter,
            dataset_name=args.dataset_name,
            walk_filter=selected_walk_filter,
        )
        creator.create_dataset()
        logging.info("Script finished successfully.")

    except ValueError as ve:
        logging.error(f"Configuration Error: {ve}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)


# --- Example Command Lines ---
#
# python nifti_to_yolo_converter.py \
#   --data_dir /path/to/MSLesSegRaw \
#   --output_dir ./MSLesSeg_YOLO \
#   --mask_modality MASK \
#   --modalities FLAIR T1 T2 MASK \
#   --splits train test \
#   --slice_dims 0 1 2 \
#   --config mslesseg \
#   --dataset_name MSLesSeg \
#   --class_id 0
