"""
Provides classes and utilities for performing inference with YOLO models on 3D NIfTI
medical images by processing them slice-by-slice. Includes functionality for
batch processing and metric calculation against ground truth masks.
"""

import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import binary_fill_holes
from skimage import measure
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.utils.metrics import ConfusionMatrix

# --- Constants ---
SLICE_NAMES: Dict[int, str] = {0: "sagittal", 1: "coronal", 2: "axial"}
DEFAULT_CONF: float = 0.25
DEFAULT_IOU_NMS: float = 0.7
DEFAULT_IMGSZ: int = 640
DEFAULT_IOU_EVAL: float = 0.5  # Default for confusion matrix TP/FP matching
DEFAULT_BATCH_SIZE: int = 8
DEFAULT_NUM_CLASSES: int = 1  # Assuming single-class detection (lesion)
DEFAULT_GT_LABELS_DICT: Dict[int, int] = {1: 0}  # Map mask label 1 to YOLO class 0
DEFAULT_PRED_LABELS_DICT: Dict[int, int] = {
    0: 1
}  # Map YOLO class 0 to output mask value 1


# --- logger Configuration ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_filepath_inference_lists(
    data_dir: str,
    split_name: str,
    subject_ids: Iterable[Union[int, str]],
    gt_key: str = "gt",
    ignore_keys: Tuple[str, ...] = ("gt", "isovox"),
    nifti_formats: Tuple[str, ...] = (".nii", ".nii.gz"),
    filename_parts_separator: str = "_",
    modality_index: int = 1,
) -> Tuple[List[str], List[Optional[str]]]:
    """
    Creates parallel lists of NIfTI image file paths and their corresponding ground truth mask paths
    for a given dataset split and range of subjects within a specific directory structure.
    Assumes a structure like: data_dir / split_name / subject_id / subject_id_modality_suffix.nii.gz

    Args:
        data_dir (str): The root directory containing the dataset splits.
        split_name (str): The name of the dataset split (e.g., "Test", "Train").
        subject_ids (Iterable[Union[int, str]]): An iterable (like range or list) of subject IDs
                                                 to process. IDs will be converted to strings.
        gt_key (str, optional): The string identifier used in filenames to denote the
                                ground truth mask (e.g., "gt", "MASK"). Defaults to "gt".
        ignore_keys (Tuple[str, ...], optional): A tuple of strings found in filenames
                                                 (typically at the modality position) that should be
                                                 ignored when creating the image list. Defaults to ("gt", "isovox").
        nifti_formats (Tuple[str, ...], optional): Valid NIfTI file extensions. Defaults to (".nii", ".nii.gz").
        filename_parts_separator (str, optional): The character used to separate parts in the filename
                                                   (e.g., "_" in "subject_modality_suffix.nii.gz"). Defaults to "_".
        modality_index (int, optional): The index (0-based) of the modality part after splitting the
                                        filename base by the separator. Defaults to 1.

    Returns:
        Tuple[List[str], List[Optional[str]]]: A tuple containing two lists:
            - imgs_list: A list of full paths to the identified image NIfTI files.
            - gts_list: A list of full paths to the corresponding ground truth NIfTI files.

    Raises:
        FileNotFoundError: If the main data directory or split directory does not exist.
        ValueError: If no subject directories are found within the split.
    """
    split_path = Path(data_dir) / split_name
    if not split_path.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_path}")

    all_subject_dirs = {p.name for p in split_path.iterdir() if p.is_dir()}
    subject_ids_str = {str(idx) for idx in subject_ids}

    # Filter to existing directories matching the requested subject IDs
    valid_subject_ids = sorted(
        list(all_subject_dirs.intersection(subject_ids_str)),
        key=lambda x: int(x) if x.isdigit() else x,
    )

    if not valid_subject_ids:
        raise ValueError(
            f"No valid subject directories found for the specified IDs in {split_path}"
        )

    logger.info(
        f"Found {len(valid_subject_ids)} valid subject directories in {split_path}"
    )

    imgs_list_all: List[str] = []
    gts_list_all: List[Optional[str]] = []

    for subj_id in valid_subject_ids:
        current_dir = split_path / subj_id
        subject_imgs: List[str] = []
        subject_gt_path: Optional[str] = None

        try:
            if not current_dir.is_dir():
                logger.warning(
                    f"Subject directory {current_dir} not found or is not a directory. Skipping."
                )
                continue

            # List files efficiently once
            files_in_dir = [
                f
                for f in current_dir.iterdir()
                if f.is_file()
                and f.name.lower().endswith(nifti_formats)
                and not f.name.startswith(".")
            ]

            if not files_in_dir:
                logger.warning(
                    f"No NIfTI files found in {current_dir}. Skipping subject."
                )
                continue

            for file_path in files_in_dir:
                filename_base = file_path.stem.split(".")[0]  # Remove .nii.gz or .nii
                parts = filename_base.split(filename_parts_separator)

                if len(parts) > modality_index:
                    modality_part = parts[modality_index]
                    if modality_part == gt_key:
                        if subject_gt_path is None:
                            subject_gt_path = str(file_path)
                        else:
                            logger.warning(
                                f"Multiple GT files found for subject {subj_id}. Using: {subject_gt_path}"
                            )
                    elif modality_part not in ignore_keys:
                        subject_imgs.append(str(file_path))
                    # else: ignore the file based on ignore_keys
                else:
                    logger.warning(
                        f"Filename format unexpected for {file_path.name}. Cannot determine modality. Skipping."
                    )

            if not subject_imgs:
                logger.warning(
                    f"No valid image files found for subject {subj_id} (excluding ignored keys)."
                )
                continue

            # Extend the main lists
            imgs_list_all.extend(subject_imgs)
            # Duplicate the single GT path for each image file found for this subject
            # Or append None if no GT was found
            gts_list_all.extend([subject_gt_path] * len(subject_imgs))

            if subject_gt_path is None:
                logger.warning(
                    f"No ground truth file ('{gt_key}') found for subject {subj_id}."
                )

        except FileNotFoundError:
            logger.warning(
                f"Subject directory {current_dir} seems to have disappeared during processing. Skipping."
            )
        except Exception as e:
            logger.error(
                f"Error processing subject directory {current_dir}: {e}", exc_info=True
            )

    return imgs_list_all, gts_list_all


class Yolo3dInference:
    """
    Performs YOLO inference on a single 3D NIfTI volume by processing 2D slices.

    This class handles loading the NIfTI image (and optional ground truth mask),
    slicing the volume along specified dimensions, preprocessing slices for YOLO input,
    running the YOLO model predictions, extracting bounding boxes (both predicted and
    ground truth), calculating detection metrics (mAP via torchmetrics and confusion
    matrix via ultralytics), reconstructing a 3D segmentation mask from predicted boxes,
    and visualizing results.

    Attributes:
        yolo_model (YOLO): The loaded Ultralytics YOLO model instance.
        nifti_filepath (str): Path to the input NIfTI image.
        gt_filepath (Optional[str]): Path to the ground truth NIfTI mask (if provided).
        slice_dims (List[int]): Dimensions (0, 1, 2) along which to slice.
        conf (float): Confidence threshold for YOLO predictions.
        iou (float): IoU threshold for NMS during YOLO prediction.
        imgsz (int): Image size used for YOLO inference.
        eval_iou_thresholds (Optional[List[float]]): IoU thresholds for torchmetrics mAP.
        cf_iou_threshold (float): IoU threshold for ultralytics confusion matrix.
        num_classes (int): Number of detection classes.
        batch_size (int): Batch size used for prediction (if applicable by YOLO method).
        device (torch.device): Device used for inference (CPU or CUDA).
        gt_labels_dict (Dict[int, int]): Mapping from mask values to YOLO class IDs.
        pred_labels_dict (Dict[int, int]): Mapping from YOLO class IDs to output mask values.

        img_3d (Optional[np.ndarray]): Loaded 3D image data.
        gt_3d (Optional[np.ndarray]): Loaded 3D ground truth mask data.
        affine (Optional[np.ndarray]): Affine matrix from the loaded NIfTI image.
        header (Optional[nib.Nifti1Header]): Header from the loaded NIfTI image.
    """

    def __init__(
        self,
        yolo_model: YOLO,
        nifti_filepath: str,
        gt_filepath: Optional[str] = None,
        slice_dims: Union[int, List[int]] = 2,
        conf: float = DEFAULT_CONF,
        iou: float = DEFAULT_IOU_NMS,
        imgsz: int = DEFAULT_IMGSZ,
        eval_iou_thresholds: Optional[List[float]] = None,  # For mAP calculation
        cf_iou_threshold: float = DEFAULT_IOU_EVAL,  # For confusion matrix calculation
        num_classes: int = DEFAULT_NUM_CLASSES,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: Optional[Union[str, torch.device]] = None,
        gt_labels_dict: Dict[int, int] = DEFAULT_GT_LABELS_DICT,
        pred_labels_dict: Dict[int, int] = DEFAULT_PRED_LABELS_DICT,
        _load_data_on_init: bool = True,
    ) -> None:
        """
        Initializes the Yolo3DInference class.

        Args:
            yolo_model (YOLO): The pre-loaded Ultralytics YOLO model. Required.
            nifti_filepath (str): Path to the NIfTI image file (.nii.gz). Required.
            gt_filepath (Optional[str], optional): Path to the ground truth NIfTI mask file (.nii.gz).
                                                   Required for metric calculation and GT box extraction. Defaults to None.
            slice_dims (Union[int, List[int]], optional): Dimension(s) along which to slice the 3D image
                                                          (0: sagittal, 1: coronal, 2: axial).
                                                          Defaults to 2 (axial).
            conf (float, optional): Confidence threshold for detections. Defaults to DEFAULT_CONF.
            iou (float, optional): IoU threshold for Non-Maximum Suppression (NMS) in YOLO model prediction.
                                   Defaults to DEFAULT_IOU_NMS.
            imgsz (int, optional): Image size for inference. Defaults to DEFAULT_IMGSZ.
            eval_iou_thresholds (Optional[List[float]], optional): List of IoU thresholds for mAP evaluation (torchmetrics).
                                                                  If None, torchmetrics defaults are used (e.g., 0.5:0.05:0.95).
                                                                  Defaults to None.
            cf_iou_threshold (float, optional): IoU threshold for considering a prediction as TP in confusion matrix (ultralytics).
                                                Defaults to DEFAULT_IOU_EVAL.
            num_classes (int, optional): Number of detection classes. Defaults to DEFAULT_NUM_CLASSES.
            batch_size (int, optional): Batch size for YOLO prediction. Defaults to DEFAULT_BATCH_SIZE.
            device (Optional[Union[str, torch.device]], optional): Device for inference ('cpu', 'cuda', torch.device).
                                                                    Defaults to CUDA if available, else CPU.
            gt_labels_dict (Dict[int, int], optional): Mapping from ground truth mask values to YOLO class IDs.
                                                      Defaults to {1: 0}.
            pred_labels_dict (Dict[int, int], optional): Mapping from predicted YOLO class IDs to output mask values.
                                                       Defaults to {0: 1}.
            _load_data_on_init (bool, optional): If False, skips loading NIfTI data and path validation
                                               during initialization. Useful when used as a base class.
                                               Defaults to True.
        """

        if isinstance(slice_dims, int):
            slice_dims = [slice_dims]
        if not all(0 <= dim <= 2 for dim in slice_dims):
            raise ValueError("Slice_dims must contain integers between 0 and 2.")

        self.yolo_model = yolo_model
        self.nifti_filepath = nifti_filepath
        self.gt_filepath = gt_filepath
        self.slice_dims = slice_dims
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.eval_iou_thresholds = eval_iou_thresholds
        self.cf_iou_threshold = cf_iou_threshold
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.gt_labels_dict = gt_labels_dict
        self.pred_labels_dict = pred_labels_dict

        # Attributes to be populated by _load_data
        self.img_3d: Optional[np.ndarray] = None
        self.gt_3d: Optional[np.ndarray] = None
        self.affine: Optional[np.ndarray] = None
        self.header: Optional[nib.Nifti1Header] = None

        # Load data only if the flag is set and the paths are valid
        if _load_data_on_init:
            if not Path(nifti_filepath).is_file():
                raise FileNotFoundError(f"Input NIfTI file not found: {nifti_filepath}")
            if gt_filepath and not Path(gt_filepath).is_file():
                raise FileNotFoundError(
                    f"Ground truth NIfTI file not found: {gt_filepath}"
                )
            self._load_data()
        else:
            logger.info(
                "Skipping data loading during Yolo3dInference initialization (likely used as base class)."
            )

    def _load_data(self) -> None:
        """Loads the NIfTI image and optionally the ground truth mask."""
        try:
            nii_image = nib.load(self.nifti_filepath)
            self.img_3d = nii_image.get_fdata()
            self.affine = nii_image.affine
            self.header = nii_image.header
            logger.info(
                f"Loaded image: {self.nifti_filepath}, shape: {self.img_3d.shape}"
            )
        except Exception as e:
            logger.error(f"Error loading NIfTI image {self.nifti_filepath}: {e}")
            raise ValueError(f"Error loading NIfTI image: {e}") from e

        if self.gt_filepath:
            try:
                nii_gt = nib.load(self.gt_filepath)
                self.gt_3d = nii_gt.get_fdata().astype(
                    np.uint8
                )  # Ensure mask is integer type
                logger.info(
                    f"Loaded GT mask: {self.gt_filepath}, shape: {self.gt_3d.shape}"
                )
                if self.img_3d is not None and self.img_3d.shape != self.gt_3d.shape:
                    logger.warning("Image and GT mask shapes do not match!")
            except Exception as e:
                logger.error(f"Error loading NIfTI GT mask {self.gt_filepath}: {e}")
                # Decide if this should be fatal or just prevent metric calculation
                self.gt_3d = None  # Set to None if loading fails

    @staticmethod
    def get_slice(img_3d: np.ndarray, slice_dim: int, slice_index: int) -> np.ndarray:
        """
        Extracts a 2D slice from a 3D array.

        Args:
            img_3d (np.ndarray): The 3D image data.
            slice_dim (int): Dimension along which to slice (0, 1, or 2).
            slice_index (int): Index of the slice to extract.

        Returns:
            np.ndarray: The extracted 2D slice..
        """
        if not (0 <= slice_dim <= 2):
            raise ValueError(f"Invalid slice_dim: {slice_dim}. Must be 0, 1, or 2.")
        if not (0 <= slice_index < img_3d.shape[slice_dim]):
            raise IndexError(
                f"slice_index {slice_index} out of bounds for dimension {slice_dim} with shape {img_3d.shape}"
            )

        if slice_dim == 0:
            return img_3d[slice_index, :, :]
        elif slice_dim == 1:
            return img_3d[:, slice_index, :]
        else:
            return img_3d[:, :, slice_index]

    @staticmethod
    def preprocess_slice(img_2d: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """
        Normalizes a 2D slice to [0, 255] uint8 and converts to 3-channel BGR format for YOLO.

        Args:
            img_2d (np.ndarray): The raw 2D image slice.
            eps (float, optional): Small value for numerical stability during normalization. Defaults to 1e-6.

        Returns:
            np.ndarray: The preprocessed 3-channel (H, W, 3) BGR image as uint8.
        """
        min_val, max_val = np.min(img_2d), np.max(img_2d)
        if max_val == min_val:
            normalized_slice = np.zeros(img_2d.shape, dtype=np.uint8)
        else:
            normalized_slice = (
                (img_2d - min_val) / (max_val - min_val + eps) * 255
            ).astype(np.uint8)

        # Convert grayscale to 3-channel BGR (standard for OpenCV/YOLO)
        slice_bgr = cv2.cvtColor(normalized_slice, cv2.COLOR_GRAY2BGR)
        return slice_bgr

    def _preprocess_img_3d(self, slice_dim: int) -> List[np.ndarray]:
        """
        Preprocesses all 2D slices along a specified dimension of the 3D image.

        Args:
            slice_dim (int): Dimension along which to slice (0, 1, or 2).

        Returns:
            List[np.ndarray]: A list of preprocessed 2D slices (H, W, 3) ready for YOLO input.
                              Returns empty list if img_3d is None.
        """
        if self.img_3d is None:
            logger.error("3D image data is not loaded. Cannot preprocess.")
            return []

        num_slices = self.img_3d.shape[slice_dim]
        return [
            self.preprocess_slice(self.get_slice(self.img_3d, slice_dim, i))
            for i in range(num_slices)
        ]

    def predict_2d_boxes(self) -> np.ndarray:
        """
        Runs YOLO prediction on all specified 2D slices of the 3D NIfTI image.
        Iterates through `self.slice_dims`, preprocesses slices for each dimension,
        and runs `self.yolo_model.predict`. Collects all detected bounding boxes.

        Returns:
            np.ndarray: An array of predicted bounding boxes across all processed slices and dimensions.
                        Format: [x1, y1, x2, y2, slice_idx, slice_dim, class_id, score].
                        Returns an empty array if no predictions are made or img_3d is None.
                        Coordinates (x1, y1, x2, y2) are in pixel space of the *original slice*.
        """
        if self.img_3d is None:
            logger.error("3D image data is not loaded. Cannot run prediction.")
            return np.empty((0, 8))

        predicted_boxes_all = []

        for slice_dim in self.slice_dims:
            img_slices = self._preprocess_img_3d(slice_dim)

            if not img_slices:
                logger.warning(
                    f"No slices generated for dimension {slice_dim}. Skipping prediction."
                )
                continue

            logger.info(
                f"Running YOLO prediction on {len(img_slices)} slices for dimension {slice_dim} ({SLICE_NAMES[slice_dim]})..."
            )

            for slice_idx, slice_img in tqdm(
                enumerate(img_slices),
                desc=f"Prediction for dimension {slice_dim}",
                total=len(img_slices),
            ):
                try:
                    results_list = self.yolo_model.predict(
                        slice_img,
                        conf=self.conf,
                        iou=self.iou,
                        imgsz=self.imgsz,
                        batch=self.batch_size,
                        verbose=False,  # Reduce console output
                        device=self.device,
                    )
                except Exception as e:
                    logger.error(
                        f"Error during YOLO prediction for dim {slice_dim}: {e}",
                        exc_info=True,
                    )
                    continue

                for result in results_list:
                    for box in result.boxes:
                        x1, y1, x2, y2 = (
                            box.xyxy[0].cpu().numpy()
                        )  # Pixel coordinates on the (potentially resized) input slice
                        score = float(box.conf[0])
                        class_id = int(box.cls[0])
                        predicted_boxes_all.append(
                            [x1, y1, x2, y2, slice_idx, slice_dim, class_id, score]
                        )

                        if not predicted_boxes_all:
                            logger.info(
                                "No bounding boxes predicted across any slices/dimensions."
                            )
                            return np.empty((0, 8))
                        # else:
                        #     logger.info(
                        #         f"Predicted {len(predicted_boxes_all)} bounding boxes in total."
                        #     )
        return np.array(predicted_boxes_all)

    def extract_gt_boxes_from_mask(self) -> np.ndarray:
        """
        Extracts ground truth bounding boxes from the loaded 3D segmentation mask.
        Iterates through specified `self.slice_dims`, extracts 2D mask slices, finds connected
        components using `skimage.measure.regionprops`, determines the corresponding YOLO class ID
        using `self.gt_labels_dict`, and returns the boxes in pixel coordinates.

        Returns:
            np.ndarray: An array of ground truth bounding boxes.
                        Format: [x1, y1, x2, y2, slice_idx, slice_dim, class_id].
                        Returns an empty array (shape (0, 7)) if gt_3d is None or no boxes are found.
                        Coordinates (x1, y1, x2, y2) are in pixel space of the original slice.
        """
        if self.gt_3d is None:
            logger.warning("Ground truth mask not loaded. Cannot extract GT boxes.")
            return np.empty((0, 7))

        gt_boxes_all: List[List[int]] = []

        for slice_dim in self.slice_dims:
            num_slices = self.gt_3d.shape[slice_dim]
            for slice_idx in range(num_slices):
                mask_2d = self.get_slice(self.gt_3d, slice_dim, slice_idx)

                if mask_2d.max() == 0:  # Skip slices with no lesions
                    continue

                labels_map = measure.label(mask_2d, connectivity=2)
                properties = measure.regionprops(labels_map)

                for prop in properties:
                    # regionprops bbox is (min_row, min_col, max_row, max_col)
                    min_r, min_c, max_r, max_c = prop.bbox
                    x1, y1 = min_c, min_r
                    x2, y2 = max_c, max_r

                    # Determine the class label from the mask value within the bbox
                    # Take the most frequent non-zero label within the object pixels
                    coords = prop.coords  # Array of (row, col) pixel coordinates
                    mask_values = mask_2d[coords[:, 0], coords[:, 1]]
                    non_zero_mask_values = mask_values[mask_values > 0]

                    if non_zero_mask_values.size > 0:
                        # Find the most frequent non-zero label value within the region
                        original_class = np.bincount(non_zero_mask_values).argmax()
                        yolo_class_id = self.gt_labels_dict.get(original_class)

                        if yolo_class_id is not None:
                            gt_boxes_all.append(
                                [x1, y1, x2, y2, slice_idx, slice_dim, yolo_class_id]
                            )
                        else:
                            logger.info(
                                f"Mask value {original_class} not found in gt_labels_dict. Skipping bbox."
                            )
                    else:
                        logger.info(
                            f"Region properties found but no non-zero mask values within coords at slice {slice_idx}, dim {slice_dim}."
                        )

        if not gt_boxes_all:
            logger.info("No ground truth bounding boxes extracted from the mask.")
            return np.empty((0, 7))
        else:
            logger.info(f"Extracted {len(gt_boxes_all)} ground truth bounding boxes.")
            return np.array(gt_boxes_all)

    @staticmethod
    def _group_boxes_by_slice(
        pred_boxes: np.ndarray, gt_boxes: np.ndarray
    ) -> Tuple[
        List[Dict[str, torch.Tensor]],
        List[Dict[str, torch.Tensor]],
        List[Tuple[int, int]],
    ]:
        """
        Groups predicted and ground truth boxes by (slice_idx, slice_dim).
        This helper function ensures that predictions and ground truths are aligned on a
        per-slice basis, which is required by evaluation metrics like torchmetrics mAP
        and ultralytics ConfusionMatrix. It handles cases where a slice might have only
        predictions, only ground truths, or neither, by adding appropriate empty tensors.

        Args:
            pred_boxes (np.ndarray): Predicted boxes [x1, y1, x2, y2, slice_idx, slice_dim, class_id, score].
            gt_boxes (np.ndarray): Ground truth boxes [x1, y1, x2, y2, slice_idx, slice_dim, class_id].

        Returns:
            Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]], List[Tuple[int, int]]]:
                - preds_grouped (List[Dict]): A list where each element is a dictionary representing
                  predictions for a single slice, formatted as required by torchmetrics:
                  {'boxes': Tensor[N, 4], 'scores': Tensor[N], 'labels': Tensor[N]}.
                - gts_grouped (List[Dict]): A list where each element is a dictionary representing
                  ground truths for a single slice, formatted as required by torchmetrics:
                  {'boxes': Tensor[M, 4], 'labels': Tensor[M]}.
                - all_slice_keys (List[Tuple[int, int]]): A sorted list of unique (slice_idx, slice_dim)
                  tuples corresponding to the elements in `preds_grouped` and `gts_grouped`.
        """
        pred_dict = defaultdict(lambda: {"boxes": [], "scores": [], "labels": []})
        gt_dict = defaultdict(lambda: {"boxes": [], "labels": []})

        # Populate dictionaries using (slice_idx, slice_dim) as keys
        if pred_boxes.size > 0:
            for box in pred_boxes:
                x1, y1, x2, y2, slice_idx, slice_dim, class_id, score = box
                key = (int(slice_idx), int(slice_dim))
                pred_dict[key]["boxes"].append([x1, y1, x2, y2])
                pred_dict[key]["scores"].append(score)
                pred_dict[key]["labels"].append(int(class_id))

        if gt_boxes.size > 0:
            for box in gt_boxes:
                x1, y1, x2, y2, slice_idx, slice_dim, class_id = box
                key = (int(slice_idx), int(slice_dim))
                gt_dict[key]["boxes"].append([x1, y1, x2, y2])
                gt_dict[key]["labels"].append(int(class_id))

        # Ensure keys are sorted for consistent ordering
        all_slice_keys = sorted(list(set(pred_dict.keys()) | set(gt_dict.keys())))

        preds_grouped = []
        gts_grouped = []

        # Iterate through all unique slices to create paired lists
        for key in all_slice_keys:
            # Predictions for the current slice
            if key in pred_dict:
                preds_grouped.append(
                    {
                        "boxes": torch.tensor(
                            pred_dict[key]["boxes"], dtype=torch.float32
                        ),
                        "scores": torch.tensor(
                            pred_dict[key]["scores"], dtype=torch.float32
                        ),
                        "labels": torch.tensor(
                            pred_dict[key]["labels"], dtype=torch.int32
                        ),
                    }
                )
            else:  # Add empty tensor if no predictions for this slice (needed for torchmetrics)
                preds_grouped.append(
                    {
                        "boxes": torch.empty((0, 4), dtype=torch.float32),
                        "scores": torch.empty(0, dtype=torch.float32),
                        "labels": torch.empty(0, dtype=torch.int32),
                    }
                )

            # Ground truths for the current slice
            if key in gt_dict:
                gts_grouped.append(
                    {
                        "boxes": torch.tensor(
                            gt_dict[key]["boxes"], dtype=torch.float32
                        ),
                        "labels": torch.tensor(
                            gt_dict[key]["labels"], dtype=torch.int32
                        ),
                    }
                )
            else:  # Add empty tensor if no GT for this slice
                gts_grouped.append(
                    {
                        "boxes": torch.empty((0, 4), dtype=torch.float32),
                        "labels": torch.empty(0, dtype=torch.int32),
                    }
                )

        return preds_grouped, gts_grouped, all_slice_keys

    def compute_map_torchmetrics(
        self, pred_boxes: np.ndarray, gt_boxes: np.ndarray
    ) -> Optional[Dict]:
        """
        Computes detection metrics (mAP, precision, recall) using torchmetrics.
        Groups predictions and ground truths by slice before calculating metrics.

        Args:
            pred_boxes (np.ndarray): Predicted boxes [x1, y1, x2, y2, slice_idx, slice_dim, class_id, score].
            gt_boxes (np.ndarray): Ground truth boxes [x1, y1, x2, y2, slice_idx, slice_dim, class_id].

        Returns:
            Optional[Dict]: Dictionary containing computed mAP metrics, or None if torchmetrics
                            is not installed or if input arrays are invalid.
        """
        try:
            from torchmetrics.detection import MeanAveragePrecision
        except ImportError:
            logger.error(
                "torchmetrics not installed. Please install with 'pip install torchmetrics'. Cannot compute mAP."
            )
            return None

        # Group boxes by slice for metric calculation
        try:
            preds_grouped, gts_grouped, _ = self._group_boxes_by_slice(
                pred_boxes, gt_boxes
            )
        except ValueError as e:
            logger.error(
                f"Error grouping boxes: {e}. Cannot compute mAP.", exc_info=True
            )
            return None

        try:
            metric = MeanAveragePrecision(
                iou_type="bbox",
                iou_thresholds=self.eval_iou_thresholds,  # Uses default 0.5:0.05:0.95 if None
                class_metrics=True,  # Calculate metrics per class
            )
            metric.update(preds_grouped, gts_grouped)
            results = metric.compute()
            logger.info(
                f"Torchmetrics mAP computed: map_50={results.get('map_50', 'N/A'):.4f}, map_75={results.get('map_75', 'N/A'):.4f}"
            )
            return results
        except Exception as e:
            logger.error(f"Error computing torchmetrics mAP: {e}", exc_info=True)
            return None

    def compute_confusion_matrix_ultralytics(
        self, pred_boxes: np.ndarray, gt_boxes: np.ndarray
    ) -> Optional[ConfusionMatrix]:
        """
        Computes the confusion matrix using ultralytics utilities.
        Processes predictions and ground truths slice by slice.

        Args:
            pred_boxes (np.ndarray): Predicted boxes [x1, y1, x2, y2, slice_idx, slice_dim, class_id, score].
            gt_boxes (np.ndarray): Ground truth boxes [x1, y1, x2, y2, slice_idx, slice_dim, class_id].

        Returns:
            Optional[ConfusionMatrix]: The computed ultralytics ConfusionMatrix object,
                                       or None if ultralytics is not installed or inputs are invalid.
        """
        try:
            from ultralytics.utils.metrics import ConfusionMatrix
        except ImportError:
            logger.error(
                "ultralytics not installed correctly or metrics module moved. Please install/update. Cannot compute confusion matrix."
            )
            return None

        if gt_boxes.size == 0 and pred_boxes.size == 0:
            logger.info(
                "Both GT and prediction boxes are empty. Returning empty confusion matrix."
            )
            # Return an empty initialized matrix
            return ConfusionMatrix(
                nc=self.num_classes, conf=self.conf, iou_thres=self.cf_iou_threshold
            )
        # Note: Ultralytics ConfusionMatrix handles empty predictions if GTs exist, and vice-versa.

        # Group boxes for processing
        try:
            preds_grouped, gts_grouped, _ = self._group_boxes_by_slice(
                pred_boxes, gt_boxes
            )
        except ValueError as e:
            logger.error(
                f"Error grouping boxes: {e}. Cannot compute confusion matrix.",
                exc_info=True,
            )
            return None

        try:
            confusion_matrix = ConfusionMatrix(
                nc=self.num_classes,
                conf=self.conf,
                iou_thres=self.cf_iou_threshold,
                task="detect",
            )

            for pred, gt in zip(preds_grouped, gts_grouped):
                if (
                    pred["boxes"].numel() > 0
                ):  # Only process if predictions exist for the slice
                    # Format predictions for process_batch: tensor [N, 6] with (x1, y1, x2, y2, score, class)
                    detections = torch.cat(
                        (
                            pred["boxes"],
                            pred["scores"].unsqueeze(1),
                            pred["labels"]
                            .unsqueeze(1)
                            .float(),  # Needs to be float for cat
                        ),
                        dim=1,
                    )
                else:
                    detections = torch.empty(
                        (0, 6), dtype=torch.float32
                    )  # Pass empty tensor if no preds

                # process_batch expects dets, gt_bboxes, gt_cls
                confusion_matrix.process_batch(detections, gt["boxes"], gt["labels"])

            logger.info("Ultralytics confusion matrix computed.")
            return confusion_matrix
        except Exception as e:
            logger.error(
                f"Error computing ultralytics confusion matrix: {e}", exc_info=True
            )
            return None

    @staticmethod
    def calculate_precision_recall_f1_from_cf(
        confusion_matrix: np.ndarray, eps: float = 1e-6
    ) -> Tuple[float, float, float]:
        """
        Calculates precision, recall, and F1 score for the primary class (class 0)
        from a 2x2 or larger confusion matrix (typically from ultralytics).

        Assumes matrix format:
            [[TP, FP],
             [FN, TN]]
        Background class predictions are often ignored in detection confusion matrices (TN=0).
        Ultralytics matrix usually has format [N+1, N+1] where last row/col is background.
        We focus on the top-left block for the target class(es). For single class (0),
        we use matrix[0,0] for TP, matrix[0,1] for FP, matrix[1,0] for FN.

        Args:
            confusion_matrix (np.ndarray): The [N+1, N+1] confusion matrix from ultralytics.
            eps (float, optional): Small value for numerical stability. Defaults to 1e-6.

        Returns:
            Tuple[float, float, float]: (precision, recall, f1_score) for class 0.
        """
        if not isinstance(confusion_matrix, np.ndarray):
            logger.error(
                f"Input must be a numpy array, but got {type(confusion_matrix)}. Returning zeros."
            )
            return 0.0, 0.0, 0.0

        if confusion_matrix.shape[0] < 2 or confusion_matrix.shape[1] < 2:
            logger.warning(
                f"Confusion matrix shape {confusion_matrix.shape} is too small for calculation. Returning zeros."
            )
            return 0.0, 0.0, 0.0

        tp = confusion_matrix[0, 0]
        fp = confusion_matrix[0, 1]
        fn = confusion_matrix[1, 0]

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1_score = 2 * (precision * recall) / (precision + recall + eps)

        return float(precision), float(recall), float(f1_score)

    def create_predicted_3d_box_mask(
        self,
        bboxes: np.ndarray,
        save_to_nifti: bool = False,
        output_filename: str = "predicted_mask.nii.gz",
        fill_holes: bool = True,
    ) -> Optional[np.ndarray]:
        """
        Creates a 3D voxel mask from predicted bounding boxes across all dimensions.

        Bounding boxes are treated as cuboids aligned with the slice dimension.
        Voxel values are determined by the predicted class ID mapped through `pred_labels_dict`.

        Args:
            bboxes (np.ndarray): Predicted boxes [x1, y1, x2, y2, slice_idx, slice_dim, class_id, score].
                                 Coordinates must be in integer pixel space.
            save_to_nifti (bool, optional): Whether to save the resulting mask as a NIfTI file. Defaults to False.
            output_filename (str, optional): Filename for the saved NIfTI mask. Defaults to "predicted_mask.nii.gz".
            fill_holes (bool, optional): Whether to fill holes in the resulting binary mask using scipy.ndimage.binary_fill_holes.
                                         Defaults to True.

        Returns:
            Optional[np.ndarray]: The 3D voxel mask (uint8), or None if img_3d is not loaded.
        """
        if self.img_3d is None:
            logger.error("3D image data is not loaded. Cannot create prediction mask.")
            return None
        if bboxes.size == 0:
            logger.info("No bounding boxes provided. Returning empty mask.")
            return np.zeros_like(self.img_3d, dtype=np.uint8)

        voxel_mask = np.zeros_like(self.img_3d, dtype=np.uint8)

        # Ensure integer types for indexing
        try:
            x1_coords = bboxes[:, 0].astype(int)
            y1_coords = bboxes[:, 1].astype(int)
            x2_coords = bboxes[:, 2].astype(int)
            y2_coords = bboxes[:, 3].astype(int)
            slice_indices = bboxes[:, 4].astype(int)
            slice_dims = bboxes[:, 5].astype(int)
            pred_class_ids = bboxes[:, 6].astype(int)
        except IndexError:
            logger.error(
                "Bounding box array has incorrect shape/columns. Cannot create mask.",
                exc_info=True,
            )
            return np.zeros_like(
                self.img_3d, dtype=np.uint8
            )  # Return empty mask on error

        logger.info(f"Creating 3D mask from {len(bboxes)} bounding boxes...")
        for i in range(len(bboxes)):
            mask_value = self.pred_labels_dict.get(pred_class_ids[i])
            if (
                mask_value is None or mask_value == 0
            ):  # Skip if class not in dict or maps to background
                continue

            x1, y1, x2, y2 = x1_coords[i], y1_coords[i], x2_coords[i], y2_coords[i]
            slice_idx = slice_indices[i]
            slice_dim = slice_dims[i]

            # Ensure coordinates are within bounds of the 3D image shape
            max_x, max_y, max_z = self.img_3d.shape
            # Clip coordinates to be within image bounds
            x1, y1 = max(0, x1), max(0, y1)

            # Bbox coords (x1,y1,x2,y2) correspond to (min_col, min_row, max_col, max_row) of the slice.
            min_c, min_r, max_c, max_r = x1, y1, x2, y2

            try:
                if slice_dim == 0:  # Sagittal plane (Y, Z)
                    dim0_max, dim1_max = self.img_3d.shape[1], self.img_3d.shape[2]
                    min_r, max_r = max(0, min_r), min(dim0_max, max_r)
                    min_c, max_c = max(0, min_c), min(dim1_max, max_c)
                    if max_r > min_r and max_c > min_c:
                        voxel_mask[slice_idx, min_r:max_r, min_c:max_c] = mask_value
                elif slice_dim == 1:  # Coronal plane (X, Z)
                    dim0_max, dim1_max = self.img_3d.shape[0], self.img_3d.shape[2]
                    min_r, max_r = (
                        max(0, min_r),
                        min(dim0_max, max_r),
                    )  # corresponds to X
                    min_c, max_c = (
                        max(0, min_c),
                        min(dim1_max, max_c),
                    )  # corresponds to Z
                    if max_r > min_r and max_c > min_c:
                        voxel_mask[min_r:max_r, slice_idx, min_c:max_c] = mask_value
                elif slice_dim == 2:  # Axial plane (X, Y)
                    dim0_max, dim1_max = self.img_3d.shape[0], self.img_3d.shape[1]
                    min_r, max_r = (
                        max(0, min_r),
                        min(dim0_max, max_r),
                    )  # corresponds to X
                    min_c, max_c = (
                        max(0, min_c),
                        min(dim1_max, max_c),
                    )  # corresponds to Y
                    if max_r > min_r and max_c > min_c:
                        voxel_mask[min_r:max_r, min_c:max_c, slice_idx] = mask_value
            except IndexError as ie:
                logger.warning(
                    f"IndexError while creating mask for bbox {i} (coords {x1, y1, x2, y2}, slice {slice_idx}, dim {slice_dim}): {ie}"
                )
                continue  # Skip this bbox if indexing fails

        if fill_holes:
            try:
                # binary_fill_holes works on binary masks. Apply per label value if needed,
                # or just on the whole mask if only one label value (e.g., 1) is used.
                # Assuming mask_value is typically 1.
                binary_mask = voxel_mask > 0
                binary_fill_holes(binary_mask)
            except Exception as e:
                logger.warning(f"Could not perform hole filling: {e}")

        if save_to_nifti:
            if self.affine is not None and self.header is not None:
                try:
                    predicted_nifti = nib.Nifti1Image(
                        voxel_mask, self.affine, self.header
                    )
                    output_dir = os.path.dirname(output_filename)
                    if output_dir:
                        os.makedirs(output_dir, exist_ok=True)
                    nib.save(predicted_nifti, output_filename)
                    logger.info(f"Saved predicted mask to: {output_filename}")
                except Exception as e:
                    logger.error(
                        f"Error saving predicted mask NIfTI to '{output_filename}': {e}"
                    )
            else:
                logger.warning(
                    "Affine or Header information missing. Cannot save mask as NIfTI."
                )

        return voxel_mask

    def visualize_slice_with_bboxes(
        self,
        slice_idx: int,
        slice_dim: int,
        bboxes: np.ndarray,
        box_format: str = "pred",  # 'pred' or 'gt'
        show_scores: bool = False,
        preprocess: bool = True,
        score_threshold: float = 0.0,
        gt_bboxes: Optional[np.ndarray] = None,  # Optional GT boxes for comparison
        pred_color: str = "r",
        gt_color: str = "g",
        fontsize: int = 4,
        title_suffix: str = "",
        filename_to_save: Optional[str] = None,
    ) -> None:
        """
        Visualizes a specific 2D slice from the loaded 3D image with overlayed bounding boxes.
        Can display predicted boxes, ground truth boxes, or both.

        Args:
            slice_idx (int): Index of the slice to visualize within the specified dimension.
            slice_dim (int): Dimension along which the slice was taken (0, 1, or 2).
            bboxes (np.ndarray): Primary bounding boxes to display (usually predictions).
                                 Format depends on `box_format`:
                                 - 'pred': [x1, y1, x2, y2, s_idx, s_dim, class, score] (pixel coords)
                                 - 'gt': [x1, y1, x2, y2, s_idx, s_dim, class] (pixel coords)
            box_format (str, optional): Specifies the format of the primary `bboxes` array.
                                       Either 'pred' (includes score) or 'gt'. Defaults to 'pred'.
            show_scores (bool, optional): If True and `box_format` is 'pred', displays confidence scores
                                          next to the predicted boxes. Defaults to False.
            preprocess (bool, optional): Whether to preprocess the slice (notmalize). Defaults to True.
            score_threshold (float, optional): Minimum score for a predicted box to be displayed.
                                               Only active if `box_format` is 'pred'. Defaults to 0.0.
            gt_bboxes (Optional[np.ndarray], optional): Secondary set of bounding boxes (usually ground truth)
                                                       to display for comparison, using `gt_color`.
                                                       Format: [x1, y1, x2, y2, s_idx, s_dim, class].
                                                       Defaults to None.
            pred_color (str, optional): Color for the primary bounding boxes. Defaults to 'r'.
            gt_color (str, optional): Color for the secondary (ground truth) bounding boxes. Defaults to 'g'.
            fontsize (int, optional): Font size for score text. Defaults to 8.
            title_suffix (str, optional): Additional text to append to the plot title. Defaults to "".
            filename_to_save (Optional[str], optional): Path to save the plot image. If None, shows the plot.
                                                        Defaults to None.
        """
        if self.img_3d is None:
            logger.error("3D image data not loaded. Cannot visualize slice.")
            return
        if not (0 <= slice_dim <= 2):
            logger.error(f"Invalid slice_dim: {slice_dim}.")
            return
        if not (0 <= slice_idx < self.img_3d.shape[slice_dim]):
            logger.error(
                f"Slice index {slice_idx} out of bounds for dim {slice_dim} (max: {self.img_3d.shape[slice_dim] - 1})."
            )
            return

        slice_img_raw = self.get_slice(self.img_3d, slice_dim, slice_idx)
        # Preprocess for consistent display (normalize, 3-channel BGR)
        if preprocess:
            slice_img_display = self.preprocess_slice(slice_img_raw)

        fig, ax = plt.subplots(1, figsize=(8, 8))
        ax.imshow(slice_img_display)
        # ax.axis("off")

        plot_title = f"Slice {slice_idx} ({SLICE_NAMES[slice_dim]})"
        if title_suffix:
            plot_title += f" - {title_suffix}"
        ax.set_title(plot_title)

        # --- Draw Primary Bounding Boxes ---
        if bboxes.size > 0:
            # Filter boxes for the current slice and dimension
            slice_boxes = bboxes[
                (bboxes[:, 4] == slice_idx) & (bboxes[:, 5] == slice_dim)
            ]

            for box in slice_boxes:
                try:
                    x1, y1, x2, y2 = box[:4].astype(int)
                    width = x2 - x1
                    height = y2 - y1

                    if width <= 0 or height <= 0:
                        continue  # Skip invalid boxes

                    score = box[7] if box_format == "pred" and len(box) > 7 else None

                    if score is not None and score < score_threshold:
                        continue  # Skip boxes below threshold

                    rect = patches.Rectangle(
                        (x1, y1),
                        width,
                        height,
                        linewidth=1.5,
                        edgecolor=pred_color,
                        facecolor="none",
                    )
                    ax.add_patch(rect)

                    if show_scores and score is not None:
                        ax.text(
                            x1,
                            y1 - 5,
                            f"{score:.2f}",
                            color="white",
                            backgroundcolor=pred_color,
                            fontsize=fontsize,
                            bbox=dict(
                                boxstyle="round,pad=0.2", fc=pred_color, alpha=0.8
                            ),  # Nicer background
                        )
                except IndexError:
                    logger.warning(f"Could not unpack primary bbox data: {box}")
                except Exception as e:
                    logger.error(f"Error drawing primary bbox {box}: {e}")

        # --- Draw Secondary (Ground Truth) Bounding Boxes ---
        if gt_bboxes is not None and gt_bboxes.size > 0:
            # Filter boxes for the current slice and dimension
            gt_slice_boxes = gt_bboxes[
                (gt_bboxes[:, 4] == slice_idx) & (gt_bboxes[:, 5] == slice_dim)
            ]

            for gt_box in gt_slice_boxes:
                try:
                    x1, y1, x2, y2 = gt_box[:4].astype(int)
                    width = x2 - x1
                    height = y2 - y1

                    if width <= 0 or height <= 0:
                        continue  # Skip invalid boxes

                    rect = patches.Rectangle(
                        (x1, y1),
                        width,
                        height,
                        linewidth=1,
                        linestyle="--",
                        edgecolor=gt_color,
                        facecolor="none",
                    )
                    ax.add_patch(rect)
                except IndexError:
                    logger.warning(f"Could not unpack GT bbox data: {gt_box}")
                except Exception as e:
                    logger.error(f"Error drawing GT bbox {gt_box}: {e}")

        # Save or show the plot
        if filename_to_save:
            try:
                save_path = Path(filename_to_save)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(save_path, bbox_inches="tight")
                logger.info(f"Slice visualization saved to: {save_path}")
            except Exception as e:
                logger.error(
                    f"Error saving slice visualization to '{filename_to_save}': {e}"
                )
            plt.close(fig)  # Close the figure after saving
        else:
            plt.show()


# --- Batch Inference Class ---


class Yolo3dBatchInference(Yolo3dInference):
    """
    Runs YOLO model inference on a batch of 3D NIfTI files.
    Inherits from Yolo3dInference and extends functionality to handle lists of
    NIfTI file paths and compute aggregate metrics.
    """

    def __init__(
        self,
        yolo_model: YOLO,
        nifti_filepaths: List[str],
        gt_filepaths: Optional[List[str]] = None,
        **kwargs,  # Pass other Yolo3dInference args like slice_dims, conf, iou etc.
    ) -> None:
        """
        Initializes the Yolo3dBatchInference class.

        Args:
            yolo_model (YOLO): The pre-loaded Ultralytics YOLO model.
            nifti_filepaths (List[str]): List of paths to input NIfTI files.
            gt_filepaths (Optional[List[str]], optional): List of paths to corresponding ground truth NIfTI masks.
                                                           Must be same length as nifti_filepaths if provided.
                                                           Defaults to None.
            **kwargs: Additional arguments to pass to the Yolo3dInference constructor
                      (e.g., slice_dims, conf, iou, imgsz, etc.). See Yolo3dInference class documentation.
        """
        if gt_filepaths and len(nifti_filepaths) != len(gt_filepaths):
            raise ValueError("Number of NIfTI files and GT files must match.")

        # Initialize the base class with placeholder paths and common args
        super().__init__(
            yolo_model=yolo_model,
            nifti_filepath="",  # Placeholder
            gt_filepath=None,  # Placeholder
            _load_data_on_init=False,  # DO NOT load data on base __init__
            **kwargs,
        )

        self.nifti_filepaths = nifti_filepaths
        self.gt_filepaths = (
            gt_filepaths if gt_filepaths else [None] * len(nifti_filepaths)
        )

        self.batch_results: Dict[str, Dict] = {}  # Store results per file

    def run_batch_inference(self, show_progress: bool = True) -> Dict[str, Dict]:
        """
        Runs inference on all NIfTI files in the batch and computes metrics if GT is available.

        Args:
            show_progress (bool, optional): Whether to display a tqdm progress bar. Defaults to True.

        Returns:
            Dict[str, Dict]: A dictionary where keys are input NIfTI file paths and values are
                             dictionaries containing results for that file:
                             - "predictions": Predicted bounding boxes (np.ndarray).
                             - "ground_truth": Ground truth bounding boxes (np.ndarray, if GT provided).
                             - "map_metrics": mAP metrics from torchmetrics (Dict, if GT provided).
                             - "confusion_matrix_obj": Ultralytics ConfusionMatrix object (if GT provided).
                             - "error": Error message string (if processing failed for this file).
        """
        self.batch_results = {}
        iterator = zip(self.nifti_filepaths, self.gt_filepaths)

        if show_progress:
            iterator = tqdm(
                iterator,
                total=len(self.nifti_filepaths),
                desc="Processing Batch Inference",
            )

        for nifti_path, gt_path in iterator:
            if show_progress:
                iterator.set_postfix_str(f"{os.path.basename(nifti_path)[:-7]}")

            file_result: Dict[
                str, Optional[Union[np.ndarray, Dict, ConfusionMatrix, str]]
            ] = {
                "predictions": None,
                "ground_truth": None,
                "map_metrics": None,
                "confusion_matrix_obj": None,
                "error": None,
            }

            try:
                # --- Process Single File ---
                # Update attributes for the current file (re-loads data)
                self.nifti_filepath = nifti_path
                self.gt_filepath = gt_path
                self._load_data()  # Load image, gt, affine, header for the current file

                if self.img_3d is None:  # Check if loading failed
                    raise ValueError("Failed to load 3D image data.")

                # Run prediction
                predicted_boxes = self.predict_2d_boxes()
                file_result["predictions"] = predicted_boxes

                # Extract GT and compute metrics if GT is available
                if self.gt_3d is not None:
                    gt_boxes = self.extract_gt_boxes_from_mask()
                    file_result["ground_truth"] = gt_boxes

                    # Ensure GT boxes were actually extracted before metric calculation
                    # if gt_boxes.size > 0 or predicted_boxes.size > 0: # Compute even if one is empty
                    map_metrics = self.compute_map_torchmetrics(
                        predicted_boxes, gt_boxes
                    )
                    file_result["map_metrics"] = map_metrics

                    cf_matrix_obj = self.compute_confusion_matrix_ultralytics(
                        predicted_boxes, gt_boxes
                    )
                    file_result["confusion_matrix_obj"] = cf_matrix_obj
                    # else:
                    #     logger.info(f"Skipping metric calculation for {nifti_path} as both GT and predictions are empty.")

            except Exception as e:
                logger.error(
                    f"Error processing file {nifti_path}: {str(e)}", exc_info=True
                )
                file_result["error"] = str(e)

            self.batch_results[nifti_path] = file_result

        return self.batch_results

    def compute_aggregate_metrics(self) -> Dict:
        """
        Computes aggregate metrics across all successfully processed images in the batch.
        Requires `run_batch_inference` to be called first.

        Returns:
            Dict: A dictionary containing aggregate metrics:
                - "processed_files": Count of files successfully processed without errors.
                - "total_gt_boxes": Total number of ground truth boxes across all files.
                - "total_pred_boxes": Total number of predicted boxes across all files.
                - "mean_mAP": Mean average precision (over IoU thresholds) across files from torchmetrics.
                - "mean_mAP_50": Mean mAP at IoU 0.50 from torchmetrics.
                - "mean_mAP_75": Mean mAP at IoU 0.75 from torchmetrics.
                - "aggregate_confusion_matrix": Combined confusion matrix (np.ndarray).
                - "aggregate_precision": Precision calculated from the aggregate confusion matrix.
                - "aggregate_recall": Recall calculated from the aggregate confusion matrix.
                - "aggregate_f1_score": F1 score calculated from the aggregate confusion matrix.
        """
        if not self.batch_results:
            logger.warning("Batch results are empty. Run `run_batch_inference` first.")
            return {}

        valid_map_metrics = []
        all_cf_matrices = []
        total_gt_boxes = 0
        total_pred_boxes = 0
        processed_files = 0

        for file_path, result in self.batch_results.items():
            if result["error"] is None:
                processed_files += 1
                if result["predictions"] is not None:
                    total_pred_boxes += len(result["predictions"])
                if result["ground_truth"] is not None:
                    total_gt_boxes += len(result["ground_truth"])

                # Aggregate torchmetrics mAP results
                if result["map_metrics"] is not None and isinstance(
                    result["map_metrics"], dict
                ):
                    # Check for tensor values before appending
                    if all(
                        isinstance(v, torch.Tensor)
                        for v in result["map_metrics"].values()
                    ):
                        valid_map_metrics.append(
                            {k: v.item() for k, v in result["map_metrics"].items()}
                        )  # Convert tensors to floats
                    else:
                        logger.warning(
                            f"Skipping mAP metrics for {file_path} due to non-tensor values."
                        )

                # Aggregate ultralytics confusion matrices
                if result["confusion_matrix_obj"] is not None and isinstance(
                    result["confusion_matrix_obj"], ConfusionMatrix
                ):
                    all_cf_matrices.append(result["confusion_matrix_obj"].matrix)
            else:
                logger.warning(
                    f"Skipping metrics for file {file_path} due to processing error: {result['error']}"
                )

        if not valid_map_metrics and not all_cf_matrices:
            logger.warning("No valid metrics found across the batch.")
            return {
                "processed_files": processed_files,
                "total_gt_boxes": total_gt_boxes,
                "total_pred_boxes": total_pred_boxes,
            }

        # Calculate mean mAP metrics
        mean_map = (
            np.mean([m.get("map", 0.0) for m in valid_map_metrics])
            if valid_map_metrics
            else 0.0
        )
        mean_map_50 = (
            np.mean([m.get("map_50", 0.0) for m in valid_map_metrics])
            if valid_map_metrics
            else 0.0
        )
        mean_map_75 = (
            np.mean([m.get("map_75", 0.0) for m in valid_map_metrics])
            if valid_map_metrics
            else 0.0
        )

        # Calculate aggregate confusion matrix and derived metrics
        aggregate_cf_matrix = (
            np.zeros_like(all_cf_matrices[0])
            if all_cf_matrices
            else np.zeros((self.num_classes + 1, self.num_classes + 1))
        )
        if all_cf_matrices:
            aggregate_cf_matrix = sum(all_cf_matrices)

        agg_precision, agg_recall, agg_f1 = self.calculate_precision_recall_f1_from_cf(
            aggregate_cf_matrix
        )

        aggregate_metrics = {
            "processed_files": processed_files,
            "total_gt_boxes": total_gt_boxes,
            "total_pred_boxes": total_pred_boxes,
            "mean_mAP": mean_map,
            "mean_mAP_50": mean_map_50,
            "mean_mAP_75": mean_map_75,
            "aggregate_confusion_matrix": aggregate_cf_matrix,
            "aggregate_precision": agg_precision,
            "aggregate_recall": agg_recall,
            "aggregate_f1_score": agg_f1,
        }

        logger.info("Computed aggregate metrics across the batch.")
        return aggregate_metrics
