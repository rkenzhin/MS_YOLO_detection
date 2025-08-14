# YOLO MS Lesion Detection on MRI

This project utilizes the Ultralytics YOLO framework to train and evaluate object detection models for identifying Multiple Sclerosis (MS) lesions on brain MRI scans. The project provides tools for converting standard medical imaging formats (NIfTI) into a YOLO-compatible 2D slice format and performing subsequent training and evaluation. It includes scripts for dataset preparation, training, inference on 3D NIfTI volumes, and analysis of results.

## Table of Contents

- [Features](#features)
- [File Structure](#file-structure)
- [Setup and Installation](#setup)
- [Usage](#usage)
  - [Dataset Creation](#dataset-creation)
  - [Training](#training)
  - [Inference](#inference)
  - [Results Analysis](#results-analysis)
  - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Model Naming Convention](#model-naming-convention)
- [License](#license)
- [Contact](#contact)

## Features

-   **Dataset Conversion:** Scripts to convert 3D NIfTI datasets (MS_Shift, MSLesSeg) into 2D slices with corresponding YOLO label files (`.txt`).
    -   Supports different slicing dimensions (axial, sagittal, coronal).
    -   Handles multiple MRI modalities.
    -   Generates bounding boxes from segmentation masks.
    -   Optionally limits the number of generated samples per split.
    -   Optionally resizes output images.
-   **YOLOv11 Training:** Script to train YOLOv11 models using a standard YAML configuration file.
-   **3D Inference:** Classes (`Yolo3dInference`, `Yolo3dBatchInference`) to perform inference on entire 3D NIfTI volumes by processing slices and aggregating results.
-   **Metric Calculation:** Calculates standard object detection metrics (mAP via `torchmetrics`, Precision/Recall/F1 via `ultralytics.utils.metrics.ConfusionMatrix`) for 3D inference results against ground truth masks.
-   **Visualization:** Utilities (`viz_utils.py`) to visualize image slices with ground truth and predicted bounding boxes.
-   **Results Analysis:** Notebook (`Results.ipynb`) to analyze training progress and validation metrics.
-   **EDA:** Notebook (`EDA.ipynb`) for exploring dataset structure, dimensions, intensity distributions, and mask statistics.

## File Structure

<pre>
project-root/
├── create_dataset.py # Script to run dataset creators
├── train.py # Training script
├── inference.py # Classes/functions for 3D inference and metrics
├── viz.py # Visualization utilities
├── utils.py # Utilities for Exploratory Data Analysis and metrics
├── results.ipynb # Notebook for analyzing results
├── requirements.txt # Project dependencies
├── *.yaml # Dataset configuration files (e.g., MS_Shift.yaml)
├── EDA.ipynb # EDA notebook
├── Results.ipynb  Notebook for validation/visualization
├── runs/ # Directory for saving training runs (created by Ultralytics)
│ └── detect/
│ └── [ExperimentName]/
│ ├── weights/
│ │ ├── best.pt
│ │ └── last.pt
│ ├── results.csv
│ └── ... (other training files)
├── README.md # This file
└── ... (other project files)
</pre>

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <https://github.com/rkenzhin/MS_YOLO_detection.git>
    cd <MS_object_detection>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # .\venv\Scripts\activate  # Windows
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *   **For GPU:** Ensure you have compatible NVIDIA drivers and CUDA Toolkit installed. You might need to install a specific PyTorch version for your CUDA version (see [PyTorch official website](https://pytorch.org/)).

4.  **Data:**

    *   **Download Datasets:** Obtain the MS_Shift and/or MSLesSeg datasets and place them in a known location (e.g., `/path/to/data/`).
    *   **Create YOLO Datasets:** Run the dataset creation script (see [Usage](#dataset-creation)).

## Usage

### Dataset Creation

Use the provided scripts to convert NIfTI datasets into YOLO format.

1.  **Configure:** Edit `create_datasets.py` to set the correct input `data_dir`, desired `output_dir`, `modalities`, `slice_dims`, etc. for each dataset you want to process.
2.  **Run:**
    ```bash
    python main_create_datasets.py
    ```
    This will generate the `images` and `labels` directories with `train`/`val`/`test` subfolders inside your specified `output_dir`.

### Training

1.  **Prepare Data YAML:** Create a YAML file (e.g., `MS_Shift.yaml`, `MSShift_MSLesSeg.yaml`) describing the paths to your generated YOLO dataset, number of classes, and class names. Example:
    ```yaml
    path: ../MS_shift_YOLO_axial_dataset # Path relative to project root or absolute
    train: images/train
    val: images/val
    test: images/test # Optional

    nc: 1  # Number of classes
    names: ['lesion'] # Class names
    ```
2.  **Run Training:** Use the `train.py` script.
    ```bash
    python train.py
    ```
    Edit `train.py` to set parameters as needed.

### Inference

Use the `inference.py` classes, typically within a Jupyter notebook `Results.ipynb`, for detailed analysis.

1.  **Load Model:** `model = YOLO("path/to/your/best.pt")`
2.  **Instantiate Inference Class:**
    ```python
    from inference_utils import Yolo3dInference, Yolo3dBatchInference

    # Single file inference
    inference_single = Yolo3dInference(
        yolo_model=model,
        nifti_filepath="/path/to/image.nii.gz",
        gt_filepath="/path/to/mask.nii.gz", # Optional
        slice_dims=[0, 1, 2],
        conf=0.1, # Adjust confidence
        iou=0.6, # Adjust iou
        imgsz=640
    )
    pred_boxes = inference_single.predict_2d_boxes()
    gt_boxes = inference_single.extract_gt_boxes_from_mask()
    metrics = inference_single.compute_map_torchmetrics(pred_boxes, gt_boxes)
    cf_matrix = inference_single.compute_confusion_matrix_ultralytics(pred_boxes, gt_boxes)
    pred_mask = inference_single.create_predicted_3d_box_mask(pred_boxes, save_to_nifti=True, output_filename="output/pred_mask.nii.gz")

    # Batch inference
    imgs_list = [...] # List of image paths
    gts_list = [...] # List of corresponding mask paths or None
    batch_inference = Yolo3dBatchInference(
        yolo_model=model,
        nifti_filepaths=imgs_list,
        gt_filepaths=gts_list,
        slice_dims=[0, 1, 2],
        conf=0.1, # Adjust confidence
        iou=0.6, # Adjust iou
        imgsz=640
    )
    batch_results = batch_inference.run_batch_inference()
    aggregate_metrics = batch_inference.compute_aggregate_metrics()
    print(aggregate_metrics)
    ```

### Results Analysis

Use the `Results.ipynb` notebook to visualize predictions, analyze and compute metrics.

1.  **Configure Paths:** Update datasets paths in the notebook with paths to your training runs.
2.  **Run Cells:** Execute the cells to load data, calculate fitness, find best epochs/metrics, compare models, and plot learning curves.

### Exploratory Data Analysis (EDA)

Use the `EDA.ipynb` notebook and `utils.py` module to explore the original NIfTI datasets.

1.  **Configure Paths:** Update datasets paths in the notebook.
2.  **Run Cells:** Execute cells to check file structures, analyze image dimensions, view multi-modal slices, analyze intensity distributions, and visualize masks/bounding boxes.

## Model Naming Convention

To keep track of experiments, model weights and run directories are named using the following convention:

`{DATASET_CODES}_{MODALITIES}_{SLICE_DIMS}_{EPOCHS}ep_{CLASSES}cls_{IMGSZ}imgsz_{AUGMENT}_{MODEL_SIZE}_{YOLO_VERSION}`

-   **DATASET_CODES:** `mss` (MS Shift), `msl` (MSLesSeg), `msl_mss` (Both)
-   **MODALITIES:** (Optional) e.g., `FLAIR` (if only one used, else omit for all)
-   **SLICE_DIMS:** (Optional) e.g., `ax` (if only axial used, else omit for all)
-   **EPOCHS:** e.g., `100ep`
-   **CLASSES:** e.g., `1cls`
-   **IMGSZ:** e.g., `640imgsz`
-   **AUGMENT:** (Optional) `aug` (if augmentation enabled)
-   **MODEL_SIZE:** `n`, `s`, `m`, `l`, `x`
-   **YOLO_VERSION:** (Optional) e.g., `v11` (if not v8)

## License

[MIT License]

## Contact

[r.kenzhin@nsu.ru](mailto:r.kenzhin@nsu.ru)
