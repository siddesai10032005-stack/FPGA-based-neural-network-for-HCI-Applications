 FPGA-Based Neural Network for HCI Applications

**Lightweight binocular gaze tracking using the U2Eyes dataset, PyTorch, webcam calibration, INT8 quantization, AMD/Xilinx Vitis AI, and FPGA-oriented XModel deployment.**

---

## Overview

This project develops a lightweight binocular eye-gaze tracking system for Human-Computer Interaction (HCI).

The system uses images of both eyes to estimate gaze direction in a 3×3 directional grid:

- UP-LEFT
- UP-CENTER
- UP-RIGHT
- CENTER-LEFT
- CENTER-CENTER
- CENTER-RIGHT
- DOWN-LEFT
- DOWN-CENTER
- DOWN-RIGHT

The complete development pipeline is:

```text
U2Eyes Dataset
      ↓
XML Metadata Parsing
      ↓
Eye Preprocessing
      ↓
Gaze Label Generation
      ↓
Head-Pose-Aware Train/Validation/Test Split
      ↓
V5 Multi-Task Binocular CNN
      ↓
Held-Out Test Evaluation
      ↓
Real-Time Webcam Inference
      ↓
MediaPipe Eye/Iris Localization
      ↓
9-Point Webcam Calibration
      ↓
Temporal Smoothing
      ↓
INT8 Quantization
      ↓
Vitis AI Deployment Graph
      ↓
XModel Generation
      ↓
Target FPGA/DPU Hardware Validation
```

The current project reaches the **XModel-generation stage**. Actual target FPGA/DPU execution remains the hardware-validation stage.

---

## Main Objectives

The project was developed around the following objectives:

1. Build a lightweight binocular gaze-estimation neural network.
2. Train and evaluate it using the U2Eyes synthetic eye-tracking dataset.
3. Predict continuous gaze angles as well as discrete horizontal, vertical, and 9-class gaze directions.
4. Validate the trained model using a real webcam.
5. Calibrate the webcam-domain predictions using a 9-point calibration process.
6. Reduce visible frame-to-frame jitter using temporal smoothing.
7. Quantize the trained model to INT8 using AMD/Xilinx Vitis AI.
8. Generate an FPGA-oriented XModel.
9. Verify that the deployment-safe graph preserves the original FP32 V5 model behavior before hardware execution.

---

## Dataset

The project uses the **U2Eyes synthetic binocular eye-tracking dataset**.

The current experiment uses:

- Subject: `User_01`
- Head-pose folders: `125`
- Images: `1,875`
- Images per head-pose folder: `15`
- Image resolution: `3840 × 2160`
- Image format: RGB

The dataset contains XML metadata describing head pose, look-at points, and left/right eye point-of-interest information.

### Dataset Structure

```text
User_01/
├── camera4K.xml
├── Grid_15/
│   ├── HP_001/
│   │   ├── 01.png
│   │   ├── 02.png
│   │   ├── ...
│   │   ├── 15.png
│   │   ├── many_headpose.xml
│   │   └── many_poi_data.xml
│   ├── HP_002/
│   ├── ...
│   └── HP_125/
```

The parser converts the XML information into a structured CSV representation.

---

## Dataset Metadata

The processed metadata contains information such as:

- Image path
- Head-pose folder
- Head rotation
- Head position
- Look-at point
- Left-eye iris position
- Left-eye pupil position
- Left-eye cornea position
- Left-eye globe position
- Right-eye iris position
- Right-eye pupil position
- Right-eye cornea position
- Right-eye globe position

The metadata is stored in:

```text
data/processed/u2eyes_metadata.csv
```

---

## Gaze Label Generation

The gaze labels are generated from the dataset geometry.

The world gaze direction is calculated from the look-at point and head position.

The project also generates:

- Continuous yaw
- Continuous pitch
- Horizontal direction
- Vertical direction
- 9-class gaze direction

The final 9-class representation is based on horizontal and vertical gaze regions.

The corresponding processed files are:

```text
data/processed/gaze_labels.csv
data/processed/final_gaze_labels.csv
```

---

## Dataset Split

The dataset was split by **head-pose folder** rather than by individual image.

This prevents images belonging to the same head-pose folder from appearing in both training and testing.

The split uses random seed:

```python
seed = 42
```

### Split

| Split | Head-Pose Folders | Images |
|---|---:|---:|
| Train | 100 | 1,500 |
| Validation | 12 | 180 |
| Test | 13 | 195 |

Processed split files:

```text
data/processed/train.csv
data/processed/val.csv
data/processed/test.csv
```

---

## Eye Preprocessing

The eye preprocessing stage extracts left- and right-eye regions from the original high-resolution images.

The preprocessing pipeline includes:

- Eye-region localization using dataset POI information
- Eye crop generation
- Crop margins
- Resizing
- RGB conversion
- Model-ready formatting

The processed eye dataset metadata is stored in:

```text
data/processed/eye_dataset.csv
```

Generated eye-crop images are intentionally excluded from GitHub because they are large generated artifacts.

---

## V5 Multi-Task Binocular CNN

The final model is **V5**, a lightweight binocular multi-task CNN.

The model contains approximately:

```python
parameters = 94821
```

trainable parameters.

The same eye-CNN feature extractor is shared between the left and right eyes.

### High-Level Architecture

```text
Left Eye
   ↓
Shared Eye CNN
   ↓
64-D Feature
   ┐
   │
   ├── Concatenate → 128-D binocular representation
   │
Right Eye
   ↓
Shared Eye CNN
   ↓
64-D Feature
```

The binocular representation is passed through a shared feature layer and then into multiple task-specific heads.

### Model Heads

The V5 model contains:

```text
Regression Head
    ↓
Yaw + Pitch

Horizontal Head
    ↓
Left / Center / Right

Vertical Head
    ↓
Down / Center / Up

Geometry Head
    ↓
4 geometry values

Direction Fusion Head
    ↓
9-class gaze direction
```

### Eye CNN

The shared eye CNN is:

```text
Conv2D
BatchNorm
ReLU
MaxPool

Conv2D
BatchNorm
ReLU
MaxPool

Conv2D
BatchNorm
ReLU

Conv2D
BatchNorm
ReLU

Adaptive Average Pooling
```

The final eye representation is 64-dimensional per eye.

The two eye representations are concatenated into a 128-dimensional binocular representation.

---

## V5 Training

The final training implementation is:

```text
src/train_multitask_v5.py
```

The model was trained using PyTorch with:

- AdamW optimizer
- Learning rate: `5e-4`
- Weight decay: `1e-4`
- Batch size: `32`
- Maximum epochs: `60`
- Gradient clipping
- ReduceLROnPlateau learning-rate scheduling
- Fixed random seed: `42`

The main loss combines the multi-task objectives:

```text
Regression Loss
+ Horizontal Classification Loss
+ Vertical Classification Loss
+ 9-Class Direction Loss
+ Geometry Loss
```

with the V5 loss weighting used during the final experiment.

---

## Final V5 Checkpoint

The final trained checkpoint is:

```text
models/best_multitask_v5.pth
```

This is the primary FP32 model used for evaluation and deployment preparation.

---

## V5 Held-Out Test Results

The final V5 model was evaluated on the untouched held-out test set containing:

```python
test_samples = 195
```

### Continuous Gaze Regression

| Metric | Result |
|---|---:|
| Yaw MAE | 1.747° |
| Pitch MAE | 1.396° |
| Overall MAE | 1.571° |
| Mean Angular Error | 2.407° |
| Predictions within 5° | 92.31% |
| Predictions within 10° | 100% |

### 9-Class Gaze Direction

| Metric | Result |
|---|---:|
| Accuracy | 97.95% |
| Macro Precision | 98.38% |
| Macro Recall | 97.44% |
| Macro F1 | 97.84% |

The V5 model produced four incorrect 9-class predictions on the 195-sample held-out test set.

---

## Real-Time Webcam Pipeline

The trained V5 model was integrated with a real webcam.

The physical camera was connected to the Windows host while model inference and development were performed inside an Ubuntu VMware virtual machine.

The working architecture was:

```text
Windows Physical Webcam
        ↓
OpenCV Camera Server
        ↓
MJPEG HTTP Stream
        ↓
Ubuntu VMware VM
        ↓
MediaPipe Face/Iris Localization
        ↓
V5 Eye Crops
        ↓
V5 Gaze Prediction
        ↓
Webcam Calibration
        ↓
Temporal Smoothing
        ↓
Real-Time Gaze Direction
```

The Windows-to-Ubuntu network stream was used because direct VMware camera passthrough produced corrupted camera frames during testing.

---

## MediaPipe Eye and Iris Localization

MediaPipe is used during live webcam inference for localization of the eye regions and iris landmarks.

The webcam image is processed to obtain:

- Face location
- Left-eye region
- Right-eye region
- Iris positions

The resulting eye crops are passed to the trained V5 network.

Relevant implementation:

```text
src/live_v5_real.py
```

---

## Webcam Calibration

A 9-point webcam calibration process was added to reduce the domain difference between the synthetic U2Eyes data and the real webcam environment.

The calibration grid is:

```text
UP-LEFT        UP-CENTER        UP-RIGHT

CENTER-LEFT    CENTER-CENTER    CENTER-RIGHT

DOWN-LEFT      DOWN-CENTER      DOWN-RIGHT
```

The calibration uses the V5 raw outputs together with webcam-domain iris features to learn a mapping from the model's raw prediction domain to the real camera domain.

The calibration artifact is:

```text
models/v5_webcam_calibration.npz
```

The live pipeline reports that the calibrated webcam predictions became much more consistent across the nine calibration regions.

---

## Temporal Smoothing

Temporal smoothing was added after calibration to reduce frame-to-frame prediction jitter.

The implementation uses an exponential moving average:

```python
smooth_t = alpha * current_t + (1 - alpha) * smooth_previous
```

with:

```python
alpha = 0.30
```

A lower alpha gives stronger smoothing while increasing the amount of temporal lag.

The smoothing state is reset when:

- No face is detected
- A new calibration is started
- Calibration data is deleted
- The tracking state is reinitialized

---

## Quantization

The V5 model was quantized using:

```text
AMD/Xilinx Vitis AI 3.5
PyTorch 1.13.1
INT8 quantization
```

The host development environment and Vitis AI environment were kept separate because the Vitis AI toolchain requires a compatible PyTorch/NNDCT environment.

The calibration process used representative training data to collect activation statistics for INT8 quantization.

The quantization workflow was:

```text
FP32 V5 Model
      ↓
Representative Calibration Images
      ↓
Vitis AI NNDCT Quantization
      ↓
INT8 Quantized Representation
      ↓
Deployment Graph
      ↓
XModel
```

---

## Deployment-Safe Graph

During XModel generation, the original V5 graph encountered deployment issues associated with graph transformations around:

- Adaptive average pooling
- Flatten
- Reshape-like operations
- Linear layers

Instead of changing the learned computation, a deployment-safe representation was constructed.

### Graph Changes

The deployment-safe model uses:

```text
Adaptive Average Pooling
        ↓
Fixed Average Pooling
```

and:

```text
Linear Layer
        ↓
Mathematically Equivalent 1×1 Convolution
```

This keeps the main computation in a 4-D convolution-friendly representation that is more suitable for the Vitis AI deployment flow.

---

## FP32 Deployment Equivalence Verification

Before accepting the deployment-safe graph, the original V5 model and the deployment-safe FP32 representation were evaluated on all 195 test samples.

The maximum numerical differences were:

| Output | Maximum Difference |
|---|---:|
| Regression | 0.0000043 |
| Horizontal logits | 0.0000038 |
| Vertical logits | 0.0000038 |
| Geometry | 0.0000001 |
| Direction logits | 0.0000076 |

Discrete prediction agreement was:

| Prediction | Agreement |
|---|---:|
| Horizontal | 100% |
| Vertical | 100% |
| 9-Class Direction | 100% |

This verification demonstrates that the deployment-safe FP32 graph preserves the original V5 model behavior across the complete held-out test set.

It does **not** by itself prove hardware accuracy or hardware latency.

The verification implementation is:

```text
src/check_fp32_deploy_equivalence.py
```

---

## Final XModel

The final XModel was successfully generated using Vitis AI.

Primary artifact:

```text
models/v5_xmodel_final/DeployV5_int.xmodel
```

Approximate size:

```python
xmodel_size_bytes = 740915
```

The deployment directory contains:

```text
models/v5_xmodel_final/
├── DeployV5.py
├── DeployV5_int.xmodel
├── bias_corr.pth
└── quant_info.json
```

The XModel generation implementation is:

```text
src/v5_xmodel_final.py
```

---

## Current Hardware Status

The project currently reaches:

```text
Python/PyTorch Training
        ↓
FP32 Evaluation
        ↓
Real-Time Webcam
        ↓
Calibration
        ↓
Temporal Smoothing
        ↓
INT8 Quantization
        ↓
Vitis AI XModel Generation
        ↓
[CURRENT POINT]
        ↓
Target FPGA/DPU Hardware Validation
```

The generated XModel has **not yet been treated as a completed hardware benchmark**.

The next hardware phase is to execute the XModel on the intended FPGA/DPU platform and measure:

- Inference latency
- Throughput
- Resource utilization
- Power
- End-to-end real-time performance

---

## Repository Structure

```text
FPGA-based-neural-network-for-HCI-Applications/
│
├── .gitignore
│
├── data/
│   └── processed/
│       ├── eye_dataset.csv
│       ├── final_gaze_labels.csv
│       ├── gaze_labels.csv
│       ├── train.csv
│       ├── val.csv
│       ├── test.csv
│       └── u2eyes_metadata.csv
│
├── models/
│   ├── best_multitask_v5.pth
│   ├── v5_webcam_calibration.npz
│   └── v5_xmodel_final/
│       ├── DeployV5.py
│       ├── DeployV5_int.xmodel
│       ├── bias_corr.pth
│       └── quant_info.json
│
├── results/
│   └── HP_110_contact_sheet.png
│
└── src/
    ├── check_fp32_deploy_equivalence.py
    ├── create_dataset_split.py
    ├── create_gaze_labels.py
    ├── create_relative_gaze_labels.py
    ├── evaluate_multitask_v5.py
    ├── live_eye_detection.py
    ├── live_iris_test.py
    ├── live_v5_real.py
    ├── parse_u2eyes.py
    ├── preprocess_u2eyes.py
    ├── quantize_v5.py
    ├── train_multitask_v5.py
    ├── v5_int8_export.py
    ├── v5_int8_final.py
    ├── v5_int8_final_export.py
    ├── v5_xmodel_final.py
    └── webcam_test.py
```

---

## Python Environment

A standard Python environment can be created programmatically:

```python
import subprocess
import sys

packages = [
    "torch",
    "torchvision",
    "pandas",
    "numpy",
    "scikit-learn",
    "matplotlib",
    "opencv-python",
    "mediapipe",
    "joblib",
]

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", *packages]
)
```

For Vitis AI quantization and XModel generation, use the compatible Vitis AI environment rather than the normal project virtual environment.

---

## Running the Main Pipeline

### Parse the Dataset

The parser is:

```python
# src/parse_u2eyes.py
```

Run the script using Python:

```python
import subprocess
import sys

subprocess.run(
    [sys.executable, "src/parse_u2eyes.py"],
    check=True,
)
```

---

### Preprocess the Eye Images

```python
import subprocess
import sys

subprocess.run(
    [sys.executable, "src/preprocess_u2eyes.py"],
    check=True,
)
```

---

### Create Gaze Labels

```python
import subprocess
import sys

subprocess.run(
    [sys.executable, "src/create_gaze_labels.py"],
    check=True,
)
```

---

### Create the Dataset Split

```python
import subprocess
import sys

subprocess.run(
    [sys.executable, "src/create_dataset_split.py"],
    check=True,
)
```

---

### Train V5

```python
import subprocess
import sys

subprocess.run(
    [sys.executable, "src/train_multitask_v5.py"],
    check=True,
)
```

The output checkpoint is:

```text
models/best_multitask_v5.pth
```

---

### Evaluate V5

```python
import subprocess
import sys

subprocess.run(
    [sys.executable, "src/evaluate_multitask_v5.py"],
    check=True,
)
```

---

### Run Webcam Testing

The primary live implementation is:

```text
src/live_v5_real.py
```

It combines:

```text
MediaPipe
+
V5
+
Webcam Calibration
+
Temporal Smoothing
```

It can be launched programmatically with:

```python
import subprocess
import sys

subprocess.run(
    [sys.executable, "src/live_v5_real.py"],
    check=True,
)
```

---

### Generate the XModel

The final deployment graph is implemented in:

```text
src/v5_xmodel_final.py
```

It can be launched with:

```python
import subprocess
import sys

subprocess.run(
    [sys.executable, "src/v5_xmodel_final.py"],
    check=True,
)
```

The final XModel is:

```text
models/v5_xmodel_final/DeployV5_int.xmodel
```

---

## Controls for the Webcam Application

The real-time webcam application uses these keyboard controls:

```text
C → Start 9-point calibration
R → Delete/reset calibration
Q → Quit
```

---

## Reproducibility

The project uses deterministic dataset splitting with:

```python
seed = 42
```

The held-out evaluation set contains:

```python
test_samples = 195
```

The final V5 checkpoint is included in the repository so that the reported FP32 results can be reproduced without retraining the model from scratch.

The raw U2Eyes dataset itself is not included because of its size and dataset distribution constraints.

---

## Files Excluded from GitHub

The repository intentionally excludes large or experimental local files, including:

```text
Raw U2Eyes ZIP
Generated eye-crop images
Experimental model checkpoints
Experimental V5/V6 variants
Intermediate INT8 test outputs
Obsolete evaluation scripts
Backup scripts
```

The final repository therefore focuses on the reproducible V5 pipeline and deployment artifacts.

---

## Technologies Used

```text
Python
PyTorch
TorchVision
NumPy
Pandas
scikit-learn
OpenCV
MediaPipe
U2Eyes Dataset
AMD/Xilinx Vitis AI
NNDCT
INT8 Quantization
XModel
FPGA / DPU
VMware
Ubuntu
```

---

## Project Status

### Completed

- [x] U2Eyes dataset parsing
- [x] XML metadata extraction
- [x] Eye preprocessing
- [x] Gaze label generation
- [x] Head-pose-aware dataset split
- [x] Lightweight V5 binocular CNN
- [x] Multi-task gaze regression/classification
- [x] Held-out test evaluation
- [x] 9-class gaze direction
- [x] Real-time webcam inference
- [x] MediaPipe eye/iris localization
- [x] 9-point webcam calibration
- [x] Temporal smoothing
- [x] INT8 quantization workflow
- [x] Deployment-safe V5 graph
- [x] FP32 deployment-equivalence verification
- [x] Vitis AI XModel generation

### Remaining

- [ ] Target FPGA/DPU hardware execution
- [ ] Hardware latency benchmark
- [ ] Throughput benchmark
- [ ] Resource-utilization analysis
- [ ] Power measurement
- [ ] End-to-end hardware HCI demonstration

---

## Limitations

The current reported results should be interpreted within the scope of the experiment.

### Dataset Scope

The current experiment uses only:

```text
User_01
```

from U2Eyes.

Therefore, the current results should not be interpreted as subject-independent generalization across multiple people.

### Synthetic-to-Real Domain Gap

U2Eyes is a synthetic dataset, while the webcam pipeline operates on real camera images.

A calibration layer was therefore introduced to compensate for the webcam-domain difference.

### Hardware Validation

The XModel has been generated successfully, but final FPGA/DPU execution and hardware benchmarking are still pending.

---

## Future Work

The next phase is hardware deployment and benchmarking.

```text
Final XModel
      ↓
FPGA / DPU
      ↓
Hardware Inference
      ↓
Measure:
      ├── Latency
      ├── Throughput
      ├── Resource Utilization
      ├── Power
      └── End-to-End Accuracy
      ↓
Real-Time HCI Demonstration
```

Potential future extensions include:

- Hardware-aware optimization
- Additional subjects
- Larger real-world webcam datasets
- Improved domain adaptation
- Hardware latency optimization
- Quantization-aware training
- FPGA resource optimization
- End-to-end gaze-controlled HCI applications

---

## Conclusion

This project demonstrates an end-to-end FPGA-oriented binocular gaze-tracking pipeline starting from the U2Eyes synthetic dataset and progressing through model training, real-time webcam calibration, temporal smoothing, INT8 quantization, and Vitis AI XModel generation.

The final V5 model is lightweight, with approximately 94.8K trainable parameters, while achieving strong held-out gaze estimation and 9-class directional classification performance on the current U2Eyes experiment.

The project currently reaches the XModel-generation stage, with target FPGA/DPU execution remaining as the final hardware-validation step.

---

## Author

**Siddharth Desai**

FPGA / AI / Embedded Systems / Human-Computer Interaction

GitHub:

```text
https://github.com/siddesai10032005-stack
```

Project:

```text
https://github.com/siddesai10032005-stack/FPGA-based-neural-network-for-HCI-Applications
```
"""

output_path = Path(__file__).resolve().parent / "README.md"
output_path.write_text(README, encoding="utf-8")

print(f"Created: {output_path}")
print(f"README size: {output_path.stat().st_size:,} bytes")
