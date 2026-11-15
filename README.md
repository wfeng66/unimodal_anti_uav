# MMAUD RGB-to-3D Prism Localization Baselines

A small, modular PyTorch baseline for predicting the 3D Leica-tracked prism position
from an MMAUD camera image. It supports `--model yolo`, `--model detr`, and
`--model dinov2`. It is intentionally a first-stage single-frame RGB regression
benchmark; it does not implement tracking, temporal fusion, or multimodal fusion.

## 1. Important MMAUD data facts

The public MMAUD release separates V1 sensor recordings, V1 ground-truth bags,
and extracted V1 filesystem data. V1 includes five recordings (Avata, M300,
Mavic2, Mavic3, and Phantom4/Pham4 in different release listings). The dataset
also supplies two camera streams, LiDAR, radar, audio, and Leica tracking.
The V1 source data is distributed as ROS1 bags and as extracted PNG/NumPy
files. Refer to the official project and Hugging Face dataset card for the
current file inventory and download instructions:

- https://github.com/ntu-aris/MMAUD
- https://huggingface.co/datasets/Yuan-Shenghai/MMAUD

The current Hugging Face release describes the V1 extracted filesystem as
33,200 PNG images and 43,586 NumPy arrays. It also provides calibration
resources, including camera parameter YAML files, reports, CAD/STL files,
and a calibration archive. File naming and timestamp conventions should be
taken from the exact V1 sequence export being used; do not infer timestamp
semantics from filenames without checking the extraction metadata.

**Important:** The official calibration overview states that camera intrinsics
are supplied, while V1 still requires trajectory-based spatial alignment to
Leica. The calibration materials are initial information; the inter-sensor
extrinsics should be estimated/refined using paired calibration/trajectory
data. See the calibration documentation before declaring the target frame.

The original MMAUD sensor page includes Leica pose ground truth and explains
that the prism is offset from the platform body. Its evaluation tutorial
explicitly compensates body-to-prism displacement and synchronizes trajectories
before comparing positions. Therefore, the label must be the **prism position**,
not a body-center position, and must be paired to the camera frame timestamp.

## 2. Dataset loading and timestamp association

Training reads the extracted V1 filesystem directly; no CSV manifest is
required. The default data root is `I:\Data\Anti-UAV\MMAUD\v1` on Windows and
`/mnt/i/Data/Anti-UAV/MMAUD/v1` on WSL/Linux. Windows drive paths passed to
`--data-root` are translated to the conventional `/mnt/<drive>/...` WSL path.
You can also pass the actual mount path explicitly. The root can point either
to the directory containing `v1` or directly to `v1`. For each recording, the
script reads PNGs from
`<v1>/<recording>/usb_cam_image_raw/decoded/` and finds the matching
relative-Leica timestamp index at
`<v1>/<date>_<recording>/leica_point_relative/timestamps.csv`.

Image and Leica timestamps are nanoseconds encoded in the filename and
`timestamp_ns` CSV column. The nearest annotation is used only when it is
within 100 ms by default (`--max-sync-delta-ms`). Its JSON
`annotation.point` supplies the prism's relative Leica-frame XYZ coordinates.
Images without a nearby annotation are skipped; a split with no synchronized
images is an error. The script does not read ROS bags, use absolute Leica
coordinates, undistort images, or estimate camera extrinsics.

The `--val` and `--test` options take recording names (comma-separated if
needed); every other discovered recording is used for training. Splitting is
by whole recording to avoid leakage between adjacent frames. For example,
`--val avata --test m300` holds those recordings out and trains on the rest.

## 3. Coordinate frame and calibration

This benchmark uses relative Leica coordinates in meters. The default metadata
label is `leica_relative`; change `--coordinate-frame` only when the labels have
actually been transformed into another frame.

### Do intrinsics/extrinsics improve RGB-only regression?

They are useful, but they do not automatically improve a CNN/ViT merely by
being present on disk.

- **Intrinsics** describe the camera projection and distortion. MMAUD's camera
  calibration documentation describes an omni + radtan camera model at
  1280 x 960. A standard pinhole undistortion routine must not be assumed to
  implement this model correctly. For this initial baseline, images are resized
  directly; camera undistortion is not silently applied.
- **Extrinsics** map points between sensor coordinate frames. For V1, the
  Leica/world-to-camera or camera-to-Leica relationship needs to be estimated
  from temporally paired trajectories/calibration data. The transform direction,
  axis convention, and units must be recorded explicitly.
- For an RGB-only network trained to predict Leica-frame coordinates, calibrated
  extrinsics are primarily needed to create correctly aligned labels and to
  validate the coordinate frame. They are not an additional input feature.
- A later, controlled variant could predict camera-frame coordinates and
  transform predictions into Leica/world coordinates, or use calibrated
  projection/geometric features. Keep that out of the first baseline so that
  the initial model comparison remains a plain image-to-coordinate regression.

The prism/body offset matters. Use prism-centered Leica ground truth, or apply
the documented body-to-prism rigid transform with the correct pose/orientation.
Do not compare a body-center prediction with prism-center labels.

## 4. Models

| CLI name | Implementation | Default checkpoint |
|---|---|---|
| `yolo` | Ultralytics YOLO26 detection backbone + feature pyramid with a new 3-value regression head | `yolo26n.pt` |
| `detr` | Hugging Face DETR encoder-decoder; first learned decoder query feeds a 3D regression MLP | `facebook/detr-resnet-50` |
| `dinov2` | Hugging Face DINOv2 ViT; CLS token feeds a 3D regression MLP | `facebook/dinov2-base` |

The YOLO adapter loads the pretrained detection model, retains its pretrained
backbone and feature-pyramid layers, removes the detection head, and attaches a
new randomly initialized 3D regression head. The pretrained layers are frozen
for the first `--freeze-pretrained-epochs` epochs (default 5), then all layers
are unfrozen for the remaining epochs. Set this option to `0` to fine-tune the
entire model from the first epoch. The same freeze/unfreeze schedule applies to
the pretrained DETR and DINOv2 encoders. The task is one 3D prism coordinate
per image, not 2D boxes.

All models output three normalized continuous values. The target mean and
standard deviation are calculated from the training split only. Predictions
are converted back to the declared coordinate units before evaluation.

## 5. Split integrity and held-out test protocol

The recordings specified with `--val` and `--test` are held out; all remaining
recordings are used for training.

- Recording sequence IDs must be disjoint across train, validation, and test.
- The validation split selects the best checkpoint.
- The test dataset is instantiated and evaluated only after training ends and
  the best validation checkpoint has been restored.
- Test metrics are never used for gradient updates, early stopping, or model
  selection.

Do not create random frame-level splits: adjacent frames from the same flight
are temporally correlated and can cause serious leakage.

## 6. Install

Use a clean Python environment with a PyTorch build matching your CUDA runtime.
Then:

```bash
python -m pip install -r requirements.txt
```

Install the requirements after installing a PyTorch build appropriate for your
platform. Model checkpoints are downloaded by their respective libraries on
first use.

## 7. Train

```bash
python train.py \
  --model yolo \
  --val avata \
  --test m300 \
  --epochs 50 \
  --freeze-pretrained-epochs 5 \
  --batch-size 16 \
  --learning-rate 0.0001 \
  --device auto
```

The default results file is named
`results/<method>_<test-split>_<month>_<year>.csv`, for example
`results/yolo_m300_11_2026.csv`. Pass `--results-csv` to override that path.

Run the four validation folds with Mavic3 held out for testing:

```bash
bash run_mavic3_cross_validation.sh
```

By default this uses YOLO. Set `MODEL=detr` or `MODEL=dinov2` to run a different
method; extra training options can be passed after the script name. Avata,
M300, Mavic2, and Pham4 rotate as validation sequences, with all other
non-test recordings used for training. The four runs append to the same
`<method>_mavic3_<month>_<year>.csv` file.

`--data_path` can point to another extracted dataset root. `--data-root` is
also accepted as an equivalent spelling.

Change the model:

```bash
python train.py --model detr --data-root /path/to/MMAUD --val avata --test m300
python train.py --model dinov2 --data-root /path/to/MMAUD --val avata --test m300
```

Useful options:

```text
--checkpoint          Override the model's default pretrained checkpoint
--data_path           Dataset root (default depends on Windows vs. WSL/Linux)
--data-root           Alias for --data_path
--val                 Comma-separated validation recording names
--test                Comma-separated test recording names
--max-sync-delta-ms   Maximum image/Leica timestamp difference (default: 100)
--freeze-pretrained-epochs
                      Freeze pretrained layers initially (default: 5; 0 disables)
--train-split         Internal training label (default: train)
--val-split           Internal validation label (default: val)
--test-split          Internal test label (default: test)
--coordinate-frame    Metadata describing x/y/z frame (default: leica_relative)
--coordinate-unit     Coordinate unit (default: m)
--image-size          Square input resolution (default: 224)
--output-dir          Checkpoint/config output directory
--results-csv         Append-only experiment CSV (default is named by method,
                      test split, month, and year)
--device              auto, cpu, or cuda
```

## 8. Outputs and CSV

Each run writes:

- Best validation checkpoint (`*_best.pt`)
- Full run configuration JSON (`*_config.json`)
- One appended row in the method/test/month/year results CSV (default)

The CSV includes model/checkpoint, machine/GPU/CPU/RAM, training hyperparameters,
dataset version and split names, coordinate frame/unit, total/trainable parameter
counts, training time, batch-1 inference latency/FPS, peak CUDA allocated/reserved
memory, sample counts, and final held-out test metrics.

Metrics (meters):

- Mean Euclidean distance error
- Median Euclidean error
- 95th-percentile Euclidean error
- Coordinate-wise MAE for x, y, z
- 3D RMSE and maximum Euclidean error

Inference latency is measured for model forward only, batch size 1, after warm-up;
it excludes image decoding and resizing. GPU memory fields are blank on CPU runs.
System RAM is recorded separately when `psutil` is available.

## 9. Tests

Run:

```bash
pytest -q
```

- Tests cover coordinate metrics/statistics, relative-Leica timestamp matching,
  and pretrained-layer freeze/unfreeze behavior. They do not download model
  checkpoints; run a real-model smoke test after installing dependencies.

## 10. Known limitations of this first version

1. Raw ROS bag extraction is not implemented; training expects the extracted
   decoded image and relative-Leica timestamp/annotation files.
2. Camera-specific omni + radtan undistortion is not implemented.
3. V1 Leica/camera extrinsic estimation is not implemented.
4. This is single-frame RGB regression, not trajectory estimation.
5. Only one prism position is predicted per frame.
6. The CSV records the selected checkpoint identifier/path, but the exact
   downloaded model-file hash is not yet captured.

## 11. Dataset citation and license

Please cite the MMAUD ICRA 2024 paper when using the dataset:

Yuan et al., “MMAUD: A Comprehensive Multi-Modal Anti-UAV Dataset for Modern
Miniature Drone Threats,” IEEE International Conference on Robotics and
Automation (ICRA), 2024, pp. 2745–2751.
DOI: https://doi.org/10.1109/ICRA57147.2024.10610957

The dataset is released under CC BY-NC-SA 4.0 according to the current dataset
card. Check the dataset terms before redistribution or commercial use.
