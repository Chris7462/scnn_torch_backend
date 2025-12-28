# SCNN Lane Detection Torch Backend

C++ backend for SCNN lane detection using LibTorch. This backend is based on the SCNN implementation in [scnn_torch](https://github.com/Chris7462/scnn_torch).

## Preparation

### Clone PyTorch and TorchVision from GitHub
```bash
cd ~/thirdparty/
git clone git@github.com:pytorch/pytorch.git --recurse-submodules
git clone git@github.com:pytorch/vision.git --recurse-submodules
```

### Build LibTorch
```bash
cd ~/thirdparty/pytorch
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DBUILD_SHARED_LIBS=ON \
         -DUSE_CUDA=ON \
         -DUSE_CUDNN=ON \
         -DUSE_CUDSS=ON \
         -DUSE_CUFILE=ON \
         -DUSE_CUSPARSELT=ON \
         -DCMAKE_INSTALL_PREFIX=$HOME/thirdparty/libtorch
cmake --build . -j8
cmake --install .
```

If you encountered issues, try adding the following to your `~/.bashrc` before building libtorch:
```bash
export PATH=/usr/local/cuda/bin:/usr/src/tensorrt/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CPATH=/usr/local/cuda/include:$CPATH
export C_INCLUDE_PATH=/usr/local/cuda/include:$C_INCLUDE_PATH
export LIBRARY_PATH=/usr/local/cuda/lib64:$LIBRARY_PATH
```

### Point to your custom libtorch installation
Add the following to your `~/.bashrc` file:
```bash
export Torch_DIR="$HOME/thirdparty/libtorch/share/cmake/Torch"
export LD_LIBRARY_PATH="$HOME/thirdparty/libtorch/lib:$LD_LIBRARY_PATH"
```

### Build LibTorchVision
```bash
cd ~/thirdparty/vision
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DBUILD_SHARED_LIBS=ON \
         -DWITH_CUDA=ON \
         -DCMAKE_PREFIX_PATH=$HOME/thirdparty/libtorch \
         -DCMAKE_INSTALL_PREFIX=$HOME/thirdparty/libtorchvision
cmake --build . -j8
cmake --install .
```

### Point to your custom libtorchvision installation
Add the following to your `~/.bashrc` file:
```bash
export TorchVision_DIR="$HOME/thirdparty/libtorchvision/share/cmake/TorchVision"
export LD_LIBRARY_PATH="$HOME/thirdparty/libtorchvision/lib:$LD_LIBRARY_PATH"
```

## Building

### Prerequisites
Before building, you need to have the trained SCNN checkpoint at `scnn_torch/checkpoints/best.pth`. Make sure you have the checkpoint file, otherwise the colcon build will fail.

To build:

```bash
colcon build --symlink-install --packages-select scnn_torch_backend
```

## Testing

```bash
colcon test --packages-select scnn_torch_backend --event-handlers console_direct+
```

## Project Structure

```
scnn_torch_backend/
├── cmake/
│   ├── Config.cmake           # Model configuration
│   ├── ModelGeneration.cmake  # TorchScript model generation
│   └── TestingSetup.cmake     # Test setup
├── include/scnn_torch_backend/
│   ├── config.hpp             # Constants (normalization, lane colors)
│   └── scnn_torch_backend.hpp # Main class declaration
├── models/
│   └── scnn_vgg16_288x800.pt        # Exported TorchScript model (generated)
├── script/
│   ├── export_scnn_to_pt.py   # Model export script
│   └── scnn_lane_detection.py # Python visualization script
├── src/
│   └── scnn_torch_backend.cpp # Implementation
├── test/
│   ├── test_scnn_torch_backend.cpp
│   ├── image_000.png          # Test image
│   └── image_001.png
├── CMakeLists.txt
├── package.xml
└── ReadMe.md
```

## Output Format

The `SCNNResult` struct contains:

| Field | Type | Description |
|-------|------|-------------|
| `seg_pred` | `cv::Mat` (CV_8UC3) | Colored lane segmentation mask (BGR) |
| `exist_pred` | `std::array<float, 4>` | Lane existence probabilities [0.0 - 1.0] |

Lane colors (BGR):
| Index | Lane | Color |
|-------|------|-------|
| 0 | Background | Black (0, 0, 0) |
| 1 | Lane 1 | Orange (0, 125, 255) |
| 2 | Lane 2 | Green (0, 255, 0) |
| 3 | Lane 3 | Red (0, 0, 255) |
| 4 | Lane 4 | Yellow (0, 255, 255) |
