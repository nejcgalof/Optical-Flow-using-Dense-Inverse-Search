# Optical flow using dense inverse search

C++ implementation of this paper:

`@inproceedings{kroegerECCV2016,
   Author    = {Till Kroeger and Radu Timofte and Dengxin Dai and Luc Van Gool},
   Title     = {Fast Optical Flow using Dense Inverse Search},
   Booktitle = {Proceedings of the European Conference on Computer Vision ({ECCV})},
   Year      = {2016}} `

We skip step Fast Variational refinement.

For testing use [MPI Sintel Dataset](http://sintel.is.tue.mpg.de/downloads).

## Prerequisites

### Windows

- Install [CMake](https://cmake.org/download/). We recommend to add CMake to path for easier console using.
- Install [opencv 2.4](https://github.com/opencv/opencv) from sources.
    - Get OpenCV [(github)](https://github.com/opencv/opencv) and put in on C:/ (It can be installed somewhere else, but it's recommended to be close to root dir to avoid too long path error). `git clone https://github.com/opencv/opencv`
    - Checkout on 2.4 branch `git checkout 2.4`.
    - Make build directory .
    - In build directory create project with cmake or cmake-gui (enable `BUILD_EXAMPLES` for later test).
    - Open project in Visual Studio.
    - Build Debug and Release versions.
    - Build `INSTALL` project.
    - Add `opencv_dir/build/bin/Release` and `opencv_dir/build/bin/Debug` to PATH variable. 
    - Test installation by running examples in `opencv/build/install/` dir.
- Install [Eigen3](https://bitbucket.org/eigen/eigen/get/3.3-beta1.zip). 

## Installing
```
git clone https://github.com/nejcgalof/Optical-Flow-using-Dense-Inverse-Search.git
```

## Build
You can use cmake-gui or write similar like this:
```
mkdir build
cd build
cmake -G "Visual Studio 14 2015 Win64" -DOpenCV_DIR="C:/opencv/build" -DEIGEN3_INCLUDE_DIR="c:/eigen" ..
```

## Usage

Three options for run program. If parameter not set, using default.

```
OpticalFlow.exe
```

```
OpticalFlow.exe folder start_num_image end_num_image
```

```
OpticalFlow.exe folder start_num_image end_num_image max_iter patch_size coarsest_scale finest_scale patch_overlap patch_norm draw_grid
```

Parameters:
```
1. Folder                  (default: 5)
2. Start image number      (default: 1)
3. End image number        (default: 50)
4. Max iterations          (default: 1000)
5. Patch size              (default: 8)
6. Coarsest_scale          (default: 3)
7. Finest_scale            (default: 0)
8. Patch overlap           (default: 0.7)
9. Mean-normalize patches  (default: 1/true)
10. Draw grids             (default: 0/false)
```

## Results

![Alley_1](./results/alley_1.gif)
![Ambush_7](./results/ambush_7.gif)
![Bandage_2](./results/bandage_2.gif)