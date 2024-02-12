# KinectFusion

Project GitHub repository: https://github.com/sssemil/kinect_fusion

## Dependencies
Install Eigen
```bash
sudo apt install libeigen3-dev
```
For installing Ceres you can follow this page http://ceres-solver.org/installation.html

For installing CUDA you can follow this page For instlling Ceres you can follow this page http://ceres-solver.org/installation.html

Install Flann
```bash
sudo apt-get install libflann-dev
```

### CxxOpts

```bash
git clone https://github.com/jarro2783/cxxopts ../Libs/cxxopts
```

# Running

Store the project inside the 3DMS exercise folder, run the following commands from the project root:

```bash
cd cmake-build-release
cmake -DCMAKE_BUILD_TYPE=Release ..
OpenCV_DIR=/opt/opencv3 make -j24
cd ..
```

And run:
```bash
./cmake-build-release/main --dataset ../Data/rgbd_dataset_freiburg1_xyz/ --output out/mesh_applyBilateral_001_4_t0_001_all_tarsrc50_ --applyBilateral --dx 0 --dy 0 --dz -2 --size 4 -s 100
```
