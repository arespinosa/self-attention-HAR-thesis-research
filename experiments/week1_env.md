# Week 1 Environment
Here I will decsribe the steps and configurations needed, to run initial experiment (i.e  OS, Python version, TensorFlow version, CPU/GPU info, Any install issues + fixes)
- **OS:** Ubuntu 24.04.3 LTS (WSL2)
- **Python version:** 3.8.15
- **TensorFlow version:** 2.3.1
- **CPU:** Intel Core i9-14900, 32 threads
- **GPU:** NVIDIA RTX A4500 detected by nvidia-smi, TensorFlow did not detect GPU
- **Other installed packages:** numpy, scipy, scikit-learn, matplotlib, pandas, requests, tqdm, seaborn, pyyaml

## Installation Notes / Issues
- TensorFlow installed via `pip` in a conda environment.
- **GPU not detected by TensorFlow** despite NVIDIA driver being installed.
  - TF 2.3.1 requires **CUDA 10.1** and **cuDNN 7.6**; current WSL2 setup has CUDA 13.
- `protobuf` version needed to be downgraded to **3.20.3** to work with TF 2.3.1.

- **Current setup will run experiments on CPU**.