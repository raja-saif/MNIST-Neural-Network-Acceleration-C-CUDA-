# MNIST Neural Network Acceleration (C / CUDA)

A two-layer neural network for MNIST digit classification. It starts as a sequential C baseline and goes through three CUDA versions, each adding more GPU optimizations: custom kernels, then batching with CUDA streams, then cuBLAS Tensor Core GEMMs with mixed precision. This was a three-person team project for an HPC course, and the repository includes the project report (`HPC_Project/HPC_Report.pdf`).

> This repository also has a separate, unrelated folder, `OOP Project/`, which holds a C++/SFML 2D shooter game. It is described [below](#oop-project-separate-c--sfml-game).

---

## Network

- Architecture: 784 inputs → 128 hidden units (ReLU) → 10 outputs (softmax)
- Trained on the 60,000-image MNIST training set for 3 epochs, then evaluated on the 10,000-image test set
- Loss is cross-entropy, and the program prints loss, training accuracy, time per epoch and test accuracy

## Versions

| Version | Location | What it does |
|---|---|---|
| **V1** | `HPC_Project/Files/V1/nn.c` | Sequential CPU baseline in C using nested loops (forward pass, backpropagation, SGD) |
| **V2** | `HPC_Project/Files/V2/nn.cu` | First CUDA port. Uses separate `__global__` kernels for the forward layers, ReLU, softmax and each backprop/update step. Weights are flattened into 1D device arrays and moved with `cudaMalloc`/`cudaMemcpy` |
| **V3** | `HPC_Project/Files/V3/nn.cu` | Batched training (`MAX_BATCH_SIZE = 256`) with batch-wide kernels. Uses 4 non-blocking CUDA streams to overlap transfers and compute, Xavier/Glorot initialization, softmax on the host, and `cudaEvent` GPU timers |
| **V4** | `HPC_Project/Files/V4/nn.cu` | Keeps the V3 pipeline but runs the matrix multiplies through **cuBLAS `cublasGemmEx`** with `CUBLAS_TENSOR_OP_MATH` (Tensor Cores). Mixed precision: FP16 inputs/weights for the GEMMs, FP32 activations and gradients, FP64 master weights. Custom kernels handle the conversions between double, float and half |

## Results (from `HPC_Report.pdf`)

These are the runs recorded in the report. The V2 run was on an NVIDIA GeForce RTX 3080.

| Version | Training time (3 epochs) | Test accuracy | Speedup reported |
|---|---|---|---|
| V1 (CPU) | 22.435 s | 96.85% | baseline |
| V2 (naive CUDA) | 41.660 s | 96.70% | 0.54× vs V1 |
| V3 (batched + streams) | 0.357 s | 92.70% | 116.69× vs V2 |
| V4 (cuBLAS Tensor Cores, mixed precision) | 0.307 s (GPU time, CUDA events) | 96.41% | 1.16× vs V3 |

The report also points out the limitations:
- V2 is slower than the CPU baseline. It processes one sample at a time and has no batching, no stream overlap and no shared-memory optimization.
- V3 has lower accuracy because batching and the new initialization changed the training dynamics.
- In V3 and V4, the host-side softmax is still a synchronization point.

## Tech stack

- **C** (V1, built with `gcc`)
- **CUDA C** with **nvcc**, CUDA streams, CUDA events and `cuda_fp16`
- **cuBLAS** (V4 uses `cublasGemmEx` with Tensor Core math)
- GNU Make

## Project structure

```
HPC_Project/
├── HPC_Report.pdf          # Project report (design, limitations, timings)
└── Files/
    ├── data/               # MNIST idx files (train/t10k images and labels)
    ├── V1/  nn.c, makefile # CPU baseline
    ├── V2/  nn.cu, makefile
    ├── V3/  nn.cu, makefile
    └── V4/  nn.cu, makefile (+ a copy of the MNIST files)
OOP Project/                # Separate C++/SFML game (see below)
```

## Build & run

Every version has its own makefile and reads the dataset from `../data/`, so run `make` from inside the version's folder.

```bash
cd HPC_Project/Files/V1 && make        # gcc -Wall -O2 ... -lm, then runs ./nn.exe
cd HPC_Project/Files/V2 && make        # nvcc -O2, then runs ./nn_v2
cd HPC_Project/Files/V3 && make && make run   # nvcc -O3 -arch=sm_70 ..., ./nn_gpu_opt
cd HPC_Project/Files/V4 && make && make run   # nvcc -O3 -arch=sm_70 ... -lcublas -lm, ./nn_v4
```

Requirements: an NVIDIA GPU with the CUDA toolkit for V2–V4. V4 also needs cuBLAS and a GPU with Compute Capability 7.0 or higher for Tensor Cores. The V3/V4 makefiles set `CUDA_ARCH = -arch=sm_70`, so change it to match your GPU (for example, `sm_86` for an RTX 30-series card).

> **Dataset note:** The `train-images.idx3-ubyte` files in this repository are smaller than the standard MNIST training-image file, which is about 47 MB. If loading fails, download the original MNIST idx files and put them in `HPC_Project/Files/data/`.

---

## OOP Project (separate C++ / SFML game)

`OOP Project/` is a separate project that has nothing to do with the neural network. It is a 2D space-shooter game written in C++ with **SFML** (Graphics and Audio). It includes:

- A menu screen (`menu.h`) and the main game loop (`game.h`), started from `main.cpp`
- A player ship that fires bullets, plus falling add-on items (fire, lives, danger) (`player.h`, `bullet.h`)
- An enemy class hierarchy (`Enemy.h`): `Invader` → `AlphaInvader` / `BetaInvader`, plus `GammaInvader`, `Monster` and `Dragon`. Enemies drop bombs (`Bomb.h`), and the dragon breathes fire
- Scoring, lives and level progression, plus sound effects and music (`sounds/*.wav`) and sprites (`img/`)

Example build (inferred, since the repo has no build file). Run it from inside `OOP Project/` so the relative `img/` and `sounds/` paths resolve:

```bash
g++ main.cpp -o game -lsfml-graphics -lsfml-window -lsfml-system -lsfml-audio
./game
```

---

**Author:** Raja Saif Ali — [GitHub](https://github.com/raja-saif) · [LinkedIn](https://www.linkedin.com/in/raja-saif-01702329a/)
