# C++ Neural Network Engine (WebAssembly) 🚀

![C++](https://img.shields.io/badge/C++-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white)
![WebAssembly](https://img.shields.io/badge/WebAssembly-654FF0?style=for-the-badge&logo=webassembly&logoColor=white)
![TypeScript](https://img.shields.io/badge/TypeScript-007ACC?style=for-the-badge&logo=typescript&logoColor=white)
![Vite](https://img.shields.io/badge/Vite-B73BFE?style=for-the-badge&logo=vite&logoColor=FFD62E)

A high-performance, fully connected neural network built **100% from scratch in pure C++** to recognize handwritten digits (MNIST). The engine is compiled into **WebAssembly (Wasm)**, allowing the heavy matrix computations to run natively and instantly in the browser without relying on any backend server.

🌟 **[Try the Live Demo Here!](https://digit-recognition-project-five.vercel.app)** 🌟

## 🧠 The Architecture & Engineering
This project was built to demonstrate a deep understanding of low-level system design, memory management, and machine learning mathematics without abstracting the difficulty away with external libraries (No PyTorch, No TensorFlow, No Eigen).

### Key Technical Highlights:
* **Engineered From Scratch:** Custom implementation of forward propagation, backpropagation, Loss calculation (Cross-Entropy), and activation functions (**ReLU** & **Softmax**).
* **Cache-Optimized Memory:** Matrices are intentionally structured as **1D Flat Arrays** instead of standard 2D arrays. This requires manual index calculation (`row * cols + col`) but ensures contiguous memory allocation, drastically reducing CPU cache misses and maximizing performance.
* **Client-Side AI:** The entire C++ engine is compiled to WebAssembly using Emscripten. The trained weights are loaded directly in the browser, providing near-native execution speed.
* **Smart Pre-processing:** The frontend includes a custom algorithm to calculate a bounding box around the user's drawing, dynamically crop, scale, and center the digit into a 20x20 pixel grid within the 28x28 canvas, mimicking the exact format of the original MNIST dataset for higher prediction accuracy.

## 🛠️ Tech Stack
* **Core Engine:** C++17
* **Wasm Compilation:** Emscripten
* **Frontend UI:** TypeScript, HTML5 Canvas, CSS3
* **Build Tool:** CMake (Backend) & Vite (Frontend)
* **Deployment:** Vercel

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/orelshushan556-stack/Digit-Recognition-Project.git](https://github.com/orelshushan556-stack/Digit-Recognition-Project.git)
   cd Digit-Recognition-Project