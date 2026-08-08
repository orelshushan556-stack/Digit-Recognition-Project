#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include <string>
#include <vector>
#include "Matrix.h"

// Struct to hold a loaded dataset (e.g., training set or test set)
struct MnistData {
    std::vector<Matrix> images; // Each image is a 1x784 Matrix (normalized 0.0 - 1.0)
    std::vector<Matrix> labels; // Each label is a 1x10 One-Hot encoded Matrix
};

class MnistLoader {
public:
    // Helper function to convert a raw digit (0-9) into a 1x10 One-Hot Matrix
    static Matrix label_to_one_hot(int label);

    // Loads images and labels from raw IDX binary files
    static bool load_dataset(const std::string& image_path,
                             const std::string& label_path,
                             MnistData& out_data);
};

#endif // MNIST_LOADER_H