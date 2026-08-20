#pragma once

#include <complex.h>

#include "NeuralNetwork.h"
#include <vector>
#include <string>

class DigitRecognizer {
private:
    NeuralNetwork nn;
    bool is_loaded;

public:
    // Constructor
    DigitRecognizer();

    // Load pre-trained model weights from binary file (returns true on success)
    bool load_model(const std::string& model_path);

    // Predict digit (0-9) from a flat 784-pixel input array
        int predict(const std::vector<double>& image_pixels);
};