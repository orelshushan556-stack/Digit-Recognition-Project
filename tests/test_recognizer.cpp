#include "DigitRecognizer.h"
#include <iostream>
#include <vector>

int main() {
    DigitRecognizer recognizer;

    // 1. Safety check: Verify inference fails before model is loaded
    std::vector<double> dummy_pixels(784, 0.0);
    int test_before_load = recognizer.predict(dummy_pixels);
    if (test_before_load == -1) {
        std::cout << "[PASS] Safety check: Predict before load returned -1" << std::endl;
    } else {
        std::cerr << "[FAIL] Safety check failed: Predict before load did not return -1" << std::endl;
        return 1;
    }

    // 2. Model loading: Load pre-trained binary weights
    std::string model_path = "mnist_model.bin";
    if (!recognizer.load_model(model_path)) {
        std::cerr << "[FAIL] Failed to load model from: " << model_path << std::endl;
        std::cerr << "Ensure mnist_model.bin exists in the working directory." << std::endl;
        return 1;
    }
    std::cout << "[PASS] Model weights loaded successfully." << std::endl;

    // 3. Inference check: Predict dummy input and validate output range
    int predicted_digit = recognizer.predict(dummy_pixels);
    std::cout << "[INFO] Predicted digit for dummy input: " << predicted_digit << std::endl;

    if (predicted_digit >= 0 && predicted_digit <= 9) {
        std::cout << "[PASS] Prediction output is in the valid digit range [0-9]." << std::endl;
    } else {
        std::cerr << "[FAIL] Invalid prediction range: " << predicted_digit << std::endl;
        return 1;
    }

    std::cout << "\nAll inference checks passed successfully." << std::endl;
    return 0;
}