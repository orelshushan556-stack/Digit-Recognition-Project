#include "DigitRecognizer.h"
#include "ReLU.h"
#include "Softmax.h"
#include <memory>
#include <iostream>

DigitRecognizer::DigitRecognizer() : is_loaded(false) {
    nn.add_layer(Layer(784, 128, std::make_shared<ReLU>()));
    nn.add_layer(Layer(128, 10, std::make_shared<Softmax>()));

}

bool DigitRecognizer::load_model(const std::string& model_path) {
    if (nn.load_weights(model_path)) {
        is_loaded = true;
        return true;
    }
    return false;

}

 int DigitRecognizer::predict(const std::vector<double>& image_pixels) {
    if (!is_loaded) {
        return -1;
    }
    Matrix input(1,784);
    for (int i = 0; i < 784; i++) {
        input(0, i) = image_pixels[i];
    }
    Matrix output = nn.forward(input);
    double max_y = output(0, 0);
    int max_index = 0;
    for (int i = 0; i < 10; i++) {
        if (output(0, i) > max_y) {
            max_y = output(0, i);
            max_index = i;
        }
    }
    return max_index;

}