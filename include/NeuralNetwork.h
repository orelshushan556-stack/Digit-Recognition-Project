#pragma once
#include "Layer.h"
#include <vector>
#include <fstream>
class NeuralNetwork {
private:
    std::vector<Layer> layers;

public:
    NeuralNetwork() = default;
    ~NeuralNetwork() = default;

    // Add a layer to the network chain
    void add_layer(const Layer& layer);

    // Forward pass through all layers (Input -> ... -> Output)
    [[nodiscard]] Matrix forward(Matrix input);

    // Backward pass through all layers in reverse order
    void backward(Matrix output_gradient, double learning_rate);
    // Save network weights and biases to a binary file
    [[nodiscard]] bool save_weights(const std::string& filename) const;

    // Load network weights and biases from a binary file
    bool load_weights(const std::string& filename);

};