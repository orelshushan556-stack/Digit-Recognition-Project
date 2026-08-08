#include "NeuralNetwork.h"
#include <fstream>
void NeuralNetwork::add_layer(const Layer& layer) {
    layers.push_back(layer);
}

Matrix NeuralNetwork::forward(Matrix input) {
    for (auto & layer : layers) {
        input = layer.forward(input);
    }
    return input;
}

void NeuralNetwork::backward(Matrix output_gradient, double learning_rate) {
    for (int i = layers.size() - 1; i >= 0; i--) {
        output_gradient = layers[i].backward(output_gradient, learning_rate);
    }
}
bool NeuralNetwork::save_weights(const std::string& filename) const {
    // Open output file stream in binary mode
    std::ofstream file(filename, std::ios::binary);

    // Check if the file opened successfully
    if (!file.is_open()) {
        return false;
    }

    // Write total number of layers to the file
    int num_layers = static_cast<int>(layers.size());
    file.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));

    // Delegate saving to each layer using Layer::save
    for (const auto& layer : layers) {
        layer.save(file);
    }

    // Close file stream and return success
    file.close();
    return true;
}
bool NeuralNetwork::load_weights(const std::string& filename) {
    // Open input file stream in binary mode
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    // Read total number of saved layers
    int num_layers = 0;
    file.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));

    // Validate layer count matches network structure
    if (num_layers != static_cast<int>(layers.size())) {
        file.close();
        return false;
    }

    // Load weights and biases for each layer
    for (int i = 0; i < num_layers; i++) {
        layers[i].load(file);
    }

    // Close file stream and return success
    file.close();
    return true;
}