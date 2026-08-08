#include "Layer.h"
#include <utility>

Layer::Layer(int input_size, int output_size, std::shared_ptr<IActivation> activation)
    // Fix 1: Dimensions set to (input_size x output_size) for Matrix multiplication (X * W)
    : weights(input_size, output_size), biases(1, output_size), activation(std::move(activation))
{
    // Randomize weights and biases to break symmetry for training
    weights.randomize(-0.5, 0.5);
    biases.randomize(-0.5, 0.5);
}

Matrix Layer::forward(const Matrix& input) {
    // Save input for backpropagation (batch_size x input_size)
    input_cache = input;

    // Fix 2: Linear transformation Z = X * W
    Matrix Z = input * weights;

    // Add row-vector bias to each row in the batch
    for (int r = 0; r < Z.getRows(); ++r) {
        for (int c = 0; c < Z.getCols(); ++c) {
            Z(r, c) += biases(0, c);
        }
    }

    z_cache = Z;

    // Pass the result through the activation function
    return activation->forward(Z);
}

Matrix Layer::backward(const Matrix& output_gradient, double learning_rate) {
    // 1. Local layer error: dZ = output_gradient (element-wise) activation_derivative(Z)
    Matrix dZ = output_gradient.hadamard(activation->backward(z_cache));

    // 2. Weight gradient: dW = X^T * dZ (input_size x output_size)
    Matrix dW = input_cache.transpose() * dZ;

    // 3. Input gradient for previous layer: dX = dZ * W^T (batch_size x input_size)
    Matrix input_gradient = dZ * weights.transpose();

    // 4. Calculate bias gradient: sum dZ across batch rows (1 x output_size)
    Matrix dB(1, biases.getCols());
    dB.fillZero();
    for (int r = 0; r < dZ.getRows(); ++r) {
        for (int c = 0; c < dZ.getCols(); ++c) {
            dB(0, c) += dZ(r, c);
        }
    }

    // 5. Update weights and biases using batch-averaged gradients
    double batch_scale = 1.0 / dZ.getRows();
    weights = weights - (dW * (learning_rate * batch_scale));
    biases  = biases  - (dB * (learning_rate * batch_scale));

    return input_gradient;
}

void Layer::save(std::ofstream& file) const {
    weights.save(file);
    biases.save(file);
}

void Layer::load(std::ifstream& file) {
    weights.load(file);
    biases.load(file);
}