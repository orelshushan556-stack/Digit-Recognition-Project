#pragma once
#include "IActivation.h"
#include "Matrix.h"
#include <memory>
#include <fstream>

class Layer {
private:
    Matrix weights;
    Matrix biases;
    Matrix input_cache;
    Matrix z_cache;
    std::shared_ptr<IActivation> activation;

public:
    // Initialize layer with weights, biases, and activation function
    Layer(int input_size, int output_size, std::shared_ptr<IActivation> activation);

    ~Layer() = default;

    // Perform forward pass: Z = W * X + B, then apply activation
    [[nodiscard]] Matrix forward(const Matrix& input);



    [[nodiscard]] Matrix backward(const Matrix& output_gradient, double learning_rate);
    // In Layer.h under public:
    void save(std::ofstream& file) const;
    void load(std::ifstream& file);

};