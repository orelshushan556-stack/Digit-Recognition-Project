#include "Sigmoid.h"
#include <cmath> // Required for std::exp()

Matrix Sigmoid::forward(const Matrix& input) const {
    // Apply the sigmoid function to each element
    return input.map([](double x) {
        double result = 1.0 / (1.0 + std::exp(-x));
        return result;
    });
}

Matrix Sigmoid::backward(const Matrix& input) const {
    // Apply the derivative of the sigmoid function to each element
    return input.map([](double x) {
        double result = 1.0 / (1.0 + std::exp(-x));
        return result * (1.0 - result);
    });
}