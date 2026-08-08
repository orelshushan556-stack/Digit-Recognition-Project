#include "Softmax.h"
#include <cmath>
#include <algorithm>

Matrix Softmax::forward(const Matrix& input) const {
    Matrix result(input.getRows(), input.getCols());

    for (int r = 0; r < input.getRows(); ++r) {

        // Find the maximum value in the row to prevent exponent overflow
        double max_val = input(r, 0);
        for (int c = 1; c < input.getCols(); ++c) {
            max_val = std::max(max_val, input(r, c));
        }

        // Compute exponent relative to max_val and accumulate total row sum
        double sum = 0.0;
        for (int c = 0; c < input.getCols(); ++c) {
            result(r, c) = std::exp(input(r, c) - max_val);
            sum += result(r, c);
        }

        // Normalize exponents to create a probability distribution summing to 1.0
        for (int c = 0; c < input.getCols(); ++c) {
            result(r, c) /= sum;
        }
    }

    return result;
}
Matrix Softmax::backward(const Matrix& input) const {
    return input;
}