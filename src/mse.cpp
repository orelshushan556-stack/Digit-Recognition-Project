#include "MSE.h"

double MSE::loss_cost(const Matrix& predicted, const Matrix& target) {
    Matrix diff = predicted - target;
    Matrix diffSquared = diff.map([](double x) { return x * x; });

    int rows = diffSquared.getRows();
    int cols = diffSquared.getCols();
    double sum = 0.0;

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            sum = sum + diffSquared(r, c); // Accessing elements via (row, col) operator
        }
    }

    int N = rows * cols;
    return sum / N; // Divide by total number of elements to get the mean
}

Matrix MSE::loss_gradient(const Matrix& predicted, const Matrix& target) {
    // 1. Calculate the total number of elements N
    double N = predicted.getRows() * predicted.getCols();

    // 2. Calculate the difference matrix: (predicted - target)
    Matrix diff = predicted - target;

    // 3. Multiply the difference matrix by (2.0 / N)
    double factor = 2.0 / N;
    return diff * factor; // Uses scalar multiplication operator
}