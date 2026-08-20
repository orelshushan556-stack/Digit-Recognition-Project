#include "CrossEntropy.h"
#include <cmath>
#include <algorithm>

double CrossEntropy::loss_cost(const Matrix& predicted, const Matrix& target) {
    double total_loss = 0.0;
    int rows = predicted.getRows();
    int cols = predicted.getCols();
    int total_elements = rows * cols;

    // Epsilon to prevent numerical instability from log(0)
    const double eps = 1e-15;

    for (int i = 0; i < total_elements; i++) {
        // Active ground truth label in One-Hot vector (target == 1.0)
        if (target(i) > 0.5) {
            // Clip predicted probability to the safe range [eps, 1.0 - eps]
            double p = std::clamp(predicted(i), eps, 1.0 - eps);

            // Accumulate negative log-likelihood
            total_loss -= std::log(p);
        }
    }

    // Return mean cross-entropy loss per sample across the batch
    return total_loss / rows;
}

Matrix CrossEntropy::loss_gradient(const Matrix& predicted, const Matrix& target) {
    Matrix grad(predicted.getRows(), predicted.getCols());
    int rows = predicted.getRows();
    int cols = predicted.getCols();
    int total_elements = rows * cols;
    for (int i = 0; i < total_elements; i++) {
        grad(i)=predicted(i)-target(i);
    }


    return grad;
}