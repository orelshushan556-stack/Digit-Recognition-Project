#pragma once

#include "ILoss.h"

class CrossEntropy : public ILoss {
public:
    CrossEntropy() = default;

    // Computes Categorical Cross Entropy Loss with numerical stability clipping
    [[nodiscard]] double compute_loss(const Matrix& predictions, const Matrix& targets) const override;

    // Returns combined gradient for Softmax + CrossEntropy: (Y_pred - Y_true)
    [[nodiscard]] Matrix compute_gradient(const Matrix& predictions, const Matrix& targets) const override;
};