#pragma once

#include "ILoss.h"

class CrossEntropy : public ILoss {
public:
    CrossEntropy() = default;
    ~CrossEntropy() override = default;

    // Calculates the Categorical Cross Entropy Loss
    double loss_cost(const Matrix& predicted, const Matrix& target) override;

    // Calculates the gradient (predicted - target)
    Matrix loss_gradient(const Matrix& predicted, const Matrix& target) override;
};