#pragma once
#include "ILoss.h"

// The Mean Squared Error loss function
class MSE : public ILoss {
public:
    MSE() = default;
    ~MSE() override = default;

    // Calculates the mean of the squared differences
    double loss_cost(const Matrix& predicted, const Matrix& target) override;

    // Calculates the derivative of the MSE formula
    Matrix loss_gradient(const Matrix& predicted, const Matrix& target) override;
};