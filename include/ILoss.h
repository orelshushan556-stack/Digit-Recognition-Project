#pragma once
#include "Matrix.h"

class ILoss {
public:
    // Virtual destructor for safe inheritance
    virtual ~ILoss() = default;

    // Calculates the error score (a single number)
    virtual double loss_cost(const Matrix& predicted, const Matrix& target) = 0;

    // Calculates the error gradient (the matrix to start backpropagation)
    virtual Matrix loss_gradient(const Matrix& predicted, const Matrix& target) = 0;
};