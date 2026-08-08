#pragma once
#include "Matrix.h"
class IActivation {
    public:
    virtual ~IActivation() = default;
    [[nodiscard]] virtual Matrix forward(const Matrix& input) const = 0;
    [[nodiscard]] virtual Matrix backward(const Matrix& input) const = 0;

};