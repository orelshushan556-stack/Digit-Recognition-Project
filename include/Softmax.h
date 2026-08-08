#pragma once

#include "IActivation.h"

class Softmax : public IActivation {
public:
    Softmax() = default;

    [[nodiscard]] Matrix forward(const Matrix& input) const override;
    [[nodiscard]] Matrix backward(const Matrix& input) const override;
};