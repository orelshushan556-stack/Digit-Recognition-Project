#pragma once
#include "IActivation.h"

class Sigmoid : public IActivation {
public:
    // Added [[nodiscard]] and const to perfectly match IActivation
    [[nodiscard]] Matrix forward(const Matrix& input) const override;

    [[nodiscard]] Matrix backward(const Matrix& input) const override;
};;