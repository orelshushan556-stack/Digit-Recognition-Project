#pragma once
#include "IActivation.h"
class ReLU :public IActivation {
    public:
    ReLU()= default;
    ~ReLU() override= default;
  [[nodiscard]] Matrix forward(const Matrix& input) const  override;
    [[nodiscard]] Matrix backward(const Matrix& input) const override;

};


