#pragma once
#include "iEvaluator.h"

class ClassifierEvaluator : public iEvaluator {
public:
    double evaluate(NeuralNetwork& network, const Matrix& X, const Matrix& Y) override;
};