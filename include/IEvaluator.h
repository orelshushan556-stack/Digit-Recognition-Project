#pragma once
#include "Matrix.h"
#include "NeuralNetwork.h"
class iEvaluator {
    public:
    virtual ~iEvaluator() = default;
    virtual double evaluate(NeuralNetwork& network,const Matrix& X,const Matrix& Y)=0;


};