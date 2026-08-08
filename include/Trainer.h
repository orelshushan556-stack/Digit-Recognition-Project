#pragma once
#include "NeuralNetwork.h"
#include "ILoss.h"
#include <vector>

class Trainer {
private:
    NeuralNetwork& network;
    ILoss& loss_fn;

public:
    Trainer(NeuralNetwork& net, ILoss& loss)
        : network(net), loss_fn(loss) {}

    // Trains the network for a set number of epochs
    void train(const Matrix& X_train, const Matrix& Y_train,
               int epochs, int batch_size, double learning_rate);
};