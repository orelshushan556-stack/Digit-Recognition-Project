#pragma once
#include "CrossEntropy.h"
#include "NeuralNetwork.h"
#include "ILoss.h"

class Trainer {
private:
    NeuralNetwork& network;
    ILoss& loss_fn;

public:
    Trainer(NeuralNetwork& net, ILoss& loss)
        : network(net), loss_fn(loss) {}

    // Trains on a single batch and returns the computed batch loss
    double train_batch(const Matrix& X_batch, const Matrix& Y_batch, double learning_rate);

    // Runs the full training loop across all epochs and batches
    void train(const Matrix& X_train, const Matrix& Y_train,
               int epochs, int batch_size, double learning_rate);
};