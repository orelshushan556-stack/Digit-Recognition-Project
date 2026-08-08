#include "Trainer.h"
#include <iostream>
#include <algorithm>

void Trainer::train(const Matrix& X_train, const Matrix& Y_train,
                    int epochs, int batch_size, double learning_rate) {

    int total_samples = X_train.getRows();

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double epoch_loss = 0.0;
        int num_batches = 0;

        for (int i = 0; i < total_samples; i += batch_size) {
            int current_batch_size = std::min(total_samples - i, batch_size);

            Matrix X_batch = X_train.getSlice(i, current_batch_size);
            Matrix Y_batch = Y_train.getSlice(i, current_batch_size);

            Matrix y_pred = network.forward(X_batch);

            double loss = loss_fn.loss_cost(y_pred, Y_batch);
            epoch_loss += loss;
            num_batches++;

            Matrix loss_grad = loss_fn.loss_gradient(y_pred, Y_batch);

            network.backward(loss_grad, learning_rate);
        }

        double avg_loss = epoch_loss / num_batches;
        std::cout << "Epoch " << (epoch + 1) << "/" << epochs
                  << " - Loss: " << avg_loss << std::endl;
    }
}