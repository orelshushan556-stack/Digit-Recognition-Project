#include "Trainer.h"

#include <iostream>
#include <algorithm>

double Trainer::train_batch(const Matrix& X_batch, const Matrix& Y_batch, double learning_rate) {
Matrix y_predicted=network.forward(X_batch);
    double loss=loss_fn.loss_cost(y_predicted, Y_batch);
    Matrix loss_gradint=loss_fn.loss_gradient(y_predicted, Y_batch);
    network.backward(loss_gradint,learning_rate);
    return loss;
}

void Trainer::train(const Matrix& X_train, const Matrix& Y_train,
                    int epochs, int batch_size, double learning_rate) {

    int total_samples = X_train.getRows();

    for (int epoch = 0; epoch < epochs; epoch++) {
        double epoch_loss = 0.0;
        int num_batches = 0;

        for (int i = 0; i < total_samples; i += batch_size) {
            // Calculate how many samples are left to prevent out-of-bounds at the end of the dataset
            int current_batch_size = std::min(total_samples - i, batch_size);

            // Slice the main dataset to extract the current batch, then train on it
            Matrix X_batch = X_train.getSlice(i, current_batch_size);
            Matrix Y_batch = Y_train.getSlice(i, current_batch_size);

            double batch_loss = train_batch(X_batch, Y_batch, learning_rate);
            epoch_loss += batch_loss;
            num_batches++;
        }

        // Print average loss at the end of each epoch to track learning progress
        double average_loss = epoch_loss / num_batches;
        std::cout << "Epoch " << (epoch + 1) << "/" << epochs
                  << " - Loss: " << average_loss << std::endl;
    }
}