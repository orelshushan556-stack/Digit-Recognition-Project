#include "ClassifierEvaluator.h"
#include <iostream>

double ClassifierEvaluator::evaluate(NeuralNetwork& network, const Matrix& X, const Matrix& Y) {
    Matrix predictions = network.forward(X);
    int count_right_scores = 0;
    int total_samples = X.getRows();
    int num_classes = Y.getCols();

    for (int i = 0; i < total_samples; i++) {
        double max_pred = -1.0;
        int max_pred_index = 0;

        double max_y = -1.0;
        int max_y_index = 0;

        for (int j = 0; j < num_classes; j++) {
            // Find class with highest predicted probability (Argmax)
            if (predictions(i, j) > max_pred) {
                max_pred = predictions(i, j);
                max_pred_index = j;
            }

            // Find ground truth target class
            if (Y(i, j) > max_y) {
                max_y = Y(i, j);
                max_y_index = j;
            }
        }

        // Check if the prediction matches the ground truth
        if (max_pred_index == max_y_index) {
            count_right_scores++;
        }
    }

    // Return overall accuracy percentage
    return (static_cast<double>(count_right_scores) / total_samples) * 100.0;
}