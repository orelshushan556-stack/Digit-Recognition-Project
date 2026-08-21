#include <iostream>
#include <vector>
#include <memory>
#include "NeuralNetwork.h"
#include "MnistLoader.h"
#include "ReLU.h"
#include "Softmax.h"
#include "CrossEntropy.h"
#include "Trainer.h"
#include "ClassifierEvaluator.h"

// Helper function to pack individual image matrices into continuous batch matrices
void pack_data(const MnistData& data, Matrix& X, Matrix& Y) {
    size_t samples = data.images.size();
    for (size_t i = 0; i < samples; ++i) {
        for (int j = 0; j < 784; ++j) {
            X(i, j) = data.images[i](0, j);
        }
        for (int j = 0; j < 10; ++j) {
            Y(i, j) = data.labels[i](0, j);
        }
    }
}

int main() {
    std::cout << "--- Starting MNIST Training with Softmax & Cross-Entropy ---\n\n";

    // 1. Load Training Dataset (60,000 samples)
    MnistData train_data;
    std::cout << "Loading training data...\n";
    if (!MnistLoader::load_dataset("C:/data/train-images-idx3-ubyte",
                                   "C:/data/train-labels-idx1-ubyte",
                                   train_data)) {
        std::cerr << "Error: Failed to load training dataset!\n";
        return 1;
    }

    // 2. Load Test Dataset (10,000 samples)
    MnistData test_data;
    std::cout << "Loading test data...\n";
    if (!MnistLoader::load_dataset("C:/data/t10k-images-idx3-ubyte",
                                   "C:/data/t10k-labels-idx1-ubyte",
                                   test_data)) {
        std::cerr << "Error: Failed to load test dataset!\n";
        return 1;
    }

    // 3. Convert vectors into continuous batch matrices
    std::cout << "Packing datasets into matrices...\n";
    Matrix X_train(train_data.images.size(), 784);
    Matrix Y_train(train_data.labels.size(), 10);
    pack_data(train_data, X_train, Y_train);

    Matrix X_test(test_data.images.size(), 784);
    Matrix Y_test(test_data.labels.size(), 10);
    pack_data(test_data, X_test, Y_test);

    // 4. Build Architecture: Input(784) -> Hidden(128, ReLU) -> Output(10, Softmax)
    std::cout << "\nBuilding Neural Network Architecture (784 -> 128 -> 10)...\n";
    NeuralNetwork nn;
    nn.add_layer(Layer(784, 128, std::make_shared<ReLU>()));
    nn.add_layer(Layer(128, 10, std::make_shared<Softmax>()));

    // 5. Initialize Components with CrossEntropy
    CrossEntropy ce_loss;
    Trainer trainer(nn, ce_loss);
    ClassifierEvaluator evaluator;

    // 6. Evaluate Baseline Accuracy (Before Training)
    double initial_acc = evaluator.evaluate(nn, X_test, Y_test);
    std::cout << "Initial Test Accuracy (Untrained): " << initial_acc << "%\n\n";

    // 7. Hyperparameters & Training Loop
    int epochs = 20;
    int batch_size = 64;
    double learning_rate = 0.01;

    std::cout << "Starting Training Loop (" << epochs << " Epochs, Batch Size: "
              << batch_size << ", Learning Rate: " << learning_rate << ")...\n";
    trainer.train(X_train, Y_train, epochs, batch_size, learning_rate);

    // 8. Final Evaluation on Test Dataset
    double final_acc = evaluator.evaluate(nn, X_test, Y_test);
    std::cout << "\n========================================\n";
    std::cout << "Final Test Accuracy (Softmax + CrossEntropy): " << final_acc << "%\n";
    std::cout << "========================================\n\n";

    // 9. Save Trained Model
    std::string model_filename = "mnist_model.bin";
    std::cout << "Saving model weights to '" << model_filename << "'...\n";
    if (nn.save_weights(model_filename)) {
        std::cout << "SUCCESS: Model saved successfully!\n";
    } else {
        std::cerr << "ERROR: Failed to save model weights.\n";
    }

    return 0;
}