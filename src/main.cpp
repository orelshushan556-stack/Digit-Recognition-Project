#include <iostream>
#include <vector>
#include <memory>
#include "NeuralNetwork.h"
#include "MnistLoader.h"
#include "ReLU.h"
#include "Sigmoid.h"
#include "MSE.h"
#include "Trainer.h"

int main() {
    std::cout << "--- Starting MNIST Training Pipeline ---\n\n";

    // 1. Load MNIST Training Dataset
    MnistData train_data;
    std::cout << "Loading MNIST dataset from binary files...\n";

    // Correct filenames with hyphens matching C:/data/
    if (!MnistLoader::load_dataset("C:/data/train-images-idx3-ubyte",
                                   "C:/data/train-labels-idx1-ubyte",
                                   train_data)) {
        std::cerr << "Error: Failed to load MNIST dataset! Check file paths.\n";
        return 1;
    }

    size_t total_samples = train_data.images.size();
    std::cout << "Successfully loaded " << total_samples << " training samples.\n\n";

    // 2. Convert Data: Pack std::vector<Matrix> into a single continuous Matrix
    // The Trainer expects a single Matrix for X (total_samples x 784) and Y (total_samples x 10)
    std::cout << "Packing data for batch training...\n";
    Matrix X_train(total_samples, 784);
    Matrix Y_train(total_samples, 10);

    for (size_t i = 0; i < total_samples; ++i) {
        for (int j = 0; j < 784; ++j) {
            // Unpack 1x784 image matrix into the large training batch matrix
            X_train(i, j) = train_data.images[i](0, j);
        }
        for (int j = 0; j < 10; ++j) {
            // Unpack 1x10 one-hot label matrix into the large training label matrix
            Y_train(i, j) = train_data.labels[i](0, j);
        }
    }

    // 3. Build Architecture: Input(784) -> Hidden(128) -> Output(10)
    std::cout << "Initializing Neural Network Architecture (784 -> 128 -> 10)...\n";
    NeuralNetwork nn;

    auto relu = std::make_shared<ReLU>();
    auto sigmoid = std::make_shared<Sigmoid>();

    nn.add_layer(Layer(784, 128, relu));   // Hidden Layer
    nn.add_layer(Layer(128, 10, sigmoid)); // Output Layer

    // 4. Initialize Loss Function and Trainer
    MSE mse_loss;
    Trainer trainer(nn, mse_loss);

    // 5. Training Hyperparameters
    int epochs = 5;
    int batch_size = 32;
    double learning_rate = 0.01;

    std::cout << "\nStarting Training Loop (" << epochs << " Epochs, Batch Size: "
              << batch_size << ", Learning Rate: " << learning_rate << ")...\n";

    // 6. Execute Batch Training
    trainer.train(X_train, Y_train, epochs, batch_size, learning_rate);

    // 7. Save Model Weights to Binary File
    std::string model_filename = "mnist_model.bin";
    std::cout << "\nSaving trained model weights to '" << model_filename << "'...\n";
    if (nn.save_weights(model_filename)) {
        std::cout << "SUCCESS: Model saved successfully!\n";
    } else {
        std::cerr << "ERROR: Failed to save model weights.\n";
    }

    return 0;
}