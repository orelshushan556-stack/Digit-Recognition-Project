# Define a variable containing all the core components (excluding main.cpp)
set(CORE_SOURCES
        Matrix.cpp
        ReLU.cpp
        Layer.cpp
        NeuralNetwork.cpp
        mse.cpp
        Trainer.cpp
        MnistLoader.cpp
        Sigmod.cpp
        Softmax.cpp
        CrossEntropy.cpp
        ClassifierEvaluator.cpp
        DigitRecognizer.cpp
)

# Create a static library named 'nn_core' from the core sources
add_library(nn_core ${CORE_SOURCES})

# Define the main training engine executable (ONLY main.cpp, without wasm_api.cpp)
add_executable(DigitRecognitionEngine main.cpp)

# Link the main engine to the core library
target_link_libraries(DigitRecognitionEngine nn_core)