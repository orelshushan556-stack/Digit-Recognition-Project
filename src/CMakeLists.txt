add_executable(DigitRecognitionEngine main.cpp Matrix.cpp
        ../include/IActivation.h
        ../include/ReLU.h
        ReLU.cpp
        ../include/Layer.h
        Layer.cpp
        ../include/NeuralNetwork.h
        NeuralNetwork.cpp
        ../include/ILoss.h
        ../include/MSE.h
        mse.cpp
        ../include/Trainer.h
        Trainer.cpp
        ../include/MnistLoader.h
        MnistLoader.cpp
        ../include/Sigmoid.h
        Sigmod.cpp
        ../include/Softmax.h
        Softmax.cpp
        ../include/CrossEntropy.h
        CrossEntropy.cpp
        ../include/IEvaluator.h
        ClassifierEvaluator.cpp
        ../include/ClassifierEvaluator.h)


target_include_directories(DigitRecognitionEngine PUBLIC ${CMAKE_SOURCE_DIR}/include)
