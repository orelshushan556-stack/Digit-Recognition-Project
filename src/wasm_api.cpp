#include <emscripten/bind.h>
#include <vector>
#include "../include/DigitRecognizer.h"

using namespace emscripten;

EMSCRIPTEN_BINDINGS(digit_recognizer_module){
    // Expose std::vector<double> to JavaScript so we can pass the 784 pixels
    register_vector<double>("VectorDouble");

    // Bind the DigitRecognizer class and its methods to the WebAssembly module
    class_<DigitRecognizer>("DigitRecognizer")
        .constructor<>()
        .function("predict", &DigitRecognizer::predict)
        .function("load_model", &DigitRecognizer::load_model); // Exposed for JS
}