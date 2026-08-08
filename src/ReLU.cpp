#include "ReLU.h"



Matrix ReLU::forward(const Matrix& input) const {
    // 
    return input.map([](double x) {
        return x > 0.0 ? x : 0.0;
    });
}
Matrix ReLU::backward(const Matrix& input) const {

  return input.map([](double x) {
      return x > 0.0 ? 1.0 : 0.0;


  });
}