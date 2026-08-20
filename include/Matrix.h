#pragma once
// (Or #ifndef MATRIX_H ... depending on what you use)

#include <vector>
#include <functional>
#include <fstream> // <--- MUST BE HERE IN THE HEADER FILE!
class Matrix {
private:
    int rows;
    int cols;
    double* data;

public:
    Matrix(int r, int c);
    ~Matrix();
    Matrix(const Matrix& other);
    Matrix& operator=(const Matrix& other);
    // Move Constructor - בנאי הזזה
    Matrix(Matrix&& other2) noexcept;
    Matrix();

    // Move Assignment Operator - אופרטור השמה בהזזה
    Matrix& operator=(Matrix&& other) noexcept;

    int getRows() const;
    int getCols() const;


    double& operator()(int row, int col);
    double operator()(int row, int col) const;

    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;
    Matrix hadamard(const Matrix& other) const;
    Matrix map(const std::function<double(double)>& f) const;

    Matrix operator*(double scalar) const;
    Matrix operator+(double scalar) const;

    Matrix transpose() const;

    void randomize(double min, double max);
    void fillZero();

    void applyReLU();
    Matrix reluDerivative() const;

    bool operator==(const Matrix& other) const;
    void print() const;
    // Returns a new Matrix containing a subset of rows from start_row
    [[nodiscard]] Matrix getSlice(int start_row, int num_rows) const;
    // Saves matrix dimensions and binary data buffer to an open file stream
    void save(std::ofstream& file) const;

    // Loads matrix dimensions and binary data buffer from an open file stream
    void load(std::ifstream& file);
    double& operator()(int index);
    double operator()(int index) const;

};