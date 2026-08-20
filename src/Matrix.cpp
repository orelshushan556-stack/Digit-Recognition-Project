#include <iostream>
#include "Matrix.h"
#include <random>
#include <cmath>
#include <fstream>


// Default Constructor: Initializes an empty matrix with null data
Matrix::Matrix() : rows(0), cols(0), data(nullptr) {}

// Constructor: Initializes a matrix with zeros
Matrix::Matrix(int r, int c) : rows(r), cols(c) {
    data = new double[r * c];
    for (int i = 0; i < r * c; i++) {
        data[i] = 0.0;
    }
}

// Copy Constructor: Performs a deep copy of another matrix
Matrix::Matrix(const Matrix& other) {
    rows = other.rows;
    cols = other.cols;
    int totalSize = rows * cols;
    data = new double[totalSize];
    for (int i = 0; i < totalSize; i++) {
        data[i] = other.data[i];
    }
}

// Move Constructor: Transfers ownership from a temporary object (O(1))
Matrix::Matrix(Matrix&& other) noexcept {
    this->rows = other.rows;
    this->cols = other.cols;
    this->data = other.data;

    // Neutralize the source object to prevent double free
    other.data = nullptr;
    other.rows = 0;
    other.cols = 0;
}

// Copy Assignment Operator: Deep copy with size check and reallocation
Matrix& Matrix::operator=(const Matrix& other) {
    if (this == &other) return *this;
    double* newData = nullptr;

    if (other.rows > 0 && other.cols > 0) {
        newData = new double[other.rows * other.cols];
        for (int i = 0; i < other.rows * other.cols; i++) {
            newData[i] = other.data[i];
        }
    }

    delete[] data;

    // 3. מעדכנים את האובייקט
    data = newData;
    rows = other.rows;
    cols = other.cols;

    return *this;
}

// Move Assignment Operator: Fast ownership transfer (O(1))
Matrix& Matrix::operator=(Matrix&& other) noexcept {
    if (this == &other) return *this;

    // Clean up current memory
    delete[] data;

    // Steal data and dimensions
    this->rows = other.rows;
    this->cols = other.cols;
    this->data = other.data;

    // Neutralize source
    other.data = nullptr;
    other.rows = 0;
    other.cols = 0;

    return *this; // Crucial for chaining and stability
}

// Accessor (Mutable): Returns reference to element at (row, col)
double& Matrix::operator()(int row, int col) {
    if (data==nullptr) {
        throw std::out_of_range("the data is null");
    }
    if (row >= rows || row < 0 || col >= cols || col < 0) {
        throw std::out_of_range("Matrix index out of bounds!");
    }
    return data[row * cols + col];
}

// Accessor (Const): Returns value at (row, col)
double Matrix::operator()(int row, int col) const {
    if (row >= rows || row < 0 || col >= cols || col < 0) {
        throw std::out_of_range("Matrix index out of bounds!");
    }
    return data[row * cols + col];
}

// Matrix Addition: Element-wise sum
Matrix Matrix::operator+(const Matrix& other) const {
    if (other.rows != rows || other.cols != cols) {
        throw std::invalid_argument("Addition failed: Matrix dimensions mismatch!");
    }
    Matrix sum_matrix(rows, cols);
    for (int i = 0; i < rows * cols; i++) {
        sum_matrix.data[i] = data[i] + other.data[i];
    }
    return sum_matrix;
}

// Scalar Multiplication
Matrix Matrix::operator*(double scalar) const {
    Matrix res(rows, cols);

    for (int i = 0; i < rows * cols; i++) {
        res.data[i] = data[i] * scalar;
    }
    return res;
}

// Matrix Multiplication: Optimized using transpose for cache efficiency
Matrix Matrix::operator*(const Matrix& other) const {
    if (this->cols != other.rows) {
        throw std::invalid_argument("Illegal multiplication: this->cols must equal other.rows.");
    }
    Matrix T = other.transpose();
    Matrix res(this->rows, other.cols);
    for (int i = 0; i < this->rows; i++) {
        for (int j = 0; j < other.cols; j++) {
            double sum = 0;
            for (int k = 0; k < this->cols; k++) {
                sum += (*this)(i, k) * T(j, k);
            }
            res(i, j) = sum;
        }
    }
    return res;
}

// Transpose: Swaps rows and columns
Matrix Matrix::transpose() const {
    Matrix T(cols, rows);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            T(j, i) = (*this)(i, j);
        }
    }
    return T;
}

// Destructor: Deallocates dynamic memory
Matrix::~Matrix() {
    delete[] data;
}

// Utility: Prints matrix to console
void Matrix::print() const {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << data[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

int Matrix::getRows() const { return rows; }
int Matrix::getCols() const { return cols; }


// Matrix Subtraction: Element-wise
Matrix Matrix::operator-(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Subtraction failed: Matrix dimensions mismatch!");
    }
    Matrix result(rows, cols);
    for (int i = 0; i < rows * cols; i++) {
        result.data[i] = data[i] - other.data[i];
    }
    return result;
}

// Hadamard Product: Element-wise multiplication
Matrix Matrix::hadamard(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Hadamard failed: Matrix dimensions mismatch!");
    }
    Matrix result(rows, cols);
    for (int i = 0; i < rows * cols; i++) {
        result.data[i] = data[i] * other.data[i];
    }
    return result;
}

// Scalar Addition: Adds a constant to all elements
Matrix Matrix::operator+(double scalar) const {
    Matrix result(rows, cols);
    for (int i = 0; i < rows * cols; i++) {
        result.data[i] = data[i] + scalar;
    }
    return result;
}

// Randomization: Fills matrix with uniform random values
void Matrix::randomize(double min, double max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(min, max);
    for (int i = 0; i < rows * cols; i++) {
        data[i] = dist(gen);
    }
}

// Fill with zeros
void Matrix::fillZero() {
    for (int i = 0; i < rows * cols; i++) {
        data[i] = 0.0;
    }
}

// Equality Operator: Comparison with epsilon for precision
bool Matrix::operator==(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) return false;
    double epsilon = 1e-9;
    for (int i = 0; i < rows * cols; i++) {
        if (std::abs(data[i] - other.data[i]) > epsilon) return false;
    }
    return true;
}
Matrix Matrix::map(const std::function<double(double)>& f) const {
    Matrix res(rows, cols);
    for (int i = 0; i < rows * cols; i++) {
        res.data[i] = f(data[i]);
    }
    return res;
}
Matrix Matrix::getSlice(int start_row, int num_rows) const {
    Matrix res(num_rows, cols);
    int start_index = start_row * cols;
    int total_elements = num_rows * cols;

    for (int i = 0; i < total_elements; i++) {
        res.data[i] = data[start_index + i]; // Reading offset from original, writing to 0..N in result
    }

    return res;
}
void Matrix::save(std::ofstream& file) const {
    file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    file.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    file.write(reinterpret_cast<const char*>(data), (rows * cols) * sizeof(double));
}
void Matrix::load(std::ifstream& file) {
    // 1. Read dimensions
    file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char*>(&cols), sizeof(cols));

    // 2. Re-allocate memory dynamically
    delete[] data;
    data = new double[rows * cols];

    // 3. Read binary buffer into newly allocated memory
    file.read(reinterpret_cast<char*>(data), (rows * cols) * sizeof(double));
}
double& Matrix::operator()(int index) {
    return data[index];
}

double Matrix::operator()(int index) const {
    return data[index];
}