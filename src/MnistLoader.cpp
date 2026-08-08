#include "MnistLoader.h"
#include <fstream>
#include <cstdint>
#include <iostream>

// Helper function to swap Endianness (Motorola Big-Endian -> Intel/AMD Little-Endian)
static uint32_t swap_endian(uint32_t val) {
    return ((val << 24) & 0xFF000000) |
           ((val << 8)  & 0x00FF0000) |
           ((val >> 8)  & 0x0000FF00) |
           ((val >> 24) & 0x000000FF);
}

Matrix MnistLoader::label_to_one_hot(int label) {
    Matrix oneHot(1, 10);
    oneHot.fillZero();
    oneHot(0, label) = 1.0;
    return oneHot;
}

bool MnistLoader::load_dataset(const std::string& image_path,
                             const std::string& label_path,
                             MnistData& out_data) {
    std::ifstream image_file(image_path, std::ios::binary);
    std::ifstream label_file(label_path, std::ios::binary);

    // Detailed file-opening checks
    if (!image_file.is_open()) {
        std::cerr << "[MNIST LOADER DEBUG ERROR] Could not open IMAGE file!\n"
                  << "  Attempted Path: " << image_path << "\n";
        return false;
    }

    if (!label_file.is_open()) {
        std::cerr << "[MNIST LOADER DEBUG ERROR] Could not open LABEL file!\n"
                  << "  Attempted Path: " << label_path << "\n";
        return false;
    }

    // Read IDX header metadata
    uint32_t magic_images = 0, num_images = 0, rows = 0, cols = 0;
    uint32_t magic_labels = 0, num_labels = 0;

    image_file.read(reinterpret_cast<char*>(&magic_images), sizeof(magic_images));
    image_file.read(reinterpret_cast<char*>(&num_images), sizeof(num_images));
    image_file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    image_file.read(reinterpret_cast<char*>(&cols), sizeof(cols));

    label_file.read(reinterpret_cast<char*>(&magic_labels), sizeof(magic_labels));
    label_file.read(reinterpret_cast<char*>(&num_labels), sizeof(num_labels));

    // Convert values from Big-Endian
    magic_images = swap_endian(magic_images);
    num_images   = swap_endian(num_images);
    rows         = swap_endian(rows);
    cols         = swap_endian(cols);

    magic_labels = swap_endian(magic_labels);
    num_labels   = swap_endian(num_labels);

    std::cout << "[MNIST LOADER DEBUG INFO]\n"
              << "  Images File -> Magic: " << magic_images << ", Count: " << num_images
              << ", Rows: " << rows << ", Cols: " << cols << "\n"
              << "  Labels File -> Magic: " << magic_labels << ", Count: " << num_labels << "\n";

    if (num_images != num_labels) {
        std::cerr << "[MNIST LOADER DEBUG ERROR] Image count (" << num_images
                  << ") does not match Label count (" << num_labels << ")!\n";
        return false;
    }

    int image_size = rows * cols; // 28 * 28 = 784
    out_data.images.reserve(num_images);
    out_data.labels.reserve(num_labels);

    // Read pixel buffers and label bytes
    std::vector<uint8_t> pixel_buffer(image_size);
    uint8_t raw_label = 0;

    for (size_t i = 0; i < num_images; ++i) {
        // Read 784 pixels and 1 label byte
        image_file.read(reinterpret_cast<char*>(pixel_buffer.data()), image_size);
        label_file.read(reinterpret_cast<char*>(&raw_label), 1);

        // Convert pixels to 1x784 normalized Matrix [0.0, 1.0]
        Matrix img(1, image_size);
        for (int p = 0; p < image_size; ++p) {
            img(0, p) = static_cast<double>(pixel_buffer[p]) / 255.0;
        }

        out_data.images.push_back(img);
        out_data.labels.push_back(label_to_one_hot(raw_label));
    }

    return true;
}