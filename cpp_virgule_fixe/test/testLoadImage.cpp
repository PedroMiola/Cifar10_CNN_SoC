#include "../include/loadImage.hpp"   // provides loadImagesFromFile, LabeledImage, label_t, data_t, IMAGE_* macros
#include "testUtils.hpp"

#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <filesystem>
#include <type_traits>
#include <cstddef>

// Helper: deterministic integer-valued pixels (exact for float)
static inline data_t mkpix(std::size_t img, int ch, std::size_t h, std::size_t w) {
    const int base = (ch + 1) * 10000;
    return static_cast<data_t>(base + static_cast<int>(img) * 1000 + static_cast<int>(h) * 100 + static_cast<int>(w));
}

// Write N images in the exact format the loader expects.
static void write_bin_dataset(const std::string& path, std::size_t N) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    for (std::size_t i = 0; i < N; ++i) {
        label_t lbl = static_cast<label_t>(i + 7); // arbitrary label base
        f.write(reinterpret_cast<const char*>(&lbl), sizeof(label_t));
        for (std::size_t h = 0; h < IMAGE_HEIGHT; ++h) {
            for (std::size_t w = 0; w < IMAGE_WIDTH; ++w) {
                data_t r = mkpix(i, 0, h, w);
                data_t g = mkpix(i, 1, h, w);
                data_t b = mkpix(i, 2, h, w);
                f.write(reinterpret_cast<const char*>(&r), sizeof(data_t));
                f.write(reinterpret_cast<const char*>(&g), sizeof(data_t));
                f.write(reinterpret_cast<const char*>(&b), sizeof(data_t));
            }
        }
    }
}

int main() {
    std::filesystem::create_directories("../log");
    std::ofstream log("../log/outputLoadImage.log");
    if (!log) {
        std::cerr << "Error: Unable to open ../log/outputLoadImage.log\n";
        return 1;
    }

    int failures = 0;

    // ---------- Test 0: Round-trip load of K images ----------
    {
        const std::string bin = "../log/images_k2.bin";
        const std::size_t K = 2;
        write_bin_dataset(bin, K);

        std::vector<LabeledImage<IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH>> out(K);
        loadImagesFromFile(bin, out.data(), K);

        bool ok = true;
        for (std::size_t i = 0; i < K && ok; ++i) {
            label_t expectedLbl = static_cast<label_t>(i + 7);
            if (out[i].label != expectedLbl) { ok = false; break; }
            for (std::size_t h = 0; h < IMAGE_HEIGHT && ok; ++h) {
                for (std::size_t w = 0; w < IMAGE_WIDTH; ++w) {
                    if (out[i].img[0][h][w] != mkpix(i, 0, h, w) ||
                        out[i].img[1][h][w] != mkpix(i, 1, h, w) ||
                        out[i].img[2][h][w] != mkpix(i, 2, h, w)) { ok = false; break; }
                }
            }
        }
        logExpect(ok, failures, log, "Round-trip: labels and pixels match for K images");
    }

    // ---------- Test 1: numImages < available (partial load) ----------
    {
        const std::string bin = "../log/images_k3.bin";
        const std::size_t total = 3;
        const std::size_t want = 2;
        write_bin_dataset(bin, total);

        // Pre-fill with sentinels to detect unintended writes
        LabeledImage<IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH> arr[total]{};
        for (std::size_t h = 0; h < IMAGE_HEIGHT; ++h)
            for (std::size_t w = 0; w < IMAGE_WIDTH; ++w)
                for (int c = 0; c < 3; ++c)
                    arr[2].img[c][h][w] = static_cast<data_t>(-12345);

        arr[2].label = static_cast<label_t>(222);

        loadImagesFromFile(bin, arr, want);

        bool firstTwoOK = true;
        for (std::size_t i = 0; i < want && firstTwoOK; ++i) {
            if (arr[i].label != static_cast<label_t>(i + 7)) { firstTwoOK = false; break; }
            for (std::size_t h = 0; h < IMAGE_HEIGHT && firstTwoOK; ++h)
                for (std::size_t w = 0; w < IMAGE_WIDTH; ++w)
                    if (arr[i].img[0][h][w] != mkpix(i, 0, h, w) ||
                        arr[i].img[1][h][w] != mkpix(i, 1, h, w) ||
                        arr[i].img[2][h][w] != mkpix(i, 2, h, w)) { firstTwoOK = false; break; }
        }

        bool thirdUntouched = (arr[2].label == static_cast<label_t>(222));
        for (std::size_t h = 0; h < IMAGE_HEIGHT && thirdUntouched; ++h)
            for (std::size_t w = 0; w < IMAGE_WIDTH; ++w)
                for (int c = 0; c < 3; ++c)
                    if (arr[2].img[c][h][w] != static_cast<data_t>(-12345)) { thirdUntouched = false; break; }

        logExpect(firstTwoOK, failures, log, "Loads exactly numImages (first two match)");
        logExpect(thirdUntouched, failures, log, "Does not write beyond numImages");
    }

    {
        const std::string bin_path = "../../cifar-10-binary/cifar-10-batches-cropped-bin/test_batch.bin";
        const std::size_t num_images = 10000;

        LabeledImage<IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH>* images = new LabeledImage<IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH>[num_images];
        loadImagesFromFile(bin_path, images, num_images);

        // Print first 5x5 image pixel values for visual inspection
        log << "First image label: " << static_cast<int>(images[0].label) << "\n";
        log << "First image pixels (first 5x5 of each channel):\n";
        for (int ch = 0; ch < IMAGE_CHANNELS; ++ch) {
            log << "Channel " << ch << ":\n";
            for (std::size_t h = 0; h < 5; ++h) {
                for (std::size_t w = 0; w < 5; ++w) {
                    log << images[0].img[ch][h][w] << " ";
                }
                log << "\n";
            }
        }  
        delete[] images;
    }

    std::filesystem::remove("../log/images_k1.bin");
    std::filesystem::remove("../log/images_k2.bin");
    std::filesystem::remove("../log/images_k3.bin");
    if (failures == 0) log << "All loadImage tests PASSED.\n"; 
    else log << failures << " loadImage test(s) FAILED.\n";
    log.close();
    return failures;
}
