//===- ResNetBenchmark.cpp ---------------------------------------------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// This file implements the benchmark for e2e resnet.
//
//===----------------------------------------------------------------------===//

#include <benchmark/benchmark.h>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <string>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {

// MemRef descriptor.
// - T represents the type of the elements.
// - N represents the number of dimensions.
// - The storage order is NCHW.
template <typename T, size_t N> class MemRef {
public:
  // Constructor from shape.
  MemRef(intptr_t sizes[N], T init = T(0));
  MemRef(std::vector<size_t> sizes, T init = T(0));
  // Constructor from data.
  MemRef(const T *data, intptr_t sizes[N], intptr_t offset = 0);
  // Constructor from a unique_ptr, taking over.
  MemRef(std::unique_ptr<T> &uptr, intptr_t sizes[N], intptr_t offset = 0);
  // Copy constructor.
  MemRef(const MemRef<T, N> &other);
  // Copy assignment operator.
  MemRef<T, N> &operator=(const MemRef<T, N> &other);
  // Move constructor.
  MemRef(MemRef<T, N> &&other) noexcept;
  // Move assignment operator.
  MemRef<T, N> &operator=(MemRef<T, N> &&other) noexcept;
  // Desctrutor.
  ~MemRef();
  // Get the data pointer.
  T *getData();
  // Get the sizes (shape).
  const intptr_t *getSizes() { return sizes; }
  // Get the strides.
  const intptr_t *getStrides() { return strides; }
  // Get the rank of the memref.
  size_t getRank() const { return N; }
  // Get the size (number of elements).
  size_t getSize() const { return size; }
  // Get the element at index.
  const T &operator[](size_t index) const;
  T &operator[](size_t index);
  // release the pointer
  T *release();

protected:
  // Default constructor.
  // This constructor is designed for derived domain-specific constructor.
  MemRef(){};
  // Set the strides.
  // Computes the strides of the transposed tensor for transpose=true.
  void setStrides();
  // Compute the product of array elements.
  size_t product(intptr_t sizes[N]) const;

  // Data.
  // The `aligned` and `allocated` members point to the same address, `aligned`
  // member is responsible for handling data, and `allocated` member is
  // resposible for handling the memory space.
  T *allocated = nullptr;
  T *aligned = nullptr;
  // Offset.
  intptr_t offset = 0;
  // Shape.
  intptr_t sizes[N];
  // Strides.
  intptr_t strides[N];
  // Number of elements.
  size_t size;
};

// MemRef Shape Constructor.
// Construct a MemRef object from the data shape and initial value.
// The default initial value is 0.
template <typename T, std::size_t N>
MemRef<T, N>::MemRef(intptr_t sizes[N], T init) {
  for (size_t i = 0; i < N; i++) {
    this->sizes[i] = sizes[i];
  }
  setStrides();
  size = product(sizes);
  allocated = new T[size];
  aligned = allocated;
  std::fill(aligned, aligned + size, init);
}

template <typename T, std::size_t N>
MemRef<T, N>::MemRef(std::vector<size_t> sizes, T init) {
  if (sizes.size() != N) {
    throw std::runtime_error("Invalid number of dimensions.");
  }
  for (size_t i = 0; i < N; i++) {
    this->sizes[i] = sizes[i];
  }
  setStrides();
  size = product(this->sizes);
  allocated = new T[size];
  aligned = allocated;
  std::fill(aligned, aligned + size, init);
}

// MemRef Array Constructor.
// Construct a MemRef object from the data pointer, sizes, and offset.
// The default offset is 0.
template <typename T, std::size_t N>
MemRef<T, N>::MemRef(const T *data, intptr_t sizes[N], intptr_t offset) {
  this->offset = offset;
  for (size_t i = 0; i < N; i++) {
    this->sizes[i] = sizes[i];
  }
  setStrides();
  size = product(sizes);
  allocated = new T[size];
  aligned = allocated;
  for (size_t i = 0; i < size; i++) {
    aligned[i] = data[i];
  }
}

// Copy Constructor.
// This constructor is used to initialize a MemRef object with another MemRef
// object.
// - Copy `offset` and `size` directly.
// - Elementwise copy `sizes` array.
// - Calculate `strides`.
// - Allocate new space.
// - Deep copy the data from the original object.
template <typename T, std::size_t N>
MemRef<T, N>::MemRef(const MemRef<T, N> &other)
    : offset(other.offset), size(other.size) {
  for (size_t i = 0; i < N; i++) {
    this->sizes[i] = other.sizes[i];
  }
  setStrides();
  allocated = new T[size];
  aligned = allocated;
  for (size_t i = 0; i < size; i++) {
    aligned[i] = other.aligned[i];
  }
}

// Copy Assignment Operator.
// - Check if they are the same object.
// - Copy `offset` and `size` directly.
// - Elementwise copy `sizes`.
// - Calculate the `strides`.
// - Free the data space of this object to avoid memory leaks.
// - Allocate new space and deep copy.
template <typename T, std::size_t N>
MemRef<T, N> &MemRef<T, N>::operator=(const MemRef<T, N> &other) {
  if (this != &other) {
    this->offset = other.offset;
    this->size = other.size;
    for (size_t i = 0; i < N; i++) {
      this->sizes[i] = other.sizes[i];
    }
    setStrides();
    // Free the original aligned and allocated space.
    delete[] allocated;
    // Allocate new space and deep copy.
    T *ptr = new T[size];
    for (size_t i = 0; i < size; i++) {
      ptr[i] = other.aligned[i];
    }
    aligned = ptr;
    allocated = ptr;
  }
  return *this;
}

// Move Constructor.
// This constructor is used to initialize a MemRef object from a rvalue.
// The move constructor steals the resources of the original object.
// Note that the original object no longer owns the members and spaces.
// - Steal members from the original object.
// - Assign the NULL pointer to the original aligned and allocated members to
//   avoid the double free error.
template <typename T, std::size_t N>
MemRef<T, N>::MemRef(MemRef<T, N> &&other) noexcept
    : allocated(other.allocated), aligned(other.aligned), offset(other.offset),
      size(other.size) {
  std::swap(this->sizes, other.sizes);
  std::swap(this->strides, other.strides);
  // Assign the NULL pointer to the original aligned and allocated members to
  // avoid the double free error.
  other.allocated = other.aligned = nullptr;
}

// Move Assignment Operator.
// Note that the original object no longer owns the members and spaces.
// - Check if they are the same object.
// - Free the data space of this object to avoid memory leaks.
// - Steal members from the original object.
// - Assign the NULL pointer to the original aligned and allocated members to
//   avoid the double free error.
template <typename T, std::size_t N>
MemRef<T, N> &MemRef<T, N>::operator=(MemRef<T, N> &&other) noexcept {
  if (this != &other) {
    // Free the original aligned and allocated space.
    delete[] allocated;
    // Steal members of the original object.
    std::swap(strides, other.strides);
    std::swap(offset, other.offset);
    std::swap(sizes, other.sizes);
    std::swap(size, other.size);
    std::swap(allocated, other.allocated);
    std::swap(aligned, other.aligned);
    // Assign the NULL pointer to the original aligned and allocated members to
    // avoid the double free error.
    other.allocated = other.aligned = nullptr;
  }
  return *this;
}

// MemRef Destructor.
// Note that the `allocated` and `aligned` point to the same address, so it is
// enough to release the space of the `allocated` pointer in the destructor.
template <typename T, std::size_t N> MemRef<T, N>::~MemRef() {
  if (allocated)
    delete allocated;
}

// Get the data pointer.
// Return the `aligned` pointer if the container data size is greater than zero.
// If the data size is negative or zero, which means no space is allocated for
// the container data pointer, the function does not allow to return the data
// pointer.
template <typename T, std::size_t N> T *MemRef<T, N>::getData() {
  assert((size > 0) && "Invalid container data size.");
  return aligned;
}

// Get the element at index.
// Return the specific element if the container data size is greater than zero.
// If the data size is negative or zero, which means no space is allocated for
// the container data pointer, this operator does not allow to return the data
// element.
template <typename T, std::size_t N>
const T &MemRef<T, N>::operator[](size_t index) const {
  assert((size > 0) && "Invalid container data size.");
  return aligned[index + offset];
}
template <typename T, std::size_t N> T &MemRef<T, N>::operator[](size_t index) {
  assert((size > 0) && "Invalid container data size.");
  return aligned[index + offset];
}

// Calculate the stride values for each dimension based on the sizes.
template <typename T, std::size_t N> void MemRef<T, N>::setStrides() {
  assert((N > 0) && "Invalid container number of dims");
  strides[N - 1] = 1;
  if (N < 2)
    return;
  // Prevent implicit conversions between unsigned and signed
  for (std::size_t i = N - 1; i > 0; i--) {
    strides[i - 1] = strides[i] * sizes[i];
  }
}

// Calculate the total number of elements in the MemRef container.
template <typename T, std::size_t N>
size_t MemRef<T, N>::product(intptr_t sizes[N]) const {
  size_t size = 1;
  for (size_t i = 0; i < N; i++)
    size *= sizes[i];
  return size;
}
template <typename T, size_t N>
MemRef<T, N>::MemRef(std::unique_ptr<T> &uptr, intptr_t *sizes,
                     intptr_t offset) {
  if (!uptr)
    assert(0 && "Taking over an empty unique pointer.");
  T *data = uptr.release();
  this->aligned = data;
  this->allocated = data;
  this->offset = offset;
  for (size_t i = 0; i < N; i++) {
    this->sizes[i] = sizes[i];
  }
  setStrides();
  size = product(sizes);
}
template <typename T, size_t N> T *MemRef<T, N>::release() {
  T *temp = aligned;
  aligned = nullptr;
  allocated = nullptr;
  return temp;
}

// Image container.
// - T represents the type of the elements.
// - N represents the number of dimensions.
// - image represents the OpenCV Mat object.
// - norm indicates whether to perform normalization, and the normalization is
//   disabled by default.
template <typename T, size_t N> class Img : public MemRef<T, N> {
public:
  Img(cv::Mat image, intptr_t sizes[N] = nullptr, bool norm = false);

private:
  // Load image data from OpenCV Mat.
  void loadImg(cv::Mat image, bool norm);
};

// Image Constructor from OpenCV Mat.
template <typename T, size_t N>
Img<T, N>::Img(cv::Mat image, intptr_t sizes[N], bool norm) : MemRef<T, N>() {
  if (image.channels() == 1) {
    assert((N == 2) && "For gray images, the number of dimensions must be 2.");
  } else if (image.channels() == 3) {
    assert((N == 4) && "For RGB images, the number of dimensions must be 4, "
                       "either in NHWC or NCHW layout.");
  } else {
    std::cerr << "Only 2-channel gray images and 3-channel RGB images are "
                 "supported, but got images' channel equal to "
              << image.channels() << "." << std::endl;
  }
  // Use default layout setting.
  if (sizes == nullptr) {
    // The size of the gray image is represented by height and width by default.
    if (N == 2) {
      this->sizes[0] = image.rows;
      this->sizes[1] = image.cols;
    }
    // For RGB images, use NHWC layout by default.
    else if (N == 4) {
      this->sizes[0] = 1;
      this->sizes[1] = image.rows;
      this->sizes[2] = image.cols;
      this->sizes[3] = 3;
    }
  } else {
    // Use custom layout setting.
    for (size_t i = 0; i < N; i++) {
      this->sizes[i] = sizes[i];
    }
  }
  this->size = this->product(this->sizes);
  this->setStrides();
  this->allocated = new T[this->size];
  this->aligned = this->allocated;
  this->loadImg(image, norm);
}

template <typename T, size_t N>
void Img<T, N>::loadImg(cv::Mat image, bool norm) {
  // Load gray image data from OpenCV Mat.
  if (N == 2) {
    size_t k = 0;
    for (int i = 0; i < this->sizes[0]; i++) {
      for (int j = 0; j < this->sizes[1]; j++) {
        if (norm) {
          this->aligned[k] = (T)image.at<uchar>(i, j) / 255;
        } else {
          this->aligned[k] = (T)image.at<uchar>(i, j);
        }
        k++;
      }
    }
  } else if (N == 4) {
    // Detect NHWC layout of RGB image data.
    if (this->sizes[1] == image.rows && this->sizes[2] == image.cols &&
        this->sizes[3] == 3) {
      size_t k = 0;
      for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
          for (int color = 0; color < 3; color++) {
            if (norm) {
              this->aligned[k] = (T)image.at<cv::Vec3b>(i, j)[2 - color] / 255;
            } else {
              this->aligned[k] = (T)image.at<cv::Vec3b>(i, j)[2 - color];
            }
            k++;
          }
        }
      }
    }
    // Detect NCHW layout of RGB image data.
    else if (this->sizes[2] == image.rows && this->sizes[3] == image.cols &&
             this->sizes[1] == 3) {
      size_t k = 0;
      for (int color = 0; color < 3; color++) {
        for (int i = 0; i < image.rows; i++) {
          for (int j = 0; j < image.cols; j++) {
            if (norm) {
              this->aligned[k] = (T)image.at<cv::Vec3b>(i, j)[2 - color] / 255;
            } else {
              this->aligned[k] = (T)image.at<cv::Vec3b>(i, j)[2 - color];
            }
            k++;
          }
        }
      }
    } else {
      std::cerr << "RGB images must be arranged in either NHWC or NCHW layout."
                << std::endl;
    }
  }
}

// Declare the resnet C interface.
extern "C" {
void _mlir_ciface_resnet(MemRef<float, 2> *output, Img<float, 4> *input);
}

const cv::Mat imagePreprocessing() {

  cv::Mat inputImage = cv::imread(
      "../../benchmarks/DeepLearning/Models/ResNet-V2-50/Images/ice-cream.png");
  assert(!inputImage.empty() && "Could not read the image.");
  cv::Mat resizedImage;
  int imageWidth = 224;
  int imageHeight = 224;
  cv::resize(inputImage, resizedImage, cv::Size(imageWidth, imageHeight),
             cv::INTER_LINEAR);
  return resizedImage;
}

cv::Mat image = imagePreprocessing();

intptr_t sizesInput[4] = {1, image.rows, image.cols, 3};
intptr_t sizesOutput[2] = {1, 1001};

Img<float, 4> input(image, sizesInput, true);
MemRef<float, 2> output(sizesOutput);

// Define benchmark function.
void BM_ResNet(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_resnet(&output, &input);
    }
  }
}

// Softmax function.
void softmax(float *input, size_t size) {
  assert(0 <= size <= sizeof(input) / sizeof(float));
  int i;
  float m, sum, constant;
  m = -INFINITY;
  for (i = 0; i < size; ++i) {
    if (m < input[i]) {
      m = input[i];
    }
  }

  sum = 0.0;
  for (i = 0; i < size; ++i) {
    sum += exp(input[i] - m);
  }

  constant = m + log(sum);
  for (i = 0; i < size; ++i) {
    input[i] = exp(input[i] - constant);
  }
}

std::string getLabel(int idx) {
  std::ifstream in(
      "../../benchmarks/DeepLearning/Models/ResNet-V2-50/Labels.txt");
  assert(in.is_open() && "Could not read the label file.");
  std::string label;
  for (int i = 0; i < idx; ++i)
    std::getline(in, label);
  std::getline(in, label);
  in.close();
  return label;
}

} // namespace

// Register benchmarking function with different arguments.
BENCHMARK(BM_ResNet)->Arg(1)->Unit(benchmark::kMillisecond);

// Print result function.
void printResult() {
  // Run the model and activation function.
  _mlir_ciface_resnet(&output, &input);
  auto out = output.getData();
  softmax(out, 1001);
  // Find the classification and print the result.
  float maxVal = 0;
  float maxIdx = 0;
  for (int i = 0; i < 1001; ++i) {
    if (out[i] > maxVal) {
      maxVal = out[i];
      maxIdx = i;
    }
  }
  std::cout << "Classification Index: " << maxIdx << std::endl;
  std::cout << "Classification: " << getLabel(maxIdx) << std::endl;
  std::cout << "Probability: " << maxVal << std::endl;
}
