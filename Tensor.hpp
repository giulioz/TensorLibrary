#ifndef TENSOR_H
#define TENSOR_H

#include <algorithm>
#include <functional>
#include <initializer_list>
#include <memory>
#include <numeric>
#include <tuple>
#include <vector>

using DimensionsList = const std::initializer_list<size_t> &;

static size_t calcDataSize(const std::initializer_list<size_t> &dimensions) {
  return std::accumulate(std::begin(dimensions), std::end(dimensions), 1,
                         std::multiplies<double>());
}

static std::vector<size_t> calcStrides(DimensionsList dimensions) {
  std::vector<size_t> strides;

  size_t lastStride = 1;
  strides.push_back(lastStride);

  for (auto &&dim : dimensions) {
    size_t stride = lastStride * dim;
    strides.push_back(stride);
    lastStride = stride;
  }

  return strides;
}

template <typename ValueType>
class Tensor {
  std::unique_ptr<ValueType> data;
  std::vector<size_t> strides;

 public:
  Tensor(DimensionsList dimensions)
      : data(new ValueType[calcDataSize(dimensions)]) {
    strides = calcStrides(dimensions);
  }

  // initData(default)

  ValueType &operator[](DimensionsList coords) {
    size_t index = 0;
    size_t strideIndex = 0;
    for (auto &&coord : coords) {
      index += coord * strides[strideIndex];
      strideIndex++;
    }

    return data.get()[index];
  }

  void printTensor() const {}
};

#endif
