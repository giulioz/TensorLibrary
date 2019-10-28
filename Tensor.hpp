#ifndef TENSOR_H
#define TENSOR_H

#include <algorithm>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <numeric>
#include <tuple>
#include <vector>

using DimensionsList = const std::initializer_list<size_t> &;

static size_t calcDataSize(DimensionsList dimensions) {
  return std::accumulate(std::begin(dimensions), std::end(dimensions), 1,
                         std::multiplies<double>());
}

static std::vector<size_t> calcStrides(DimensionsList dimensions) {
  std::vector<size_t> strides;

  size_t lastStride = 1;
  strides.emplace(strides.begin(), lastStride);
  // strides.push_back(lastStride);

  for (auto &&dim : dimensions) {
    size_t stride = lastStride * dim;
    strides.emplace(strides.begin(), stride);
    lastStride = stride;
  }

  return strides;
}

template <typename ValueType>
class TensorIterator {
  size_t currentPos;

 public:
  TensorIterator(size_t startPos = 0) : currentPos(startPos) {}

  void operator++() { currentPos++; }
};

template <typename ValueType>
class Tensor {
  std::vector<ValueType> data;
  std::vector<size_t> strides;
  std::vector<size_t> sizes;
  size_t _totalItems;

  size_t coordsToIndex(DimensionsList coords) {
    size_t index = 0;
    size_t strideIndex = 0;
    // size_t size = strides.size();
    for (auto &&coord : coords) {
      index += coord * (strides.at(strideIndex));
      strideIndex++;
    }

    return index;
  }

 public:
  Tensor(DimensionsList sizes) : data(calcDataSize(sizes)) {
    this->sizes = sizes;
    _totalItems = data.size();
    strides = calcStrides(sizes);
  }

  size_t totalItems() const { return _totalItems; }

  size_t items(size_t dimension) const { return sizes.at(dimension); }

  TensorIterator<ValueType> begin() {}

  void initData(ValueType defaultValue) {
    // std::fill(begin(), end(), defaultValue);
  }

  ValueType &operator[](DimensionsList coords) {
    auto index = coordsToIndex(coords);
    return data.at(index);
  }

  void printTensor() const {
    for (size_t i = 0; i < _totalItems; i++) {
      for (size_t j = 0; j < sizes.size(); j++) {
        if (i % sizes.size() == 0 && i != 0) {
          std::cout << std::endl;
        }
      }
      std::cout << data[i] << " ";
    }
    std::cout << std::endl;
    for (size_t i = 0; i < _totalItems; i++) {
      std::cout << data[i] << " ";
    }
    std::cout << std::endl;
    for (size_t i = 0; i < strides.size(); i++) {
      std::cout << strides[i] << " ";
    }
  }
};

#endif