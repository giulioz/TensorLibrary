#ifndef TENSOR_H
#define TENSOR_H

#include <functional>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <numeric>
#include <tuple>
#include <vector>

const static int VARIABLE_INDEX = -1;

using DimensionsList = const std::initializer_list<size_t>&;
using FixedDimensionsList = const std::initializer_list<int>&;

static inline size_t calcDataSize(DimensionsList dimensions) {
  return std::accumulate(std::begin(dimensions), std::end(dimensions), 1,
                         std::multiplies<double>());
}

static inline std::vector<size_t> calcStrides(DimensionsList dimensions) {
  std::vector<size_t> strides;

  size_t stride = 1;
  for (auto&& dim : dimensions) {
    strides.push_back(stride);
    stride *= dim;
  }

  return strides;
}

static inline size_t findFixedStride(std::vector<size_t> strides,
                                     FixedDimensionsList indexList) {
  size_t fixedStride = 0;
  size_t index = 0;
  for (auto&& fixedValue : indexList) {
    if (fixedValue != VARIABLE_INDEX) {
      fixedStride += fixedValue * strides[index];
    } else {
      fixedStride += strides[index];
    }
    index++;
  }

  return fixedStride;
}

static inline size_t findFixedIndex(FixedDimensionsList indexList) {
  size_t index = 0;
  for (auto&& fixedValue : indexList) {
    if (fixedValue == VARIABLE_INDEX) {
      return index;
    }
    index++;
  }

  return index;
}

static inline size_t findInitialPosition(std::vector<size_t> strides,
                                         FixedDimensionsList indexList,
                                         size_t width = 0) {
  size_t index = 0;
  size_t position = 0;
  for (auto&& fixedValue : indexList) {
    if (fixedValue != VARIABLE_INDEX) {
      position += fixedValue * strides[index];
    } else if (width != 0) {
      position += width * strides[index];
    }
    index++;
  }

  return position;
}

template <typename ValueType>
class Tensor {
 public:
  class TensorIterator
      : public std::iterator<std::random_access_iterator_tag, ValueType> {
    friend class Tensor;

   public:
    // Instance types
    using pointer = typename std::iterator<std::random_access_iterator_tag,
                                           ValueType>::pointer;
    using reference = typename std::iterator<std::random_access_iterator_tag,
                                             ValueType>::reference;
    using difference_type =
        typename std::iterator<std::random_access_iterator_tag,
                               ValueType>::difference_type;

    using iterator_category = typename std::random_access_iterator_tag;

   private:
    // Instance variables
    Tensor<ValueType> &tensor;
    size_t currentPos;

    TensorIterator(Tensor<ValueType>& tensor, size_t startPos = 0)
        : tensor(tensor), currentPos(startPos) {}

   public:
    TensorIterator(const TensorIterator& copy)
        : tensor(copy.tensor), currentPos(copy.currentPos) {}

    TensorIterator& operator=(const TensorIterator& other) {
      tensor = other.tensor;
      currentPos = other.currentPos;
      return *this;
    }

    reference operator*() { return tensor[currentPos]; }

    pointer operator->() { return &tensor[currentPos]; }

    reference operator[](const difference_type& n) {
      return tensor[n];
    }

    reference operator[](DimensionsList& coords) {
      return tensor[tensor.coordsToIndex(coords)];
    }

#pragma region Seek Operators

    TensorIterator& operator++() {
      currentPos++;
      return *this;
    }
    TensorIterator& operator--() {
      currentPos--;
      return *this;
    }

    TensorIterator operator++(int) {
      TensorIterator result(*this);
      operator++();
      return result;
    }
    TensorIterator operator--(int) {
      TensorIterator result(*this);
      operator--();
      return result;
    }

    TensorIterator operator+(const difference_type& n) const {
      return TensorIterator(tensor, currentPos + n);
    }

    TensorIterator& operator+=(const difference_type& n) {
      currentPos += n;
      return *this;
    }

    TensorIterator operator-(const difference_type& n) const {
      return TensorIterator(tensor, currentPos + n);
    }

    TensorIterator& operator-=(const difference_type& n) {
      currentPos -= n;
      return *this;
    }
#pragma endregion

#pragma region Comparison Operators

    bool operator==(const TensorIterator& other) const {
      return currentPos == other.currentPos;
    }

    bool operator!=(const TensorIterator& other) const {
      return currentPos != other.currentPos;
    }

    bool operator<(const TensorIterator& other) const {
      return currentPos < other.currentPos;
    }

    bool operator>(const TensorIterator& other) const {
      return currentPos > other.currentPos;
    }

    bool operator<=(const TensorIterator& other) const {
      return currentPos <= other.currentPos;
    }

    bool operator>=(const TensorIterator& other) const {
      return currentPos >= other.currentPos;
    }

    difference_type operator+(const TensorIterator& other) const {
      return currentPos + other.currentPos;
    }

    difference_type operator-(const TensorIterator& other) const {
      return currentPos - other.currentPos;
    }
#pragma endregion
  };

 private:
  using ArrayType = typename std::vector<ValueType>;

  ArrayType data;
  std::vector<size_t> strides;
  std::vector<size_t> sizes;
  size_t _totalItems;

  size_t coordsToIndex(DimensionsList coords) {
    size_t index = 0;
    size_t strideIndex = 0;
    for (auto&& coord : coords) {
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

  Tensor(const Tensor& other)
      : data(other.data),
        strides(other.strides),
        sizes(other.sizes),
        _totalItems(other._totalItems) {}

  // Tensor& operator=(const Tensor& other) {
  //   return *this;
  // }

  size_t totalItems() const { return _totalItems; }
  size_t itemsAt(size_t dimension) const { return sizes.at(dimension); }

  TensorIterator begin() { return TensorIterator(*this); }
  TensorIterator end() { return TensorIterator(*this, _totalItems); }

  ValueType& operator[](const size_t linearCoord) {
    return data.at(linearCoord);
  }

  ValueType& operator[](DimensionsList& coords) {
    auto index = coordsToIndex(coords);
    return data.at(index);
  }
};

#endif
