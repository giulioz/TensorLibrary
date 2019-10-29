#ifndef TENSOR_H
#define TENSOR_H

#include <functional>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <tuple>
#include <vector>

using DimensionsList = const std::initializer_list<size_t>&;

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

template <typename ValueType>
class Tensor {
 public:
  template <typename ValueTypeIter>
  class TensorIterator : public std::iterator<std::random_access_iterator_tag,
                                              ValueTypeIter, int> {
    friend class Tensor;
    Tensor* tensor;
    size_t currentPos;

    using pointer = typename std::iterator<std::random_access_iterator_tag,
                                           ValueTypeIter>::pointer;
    using reference = typename std::iterator<std::random_access_iterator_tag,
                                             ValueTypeIter>::reference;
    using difference_type =
        typename std::iterator<std::random_access_iterator_tag,
                               ValueTypeIter>::difference_type;

    TensorIterator(Tensor* tensor, size_t startPos = 0)
        : tensor(tensor), currentPos(startPos) {}

    /*
    TensorIterator(Tensor* tensor, DimensionsList fixed ,size_t startPos = 0)
      : tensor(tensor), currentPos(startPos) {}

      iterators: the class must provide random-access iterators to the full
    content of the tensor or to the content along any one index, keeping the
    other indices fixed
    */

   public:
    TensorIterator& operator=(const TensorIterator<ValueTypeIter>& other) {
      currentPos = other.currentPos;
      return *this;
    }

    reference operator*() const { return (*tensor)[currentPos]; }

    pointer operator->() const { return &((*tensor)[currentPos]); }

    TensorIterator& operator++() {
      ++currentPos;
      return *this;
    }

    TensorIterator& operator--() {
      --currentPos;
      return *this;
    }

    TensorIterator operator++(int) {
      return TensorIterator(tensor, currentPos++);
    }

    TensorIterator operator--(int) {
      return TensorIterator(tensor, currentPos--);
    }

    TensorIterator operator+(const difference_type& n) const {
      return TensorIterator(tensor, (currentPos + n));
    }

    TensorIterator& operator+=(const difference_type& n) {
      currentPos += n;
      return *this;
    }

    TensorIterator operator-(const difference_type& n) const {
      return TensorIterator(tensor, (currentPos - n));
    }

    TensorIterator& operator-=(const difference_type& n) {
      currentPos -= n;
      return *this;
    }

    reference operator[](const difference_type& n) const {
      return (*tensor)[currentPos + n];
    }

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
  };

  template <typename ValueTypeIter>
  class TensorIteratorFixed
      : public std::iterator<std::random_access_iterator_tag, ValueTypeIter,
                             int> {
    friend class Tensor;
    Tensor* tensor;
    size_t currentPos;
    size_t stride;
    // size_t width; ?

    using pointer = typename std::iterator<std::random_access_iterator_tag,
                                           ValueTypeIter>::pointer;
    using reference = typename std::iterator<std::random_access_iterator_tag,
                                             ValueTypeIter>::reference;
    using difference_type =
        typename std::iterator<std::random_access_iterator_tag,
                               ValueTypeIter>::difference_type;

    TensorIteratorFixed(Tensor* tensor, size_t fixedIndex, size_t startPos = 0)
        : tensor(tensor),
          currentPos(startPos),
          stride(tensor.strides[fixedIndex]) {}

    /*
    TensorIteratorFixed(Tensor* tensor, DimensionsList fixed ,size_t startPos =
    0) : tensor(tensor), currentPos(startPos) {}

      iterators: the class must provide random-access iterators to the full
    content of the tensor or to the content along any one index, keeping the
    other indices fixed
    */

   public:
    TensorIteratorFixed& operator=(
        const TensorIteratorFixed<ValueTypeIter>& other) {
      currentPos = other.currentPos;
      return *this;
    }

    reference operator*() const { return (*tensor)[currentPos]; }

    pointer operator->() const { return &((*tensor)[currentPos]); }

    TensorIteratorFixed& operator++() {
      currentPos += stride;
      return *this;
    }

    TensorIteratorFixed& operator--() {
      currentPos -= stride;
      return *this;
    }

    TensorIteratorFixed operator++(int) {
      currentPos += stride;
      return TensorIteratorFixed(tensor, currentPos - stride);
    }

    TensorIteratorFixed operator--(int) {
      currentPos -= stride;
      return TensorIteratorFixed(tensor, currentPos + stride);
    }

    TensorIteratorFixed operator+(const difference_type& n) const {
      return TensorIteratorFixed(tensor, (currentPos + (n * stride)));
    }

    TensorIteratorFixed& operator+=(const difference_type& n) {
      currentPos += (n * stride);
      return *this;
    }

    TensorIteratorFixed operator-(const difference_type& n) const {
      return TensorIteratorFixed(tensor, (currentPos - (n * stride)));
    }

    TensorIteratorFixed& operator-=(const difference_type& n) {
      currentPos -= (n * stride);
      return *this;
    }

    reference operator[](const difference_type& n) const {
      return (*tensor)[currentPos + (n * stride)];
    }

    bool operator==(const TensorIteratorFixed& other) const {
      return currentPos == other.currentPos;
    }

    bool operator!=(const TensorIteratorFixed& other) const {
      return currentPos != other.currentPos;
    }

    bool operator<(const TensorIteratorFixed& other) const {
      return currentPos < other.currentPos;
    }

    bool operator>(const TensorIteratorFixed& other) const {
      return currentPos > other.currentPos;
    }

    bool operator<=(const TensorIteratorFixed& other) const {
      return currentPos <= other.currentPos;
    }

    bool operator>=(const TensorIteratorFixed& other) const {
      return currentPos >= other.currentPos;
    }

    difference_type operator+(const TensorIteratorFixed& other) const {
      return currentPos + other.currentPos;
    }

    difference_type operator-(const TensorIteratorFixed& other) const {
      return currentPos - other.currentPos;
    }
  };

 private:
  std::vector<ValueType> data;
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
  using iterator = TensorIterator<ValueType>;
  using const_iterator = TensorIterator<const ValueType>;

  using iteratorFixed = TensorIteratorFixed<ValueType>;
  using const_iteratorFixed = TensorIteratorFixed<const ValueType>;

  Tensor(DimensionsList sizes) : data(calcDataSize(sizes)) {
    this->sizes = sizes;
    _totalItems = data.size();
    strides = calcStrides(sizes);
  }

  size_t totalItems() const { return _totalItems; }
  size_t items(size_t dimension) const { return sizes.at(dimension); }

  iterator begin() { return iterator(this); }
  const_iterator begin() const { return const_iterator(this); }
  const_iterator cbegin() const { return const_iterator(this); }
  iterator end() { return iterator(this, _totalItems); }
  const_iterator end() const { return const_iterator(this, _totalItems); }
  const_iterator cend() const { return const_iterator(this, _totalItems); }

  iteratorFixed begin(fixedIndex) { return iterator(this, fixedIndex); }
  const_iteratorFixed begin(fixedIndex) const {
    return const_iterator(this, fixedIndex);
  }
  const_iteratorFixed cbegin(fixedIndex) const {
    return const_iterator(this, fixedIndex);
  }
  iteratorFixed end(fixedIndex) {
    return iterator(this, fixedIndex, _totalItems);
  }
  const_iteratorFixed end(fixedIndex) const {
    return const_iterator(this, fixedIndex, _totalItems);
  }
  const_iteratorFixed cend(fixedIndex) const {
    return const_iterator(this, fixedIndex, _totalItems);
  }

  ValueType& operator[](int linearCoord) { return data.at(linearCoord); }

  ValueType& operator[](DimensionsList coords) {
    auto index = coordsToIndex(coords);
    return data.at(index);
  }

  void printTensor() const {
    // for (size_t i = 0; i < _totalItems;currentPosi++) {
    //   for (size_t j = 0; j < sizes.size(); j++) {
    //     if (i % sizes.size() == 0 && i != 0) {
    //       std::cout << std::endl;
    //     }
    //   }
    //   std::cout << data[i] << ", ";
    // }
    // std::cout << std::endl;

    std::cout << std::endl << "Data" << std::endl;
    for (size_t i = 0; i < _totalItems; i++) {
      std::cout << data[i] << ", ";
    }

    std::cout << std::endl << "Strides" << std::endl;
    for (size_t i = 0; i < strides.size(); i++) {
      std::cout << strides[i] << " ";
    }

    std::cout << std::endl;
  }
};

#endif
