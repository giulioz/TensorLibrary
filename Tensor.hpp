#ifndef TENSOR_H
#define TENSOR_H

#include <functional>
#include <initializer_list>
#include <iterator>
#include <iostream>
#include <memory>
#include <numeric>
#include <tuple>
#include <vector>

namespace TensorLib {

const static size_t VARIABLE_INDEX = -1;

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

static inline size_t calcFixedStartIndex(std::vector<size_t>& strides,
                                         DimensionsList& indexList,
                                         const size_t& width = 0) {
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

static inline size_t findFixedIndex(DimensionsList& indexList) {
  size_t index = 0;
  for (auto&& fixedValue : indexList) {
    if (fixedValue == VARIABLE_INDEX) {
      return index;
    }
    index++;
  }

  return index;
}

template <typename ValueType>
class Tensor;

// Standard iterator types
template <typename ITValueType>
struct IteratorTypeStandard {
  using ValueType = ITValueType;
  using ReturnType = ITValueType;
  using InternalTensorRef = Tensor<ValueType>&;

  static ValueType& getElementRef(InternalTensorRef& tensor, size_t pos) {
    return tensor.at(pos);
  }
};

// Constant iterator types
template <typename ITValueType>
struct IteratorTypeConst {
  using ValueType = ITValueType;
  using ReturnType = const ITValueType;
  using InternalTensorRef = const Tensor<ValueType>&;

  static const ValueType& getElementRef(InternalTensorRef& tensor, size_t pos) {
    return tensor.c_at(pos);
  }
};

template <typename ValueType>
class Tensor {
 public:
  // Iterator

  template <typename ITType>
  class Iterator : public std::iterator<std::random_access_iterator_tag,
                                        typename ITType::ValueType, size_t> {
    friend class Tensor;

    using iterator_type =
        typename std::iterator<std::random_access_iterator_tag,
                               typename ITType::ValueType, size_t>;

   public:
    using pointer = typename iterator_type::pointer;
    using reference = typename iterator_type::reference;
    using difference_type = typename iterator_type::difference_type;
    using iterator_category = typename std::random_access_iterator_tag;

   private:
    typename ITType::InternalTensorRef tensor;
    size_t currentPos;
    size_t fixedStride;

    Iterator(typename ITType::InternalTensorRef tensor, size_t fixedStride,
             size_t startPos)
        : tensor(tensor), currentPos(startPos), fixedStride(fixedStride) {}

   public:
    Iterator(const Iterator& copy)
        : tensor(copy.tensor),
          currentPos(copy.currentPos),
          fixedStride(copy.fixedStride) {}

    Iterator(const Iterator&& move)
        : tensor(move.tensor),
          currentPos(move.currentPos),
          fixedStride(move.fixedStride) {}

    auto& operator*() { return ITType::getElementRef(tensor, currentPos); }

    auto* operator-> () { return &ITType::getElementRef(tensor, currentPos); }

    auto& operator[](const difference_type& n) {
      return ITType::getElementRef(tensor, n);
    }

    Iterator& operator++() {
      currentPos += fixedStride;
      return *this;
    }
    Iterator& operator--() {
      currentPos -= fixedStride;
      return *this;
    }

    Iterator operator++(int) {
      Iterator result(*this);
      operator++();
      return result;
    }
    Iterator operator--(int) {
      Iterator result(*this);
      operator--();
      return result;
    }

    Iterator operator+(const difference_type& n) const {
      return Iterator(tensor, currentPos + (n * fixedStride));
    }

    Iterator& operator+=(const difference_type& n) {
      currentPos += (n * fixedStride);
      return *this;
    }

    Iterator operator-(const difference_type& n) const {
      return Iterator(tensor, currentPos - (n * fixedStride));
    }

    Iterator& operator-=(const difference_type& n) {
      currentPos -= (n * fixedStride);
      return *this;
    }

    bool operator==(const Iterator& other) const {
      return currentPos == other.currentPos;
    }

    bool operator!=(const Iterator& other) const {
      return currentPos != other.currentPos;
    }

    bool operator<(const Iterator& other) const {
      return currentPos < other.currentPos;
    }

    bool operator>(const Iterator& other) const {
      return currentPos > other.currentPos;
    }

    bool operator<=(const Iterator& other) const {
      return currentPos <= other.currentPos;
    }

    bool operator>=(const Iterator& other) const {
      return currentPos >= other.currentPos;
    }

    difference_type operator+(const Iterator& other) const {
      return (currentPos + other.currentPos) / fixedStride;
    }

    difference_type operator-(const Iterator& other) const {
      return (currentPos - other.currentPos) / fixedStride;
    }
  };

 private:
  using ArrayType = typename std::vector<ValueType>;

  ArrayType data;
  std::vector<size_t> strides;
  std::vector<size_t> sizes;
  size_t _totalItems;

  size_t coordsToIndex(DimensionsList& coords) {
    size_t index = 0;
    size_t strideIndex = 0;
    for (auto&& coord : coords) {
      index += coord * (strides.at(strideIndex));
      strideIndex++;
    }

    return index;
  }

  using standard_iterator = Iterator<IteratorTypeStandard<ValueType>>;
  using const_iterator = Iterator<IteratorTypeConst<ValueType>>;
  using constrained_iterator = Iterator<IteratorTypeStandard<ValueType>>;

 public:
  Tensor(DimensionsList sizes) : data(calcDataSize(sizes)) {
    this->sizes = sizes;
    _totalItems = data.size();
    strides = calcStrides(sizes);

    std::cout << "Strides ";
    for (auto&& stride : strides) {
      std::cout << stride << " ";
    }
    std::cout << std::endl;
  }

  Tensor(const Tensor& other)
      : data(other.data),
        strides(other.strides),
        sizes(other.sizes),
        _totalItems(other._totalItems) {}

  size_t totalItems() const { return _totalItems; }
  size_t itemsAt(size_t dimension) const { return sizes.at(dimension); }

  standard_iterator begin() { return standard_iterator(*this, 1, 0); }
  standard_iterator end() { return standard_iterator(*this, 1, _totalItems); }
  const_iterator cbegin() { return const_iterator(*this, 1, 0); }
  const_iterator cend() { return const_iterator(*this, 1, _totalItems); }

  constrained_iterator constrained_begin(DimensionsList& indexList) {
    std:: cout << "Fixed stride: " << strides[findFixedIndex(indexList)] << std::endl;
    std:: cout << "Start index: " << calcFixedStartIndex(strides, indexList) << std::endl;
    return constrained_iterator(*this,
                                strides[findFixedIndex(indexList)],
                                calcFixedStartIndex(strides, indexList));
  }
  constrained_iterator constrained_end(DimensionsList& indexList) {
    std:: cout << "Fixed stride: " << strides[findFixedIndex(indexList)] << std::endl;
    std:: cout << "Start index: " << calcFixedStartIndex(strides, indexList,sizes[findFixedIndex(indexList)]) << std::endl;
    return constrained_iterator(
        *this, strides[findFixedIndex(indexList)],
        calcFixedStartIndex(strides, indexList,
                            sizes[findFixedIndex(indexList)]));
  }

  ValueType& operator[](const size_t linearCoord) { return at(linearCoord); }
  ValueType& operator[](DimensionsList& coords) { return at(coords); }

  ValueType& at(const size_t linearCoord) { return data.at(linearCoord); }

  ValueType& at(DimensionsList& coords) {
    auto index = coordsToIndex(coords);
    return data.at(index);
  }

  const ValueType& c_at(const size_t linearCoord) const {
    return data.at(linearCoord);
  }

  const ValueType& c_at(DimensionsList& coords) const {
    auto index = coordsToIndex(coords);
    return data.at(index);
  }
};

}  // namespace TensorLib

#endif
