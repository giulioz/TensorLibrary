#ifndef TENSOR_H
#define TENSOR_H

#include <array>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <numeric>
#include <tuple>
#include <vector>
#include <assert.h>

namespace TensorLib {

// Placeholder for Constrained Iterator
extern const size_t VARIABLE_INDEX;

// Utility functions
template <typename StridesType, typename CoordsType>
size_t coordsToIndex(const CoordsType& coords, const StridesType& strides) {
  size_t index = 0;
  size_t strideIndex = 0;
  for (auto&& coord : coords) {
    index += coord * (strides.at(strideIndex));
    strideIndex++;
  }

  return index;
}

size_t calcDataSize();

template <typename... Args>
size_t calcDataSize(size_t first, Args... args) {
  return first * calcDataSize(args...);
}

template <typename CoordsType>
size_t calcDataSize(const CoordsType& dimensions) {
  return std::accumulate(std::begin(dimensions), std::end(dimensions), 1,
                         std::multiplies<double>());
}

template <typename CoordsType>
void calcStrides(const CoordsType& dimensions, std::vector<size_t>& strides) {
  size_t stride = 1;
  int i = 0;
  for (auto&& dim : dimensions) {
    strides.push_back(stride);
    stride *= dim;
    i++;
  }
}

template <size_t Rank, typename CoordsType>
void calcStrides(const CoordsType& dimensions,
                 std::array<size_t, Rank>& strides) {
  size_t stride = 1;
  int i = 0;
  for (auto&& dim : dimensions) {
    strides[i] = stride;
    stride *= dim;
    i++;
  }
}

template <typename StridesType, typename CoordsType>
size_t calcFixedStartIndex(const StridesType& strides,
                           const CoordsType& indexList,
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

template <typename CoordsType>
size_t findFixedIndex(const CoordsType& indexList) {
  size_t index = 0;
  for (auto&& fixedValue : indexList) {
    if (fixedValue == VARIABLE_INDEX) {
      return index;
    }
    index++;
  }

  return index;
}

struct TensorTypeDynamicRank {
  using StridesType = std::vector<size_t>;
  using SizesType = std::vector<size_t>;
  using DimensionsType = const std::vector<size_t>&;
  using InitializerDimensionsType = const std::initializer_list<size_t>&;
};

template <size_t Rank>
struct TensorTypeFixedRank {
  using StridesType = std::array<size_t, Rank>;
  using SizesType = std::array<size_t, Rank>;
  using DimensionsType = const std::array<size_t, Rank>&;
  using InitializerDimensionsType = const std::array<size_t, Rank>&;
};

// Forward Declarations
template <typename ValueType, typename TensorType = TensorTypeDynamicRank>
class Tensor;

// Standard iterator types
template <typename ITValueType, typename TensorType>
struct IteratorTypeStandard {
  using ValueType = ITValueType;
  using ReturnType = ITValueType;
  using InternalTensorRef = Tensor<ValueType, TensorType>&;

  static ValueType& getElementRef(InternalTensorRef& tensor, size_t pos) {
    return tensor.at(pos);
  }
};

// Constant iterator types
template <typename ITValueType, typename TensorType>
struct IteratorTypeConst {
  using ValueType = ITValueType;
  using ReturnType = const ITValueType;
  using InternalTensorRef = const Tensor<ValueType, TensorType>&;

  static const ValueType& getElementRef(InternalTensorRef& tensor, size_t pos) {
    return tensor.c_at(pos);
  }
};

//
// Custom Iterator for Tensor
//
template <typename ITType>
class Iterator : public std::iterator<std::random_access_iterator_tag,
                                      typename ITType::ValueType, size_t> {
  template <typename, typename>
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

  Iterator(typename ITType::InternalTensorRef tensor, size_t startPos = 0,
           size_t fixedStride = 1)
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
    return Iterator(tensor, currentPos + (n * fixedStride), fixedStride);
  }

  Iterator& operator+=(const difference_type& n) {
    currentPos += (n * fixedStride);
    return *this;
  }

  Iterator operator-(const difference_type& n) const {
    return Iterator(tensor, currentPos - (n * fixedStride), fixedStride);
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

template <typename ValueType, typename TensorType>
class Tensor {
 private:
  using StandardIterator =
      Iterator<IteratorTypeStandard<ValueType, TensorType>>;
  using ConstIterator = Iterator<IteratorTypeConst<ValueType, TensorType>>;
  using ConstrainedIterator =
      Iterator<IteratorTypeStandard<ValueType, TensorType>>;
  using ConstrainedConstIterator =
      Iterator<IteratorTypeStandard<ValueType, TensorType>>;

  using DimsType = typename TensorType::DimensionsType;
  using DimsInitType = typename TensorType::InitializerDimensionsType;

  // Internal Data
  std::vector<ValueType> data;
  typename TensorType::StridesType strides;
  typename TensorType::SizesType sizes;
  size_t _totalItems;

 public:
  // Public Constructors

  Tensor(typename TensorType::InitializerDimensionsType sizes)
      : data(calcDataSize(sizes)), sizes(sizes) {
    calcStrides(this->sizes, strides);
    _totalItems = data.size();
  }

  template <typename... Sizes>
  Tensor(Sizes... sizes)
      : data(calcDataSize(sizes...)),
        sizes(DimsInitType{static_cast<size_t>(sizes)...}) {
    calcStrides(this->sizes, strides);
    _totalItems = data.size();
  }

  // Copy Constructor
  Tensor(const Tensor& copy)
      : data(copy.data),
        strides(copy.strides),
        sizes(copy.sizes),
        _totalItems(copy._totalItems) {}

  // Move Constructor
  Tensor(const Tensor&& move)
      : data(move.data),
        strides(move.strides),
        sizes(move.sizes),
        _totalItems(move._totalItems) {}

  //
  // Info Getters
  //

  size_t rank() const { return sizes.size(); }
  size_t totalItemsCount() const { return _totalItems; }
  size_t itemsCountAt(size_t dimension) const { return sizes.at(dimension); }

  //
  // Iterators Initializers
  //

  StandardIterator begin() { return StandardIterator(*this); }
  StandardIterator end() { return StandardIterator(*this, _totalItems); }
  ConstIterator cbegin() { return ConstIterator(*this); }
  ConstIterator cend() { return ConstIterator(*this, _totalItems); }

  ConstrainedIterator constrained_begin(
      typename TensorType::DimensionsType& indexList) {
    return ConstrainedIterator(*this, calcFixedStartIndex(strides, indexList),
                               strides[findFixedIndex(indexList)]);
  }
  ConstrainedIterator constrained_end(
      typename TensorType::DimensionsType& indexList) {
    const auto variableIndex = findFixedIndex(indexList);
    return ConstrainedIterator(
        *this, calcFixedStartIndex(strides, indexList, sizes[variableIndex]),
        strides[variableIndex]);
  }

  ConstrainedIterator constrained_begin(
      typename TensorType::DimensionsType& indexList, size_t variableIndex) {
    return ConstrainedIterator(*this, calcFixedStartIndex(strides, indexList),
                               strides[variableIndex]);
  }
  ConstrainedIterator constrained_end(
      typename TensorType::DimensionsType& indexList, size_t variableIndex) {
    return ConstrainedIterator(
        *this, calcFixedStartIndex(strides, indexList, sizes[variableIndex]),
        strides[variableIndex]);
  }

  ConstrainedConstIterator constrained_cbegin(
      typename TensorType::DimensionsType& indexList) {
    return ConstrainedConstIterator(*this,
                                    calcFixedStartIndex(strides, indexList),
                                    strides[findFixedIndex(indexList)]);
  }
  ConstrainedConstIterator constrained_cend(
      typename TensorType::DimensionsType& indexList) {
    const auto variableIndex = findFixedIndex(indexList);
    return ConstrainedConstIterator(
        *this, calcFixedStartIndex(strides, indexList, sizes[variableIndex]),
        strides[variableIndex]);
  }

  ConstrainedConstIterator constrained_cbegin(
      typename TensorType::DimensionsType& indexList, size_t variableIndex) {
    return ConstrainedConstIterator(
        *this, calcFixedStartIndex(strides, indexList), strides[variableIndex]);
  }
  ConstrainedConstIterator constrained_cend(
      typename TensorType::DimensionsType& indexList, size_t variableIndex) {
    return ConstrainedConstIterator(
        *this, calcFixedStartIndex(strides, indexList, sizes[variableIndex]),
        strides[variableIndex]);
  }

  //
  // Access Operators, both linear and coordinates
  //

  ValueType& operator[](const size_t linearCoord) { return data[linearCoord]; }
  ValueType& operator[](typename TensorType::DimensionsType& coords) {
    assert(coords.size() == rank());
    auto index = coordsToIndex(coords, strides);
    return data[index];
  }

  ValueType& at(const size_t linearCoord) { return data.at(linearCoord); }
  ValueType& at(typename TensorType::DimensionsType& coords) {
    assert(coords.size() == rank());
    auto index = coordsToIndex(coords, strides);
    return data.at(index);
  }

  const ValueType& c_at(const size_t linearCoord) const {
    return data.at(linearCoord);
  }
  const ValueType& c_at(typename TensorType::DimensionsType& coords) const {
    assert(coords.size() == rank());
    auto index = coordsToIndex(coords, strides);
    return data.at(index);
  }
};

}  // namespace TensorLib

#endif
