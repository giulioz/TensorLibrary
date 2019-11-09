#ifndef TENSOR_H
#define TENSOR_H

#include <functional>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <numeric>
#include <tuple>
#include <vector>

namespace TensorLib {

// Placeholder for Constrained Iterator
extern const size_t VARIABLE_INDEX;

using DimensionsList = const std::initializer_list<size_t>&;

// Utility functions
size_t coordsToIndex(DimensionsList& coords,
                     const std::vector<size_t>& strides);
size_t calcDataSize(DimensionsList dimensions);
std::vector<size_t> calcStrides(DimensionsList dimensions);
size_t calcFixedStartIndex(const std::vector<size_t>& strides,
                           DimensionsList& indexList, const size_t& width = 0);
size_t findFixedIndex(DimensionsList& indexList);

// Forward Declarations
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

//
// Custom Iterator for Tensor
//
template <typename ITType>
class Iterator : public std::iterator<std::random_access_iterator_tag,
                                      typename ITType::ValueType, size_t> {
  friend class Tensor<typename ITType::ValueType>;

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

template <typename ValueType>
class Tensor {
 private:
  using ArrayType = typename std::vector<ValueType>;
  using StandardIterator = Iterator<IteratorTypeStandard<ValueType>>;
  using ConstIterator = Iterator<IteratorTypeConst<ValueType>>;
  using ConstrainedIterator = Iterator<IteratorTypeStandard<ValueType>>;

  // Internal Data
  ArrayType data;
  std::vector<size_t> strides;
  std::vector<size_t> sizes;
  size_t _totalItems;

 public:
  // Public constructors
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

  Tensor(const Tensor&& other)
      : data(other.data),
        strides(other.strides),
        sizes(other.sizes),
        _totalItems(other._totalItems) {}

  //
  // Info getters
  //

  size_t totalItemsCount() const { return _totalItems; }
  size_t itemsCountAt(size_t dimension) const { return sizes.at(dimension); }

  //
  // Iterators initializers
  //

  StandardIterator begin() { return StandardIterator(*this); }
  StandardIterator end() { return StandardIterator(*this, _totalItems); }
  ConstIterator cbegin() { return ConstIterator(*this); }
  ConstIterator cend() { return ConstIterator(*this, _totalItems); }

  ConstrainedIterator constrained_begin(DimensionsList& indexList) {
    return ConstrainedIterator(*this, calcFixedStartIndex(strides, indexList),
                               strides[findFixedIndex(indexList)]);
  }
  ConstrainedIterator constrained_end(DimensionsList& indexList) {
    const auto variableIndex = findFixedIndex(indexList);
    return ConstrainedIterator(
        *this, calcFixedStartIndex(strides, indexList, sizes[variableIndex]),
        strides[variableIndex]);
  }

  ConstrainedIterator constrained_begin(DimensionsList& indexList,
                                        size_t variableIndex) {
    return ConstrainedIterator(*this, calcFixedStartIndex(strides, indexList),
                               strides[variableIndex]);
  }
  ConstrainedIterator constrained_end(DimensionsList& indexList,
                                      size_t variableIndex) {
    return ConstrainedIterator(
        *this, calcFixedStartIndex(strides, indexList, sizes[variableIndex]),
        strides[variableIndex]);
  }

  //
  // Access Operators
  //

  ValueType& operator[](const size_t linearCoord) { return at(linearCoord); }
  ValueType& operator[](DimensionsList& coords) { return at(coords); }

  ValueType& at(const size_t linearCoord) { return data.at(linearCoord); }

  ValueType& at(DimensionsList& coords) {
    auto index = coordsToIndex(coords, strides);
    return data.at(index);
  }

  const ValueType& c_at(const size_t linearCoord) const {
    return data.at(linearCoord);
  }

  const ValueType& c_at(DimensionsList& coords) const {
    auto index = coordsToIndex(coords, strides);
    return data.at(index);
  }
};

}  // namespace TensorLib

#endif
