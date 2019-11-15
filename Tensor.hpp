#ifndef TENSOR_H
#define TENSOR_H

#include <assert.h>
#include <array>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <numeric>
#include <tuple>
#include <vector>

#include <iostream>

namespace TensorLib {

// Placeholder for Constrained Iterator and Dynamic Tensor
inline constexpr size_t VARIABLE_INDEX = -1;
inline constexpr int DYNAMIC_TENSOR_TAG = -1;

// Forward Declarations
// Rank DYNAMIC_TENSOR_TAG = Dynamic
template <typename ValueType, int Rank = DYNAMIC_TENSOR_TAG>
class Tensor;

namespace InternalUtils {

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

inline constexpr size_t calcDataSize() { return 1; }

template <typename... Args>
size_t calcDataSize(const size_t first, const Args... args) {
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
    ++i;
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
    ++i;
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

// Standard iterator types
template <typename ITValueType, int Rank>
struct IteratorTypeStandard {
  using ValueType = ITValueType;
  using ReturnType = ITValueType;
  using InternalTensorRef = Tensor<ValueType, Rank>&;

  static inline ValueType& getElementRef(InternalTensorRef& tensor,
                                         size_t pos) {
    return tensor.at(pos);
  }
};

// Constant iterator types
template <typename ITValueType, int Rank>
struct IteratorTypeConst {
  using ValueType = ITValueType;
  using ReturnType = const ITValueType;
  using InternalTensorRef = const Tensor<ValueType, Rank>&;

  static inline const ValueType& getElementRef(InternalTensorRef& tensor,
                                               size_t pos) {
    return tensor.c_at(pos);
  }
};

}  // namespace InternalUtils

//
// Custom Iterator for Tensor
//
template <typename ITType>
class Iterator {
  template <typename, int>
  friend class Tensor;

 public:
  // Iterator Traits
  using difference_type = size_t;
  using value_type = typename ITType::ValueType;
  using pointer = typename ITType::ValueType*;
  using reference = typename ITType::ValueType&;
  using iterator_category = std::random_access_iterator_tag;

 private:
  // Instance Fields
  typename ITType::InternalTensorRef tensor;
  difference_type currentPos;
  difference_type fixedStride;

  Iterator(typename ITType::InternalTensorRef tensor,
           difference_type startPos = 0, difference_type fixedStride = 1)
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
class Tensor<ValueType, DYNAMIC_TENSOR_TAG> {
  template <typename, int>
  friend class Tensor;

 private:
  using StandardIterator =
      Iterator<typename InternalUtils::IteratorTypeStandard<
          ValueType, DYNAMIC_TENSOR_TAG>>;
  using ConstIterator = Iterator<
      typename InternalUtils::IteratorTypeConst<ValueType, DYNAMIC_TENSOR_TAG>>;
  using ConstrainedIterator =
      Iterator<typename InternalUtils::IteratorTypeStandard<
          ValueType, DYNAMIC_TENSOR_TAG>>;
  using ConstrainedConstIterator =
      Iterator<typename InternalUtils::IteratorTypeStandard<
          ValueType, DYNAMIC_TENSOR_TAG>>;

  using StridesType = std::vector<size_t>;
  using SizesType = std::vector<size_t>;
  using DimensionsType = const std::vector<size_t>&;

  // Internal Data
  std::shared_ptr<std::vector<ValueType>> data;
  StridesType strides;
  SizesType sizes;
  size_t _totalItems;
  size_t offset;

  // Empty Constructor
  Tensor()
      : data(std::make_shared<std::vector<ValueType>>()),
        strides(0),
        sizes(0),
        _totalItems(0),
        offset(0) {
    std::cout << "CALLED: EMPTY CTOR" << std::endl;
  }

 public:
  // Builder static method with varidic sizes
  // Static method because of overload resolution
  template <typename... Sizes>
  static Tensor<ValueType, sizeof...(Sizes)> buildTensor(Sizes... sizes) {
    Tensor<ValueType, sizeof...(Sizes)> tensor;

    tensor.sizes =
        std::array<size_t, sizeof...(Sizes)>{static_cast<size_t>(sizes)...};
    tensor.offset = 0;
    tensor.data = std::make_shared<std::vector<ValueType>>(
        std::vector<ValueType>(InternalUtils::calcDataSize(sizes...)));
    InternalUtils::calcStrides(tensor.sizes, tensor.strides);
    tensor._totalItems = (*(tensor.data)).size();
    std::cout << "CALLED: DYNAMIC SIZES CTOR" << std::endl;

    return tensor;
  }

  //
  // Public Constructors
  //

  Tensor(const std::initializer_list<size_t>& sizes)
      : sizes(sizes.begin(), sizes.end()), offset(0) {
    data = std::make_shared<std::vector<ValueType>>(
        std::vector<ValueType>(InternalUtils::calcDataSize(sizes)));
    InternalUtils::calcStrides(this->sizes, strides);
    _totalItems = (*data).size();
    std::cout << "CALLED: SIZES CTOR" << std::endl;
  }

  // Copy Constructor
  template <int Rank>
  Tensor(const Tensor<ValueType, Rank>& copy)
      : data(std::make_shared<std::vector<ValueType>>((*(copy.data)).cbegin(),
                                                      (*(copy.data)).cend())),
        strides(copy.strides),
        sizes(copy.sizes),
        _totalItems(copy._totalItems),
        offset(copy.offset) {
    std::cout << "CALLED: COPY CTOR" << std::endl;
  }

  // Move Constructor
  template <int Rank>
  Tensor(Tensor<ValueType, Rank>&& move)
      : data(std::move(move.data)),
        strides(move.strides.cbegin(), move.strides.cend()),
        sizes(move.sizes.cbegin(), move.sizes.cend()),
        _totalItems(move._totalItems),
        offset(move.offset) {
    std::cout << "CALLED: MOVE CTOR" << std::endl;
  }

  template <int Rank>
  Tensor<ValueType, Rank>& operator=(const Tensor<ValueType, Rank>& copy) {
    std::cout << "CALLED: COPY EQUAL" << std::endl;
    data = std::make_shared<std::vector<ValueType>>((*(copy.data)).cbegin(),
                                                    (*(copy.data)).cend());

    strides = copy.strides;
    sizes = copy.sizes;
    offset = copy.offset;
    _totalItems = copy._totalItems;
    return *this;
  }

  template <int Rank>
  Tensor<ValueType, Rank>& operator=(Tensor<ValueType, Rank>&& move) {
    std::cout << "CALLED: MOVE EQUAL" << std::endl;
    data = std::move(move.data);
    strides = std::move(move.strides);
    sizes = std::move(move.sizes);
    offset = move.offset;
    _totalItems = move._totalItems;
    return *this;
  }

  //
  // Info Getters
  //

  size_t rank() const { return sizes.size(); }
  size_t size() const { return _totalItems; }
  size_t sizeAt(size_t dimension) const { return sizes.at(dimension); }

  //
  // Iterators Initializers
  //

  StandardIterator begin() { return StandardIterator(*this, 0, strides[0]); }
  StandardIterator end() {
    return StandardIterator(*this, _totalItems * strides[0]);
  }
  ConstIterator cbegin() const { return ConstIterator(*this, 0, strides[0]); }
  ConstIterator cend() const {
    return ConstIterator(*this, _totalItems * strides[0]);
  }

  ConstrainedIterator constrained_begin(DimensionsType& coords) {
    assert(coords.size() == rank());
    return ConstrainedIterator(
        *this, InternalUtils::calcFixedStartIndex(strides, coords),
        strides[InternalUtils::findFixedIndex(coords)]);
  }
  ConstrainedIterator constrained_end(DimensionsType& coords) {
    assert(coords.size() == rank());
    const auto variableIndex = InternalUtils::findFixedIndex(coords);
    return ConstrainedIterator(*this,
                               InternalUtils::calcFixedStartIndex(
                                   strides, coords, sizes[variableIndex]),
                               strides[variableIndex]);
  }

  ConstrainedIterator constrained_begin(DimensionsType& coords,
                                        size_t variableIndex) {
    assert(coords.size() == rank());
    return ConstrainedIterator(
        *this, InternalUtils::calcFixedStartIndex(strides, coords),
        strides[variableIndex]);
  }
  ConstrainedIterator constrained_end(DimensionsType& coords,
                                      size_t variableIndex) {
    assert(coords.size() == rank());
    return ConstrainedIterator(*this,
                               InternalUtils::calcFixedStartIndex(
                                   strides, coords, sizes[variableIndex]),
                               strides[variableIndex]);
  }

  ConstrainedConstIterator constrained_cbegin(DimensionsType& coords) {
    assert(coords.size() == rank());
    return ConstrainedConstIterator(
        *this, InternalUtils::calcFixedStartIndex(strides, coords),
        strides[InternalUtils::findFixedIndex(coords)]);
  }
  ConstrainedConstIterator constrained_cend(DimensionsType& coords) {
    assert(coords.size() == rank());
    const auto variableIndex = InternalUtils::findFixedIndex(coords);
    return ConstrainedConstIterator(*this,
                                    InternalUtils::calcFixedStartIndex(
                                        strides, coords, sizes[variableIndex]),
                                    strides[variableIndex]);
  }

  ConstrainedConstIterator constrained_cbegin(DimensionsType& coords,
                                              size_t variableIndex) {
    assert(coords.size() == rank());
    return ConstrainedConstIterator(
        *this, InternalUtils::calcFixedStartIndex(strides, coords),
        strides[variableIndex]);
  }
  ConstrainedConstIterator constrained_cend(DimensionsType& coords,
                                            size_t variableIndex) {
    assert(coords.size() == rank());
    return ConstrainedConstIterator(*this,
                                    InternalUtils::calcFixedStartIndex(
                                        strides, coords, sizes[variableIndex]),
                                    strides[variableIndex]);
  }

  //
  // Access Operators, both linear and with coordinates
  //

  ValueType& operator[](const size_t linearCoord) {
    return (*data)[linearCoord + offset];
  }
  ValueType& operator[](DimensionsType& coords) {
    assert(coords.size() == rank());
    auto index = InternalUtils::coordsToIndex(coords, strides) + offset;
    return (*data)[index];
  }

  ValueType& at(const size_t linearCoord) {
    return (*data).at(linearCoord + offset);
  }
  ValueType& at(DimensionsType& coords) {
    assert(coords.size() == rank());
    auto index = InternalUtils::coordsToIndex(coords, strides) + offset;
    return (*data).at(index);
  }

  const ValueType& c_at(const size_t linearCoord) const {
    return (*data).at(linearCoord + offset);
  }
  const ValueType& c_at(DimensionsType& coords) const {
    assert(coords.size() == rank());
    auto index = InternalUtils::coordsToIndex(coords, strides) + offset;
    return (*data).at(index);
  }

  //
  // Manipulation functions
  //

  // Clones the tensor without sharing data
  Tensor<ValueType, DYNAMIC_TENSOR_TAG> clone() const {
    std::cout << "CALLED: CLONE" << std::endl;

    Tensor<ValueType, DYNAMIC_TENSOR_TAG> building;
    building.data = std::make_shared<std::vector<ValueType>>((*data).cbegin(),
                                                             (*data).cend());
    building.strides = strides;
    building.sizes = sizes;
    building.offset = offset;
    building._totalItems = _totalItems;
    return building;
  }

  // Clones the tensor sharing data
  Tensor<ValueType, DYNAMIC_TENSOR_TAG> share() const {
    std::cout << "CALLED: SHARE" << std::endl;

    Tensor<ValueType, DYNAMIC_TENSOR_TAG> building;
    building.data = data;
    building.strides = strides;
    building.sizes = sizes;
    building.offset = offset;
    building._totalItems = _totalItems;
    return building;
  }

  // Returns a slice of the tensor
  Tensor<ValueType, DYNAMIC_TENSOR_TAG> slice(
      size_t dimensionIndex, size_t fixedDimensionValue) const {
    assert(dimensionIndex < rank() &&
           fixedDimensionValue <= sizes[dimensionIndex]);

    Tensor<ValueType, DYNAMIC_TENSOR_TAG> sliced;
    sliced.data = data;
    sliced.offset = offset + (fixedDimensionValue * strides[dimensionIndex]);

    sliced.sizes.insert(sliced.sizes.end(), sizes.begin(),
                        sizes.begin() + dimensionIndex);
    sliced.sizes.insert(sliced.sizes.end(), sizes.begin() + dimensionIndex + 1,
                        sizes.end());
    sliced.strides.insert(sliced.strides.end(), strides.begin(),
                          strides.begin() + dimensionIndex);
    sliced.strides.insert(sliced.strides.end(),
                          strides.begin() + dimensionIndex + 1, strides.end());
    sliced._totalItems = InternalUtils::calcDataSize(sliced.sizes);

    return sliced;
  }

  // Merge Strides
  Tensor<ValueType, DYNAMIC_TENSOR_TAG> flatten(size_t start,
                                                size_t end) const {
    assert(start <= strides.size() && end <= strides.size());
    assert(start <= sizes.size() && end <= sizes.size());

    Tensor<ValueType, DYNAMIC_TENSOR_TAG> flatted;
    flatted.data = data;
    flatted.offset = offset;
    flatted._totalItems = _totalItems;

    flatted.strides.insert(flatted.strides.end(), strides.begin(),
                           strides.begin() + start);
    flatted.strides.insert(flatted.strides.end(), strides.begin() + end,
                           strides.end());
    flatted.sizes.insert(flatted.sizes.end(), sizes.begin(),
                         sizes.begin() + start);
    flatted.sizes.insert(flatted.sizes.end(), sizes.begin() + end, sizes.end());

    for (size_t i = start; i != end; ++i) {
      flatted.sizes.at(start) *= sizes[i];
    }

    return flatted;
  }
};

template <typename ValueType, int Rank>
class Tensor {
  template <typename, int>
  friend class Tensor;

 private:
  using StandardIterator =
      Iterator<typename InternalUtils::IteratorTypeStandard<ValueType, Rank>>;
  using ConstIterator =
      Iterator<typename InternalUtils::IteratorTypeConst<ValueType, Rank>>;
  using ConstrainedIterator =
      Iterator<typename InternalUtils::IteratorTypeStandard<ValueType, Rank>>;
  using ConstrainedConstIterator =
      Iterator<typename InternalUtils::IteratorTypeStandard<ValueType, Rank>>;

  using StridesType = std::array<size_t, Rank>;
  using SizesType = std::array<size_t, Rank>;
  using DimensionsType = const std::array<size_t, Rank>&;
  using InitializerDimensionsType = const std::array<size_t, Rank>&;

  // Internal Data
  std::shared_ptr<std::vector<ValueType>> data;
  StridesType strides;
  SizesType sizes;
  size_t _totalItems;
  size_t offset;

  // Empty Constructor
  Tensor()
      : data(std::make_shared<std::vector<ValueType>>()),
        _totalItems(0),
        offset(0) {
    std::cout << "CALLED: EMPTY CTOR" << std::endl;
  }

 public:
  // Builder static method with varidic sizes
  // Static method because of overload resolution
  template <typename... Sizes>
  static Tensor<ValueType, sizeof...(Sizes)> buildTensor(Sizes... sizes) {
    static_assert(sizeof...(Sizes) == Rank);

    Tensor<ValueType, sizeof...(Sizes)> tensor;

    tensor.sizes =
        std::array<size_t, sizeof...(Sizes)>{static_cast<size_t>(sizes)...};
    tensor.offset = 0;
    tensor.data = std::make_shared<std::vector<ValueType>>(
        std::vector<ValueType>(InternalUtils::calcDataSize(sizes...)));
    InternalUtils::calcStrides(tensor.sizes, tensor.strides);
    tensor._totalItems = (*(tensor.data)).size();
    std::cout << "CALLED: DYNAMIC SIZES CTOR" << std::endl;

    return tensor;
  }

  //
  // Public Constructors
  //

  Tensor(const std::initializer_list<size_t>& sizes) : offset(0) {
    assert(sizes.size() == Rank);

    std::copy(sizes.begin(), sizes.end(), this->sizes.begin());
    data = std::make_shared<std::vector<ValueType>>(
        std::vector<ValueType>(InternalUtils::calcDataSize(sizes)));
    InternalUtils::calcStrides(this->sizes, strides);
    _totalItems = (*data).size();
    std::cout << "CALLED: SIZES CTOR DYN" << std::endl;
  }

  // Copy Constructor
  Tensor(const Tensor<ValueType, DYNAMIC_TENSOR_TAG>& copy)
      : data(std::make_shared<std::vector<ValueType>>((*(copy.data)).cbegin(),
                                                      (*(copy.data)).cend())),
        strides(copy.strides),
        sizes(copy.sizes),
        _totalItems(copy._totalItems),
        offset(copy.offset) {
    assert(copy.rank() == Rank);
    std::cout << "CALLED: COPY CTOR" << std::endl;
  }

  // Copy Constructor
  Tensor(const Tensor<ValueType, Rank>& copy)
      : data(std::make_shared<std::vector<ValueType>>((*(copy.data)).cbegin(),
                                                      (*(copy.data)).cend())),
        strides(copy.strides),
        sizes(copy.sizes),
        _totalItems(copy._totalItems),
        offset(copy.offset) {
    std::cout << "CALLED: COPY CTOR" << std::endl;
  }

  // Move Constructor
  Tensor(Tensor<ValueType, Rank>&& move)
      : data(std::move(move.data)),
        strides(move.strides),
        sizes(move.sizes),
        _totalItems(move._totalItems),
        offset(move.offset) {
    std::cout << "CALLED: MOVE CTOR" << std::endl;
  }

  // Move Constructor
  Tensor(Tensor<ValueType, DYNAMIC_TENSOR_TAG>&& move)
      : data(std::move(move.data)),
        strides(move.strides),
        sizes(move.sizes),
        _totalItems(move._totalItems),
        offset(move.offset) {
    assert(move.rank() == Rank);
    std::cout << "CALLED: MOVE CTOR" << std::endl;
  }

  Tensor<ValueType, Rank>& operator=(const Tensor<ValueType, Rank>& copy) {
    data = std::make_shared<std::vector<ValueType>>((*data).cbegin(),
                                                    (*data).cend());
    strides = copy.strides;
    sizes = copy.sizes;
    offset = copy.offset;
    _totalItems = copy._totalItems;
    std::cout << "CALLED: COPY EQUAL" << std::endl;
    return *this;
  }

  Tensor<ValueType, Rank>& operator=(Tensor<ValueType, Rank>&& move) {
    data = std::move(move.data);
    strides = move.strides;
    sizes = move.sizes;
    offset = move.offset;
    _totalItems = move._totalItems;
    std::cout << "CALLED: MOVE EQUAL" << std::endl;
    return *this;
  }

  //
  // Info Getters
  //

  constexpr size_t rank() const { return Rank; }
  size_t size() const { return _totalItems; }
  size_t sizeAt(size_t dimension) const { return sizes.at(dimension); }

  //
  // Iterators Initializers
  //

  StandardIterator begin() { return StandardIterator(*this, 0, strides[0]); }
  StandardIterator end() {
    return StandardIterator(*this, _totalItems * strides[0]);
  }
  ConstIterator cbegin() const { return ConstIterator(*this, 0, strides[0]); }
  ConstIterator cend() const {
    return ConstIterator(*this, _totalItems * strides[0]);
  }

  ConstrainedIterator constrained_begin(DimensionsType& coords) {
    static_assert(coords.size() == Rank);
    return ConstrainedIterator(
        *this, InternalUtils::calcFixedStartIndex(strides, coords),
        strides[InternalUtils::findFixedIndex(coords)]);
  }
  ConstrainedIterator constrained_end(DimensionsType& coords) {
    static_assert(coords.size() == Rank);
    const auto variableIndex = InternalUtils::findFixedIndex(coords);
    return ConstrainedIterator(*this,
                               InternalUtils::calcFixedStartIndex(
                                   strides, coords, sizes[variableIndex]),
                               strides[variableIndex]);
  }

  ConstrainedIterator constrained_begin(DimensionsType& coords,
                                        size_t variableIndex) {
    static_assert(coords.size() == Rank);
    return ConstrainedIterator(
        *this, InternalUtils::calcFixedStartIndex(strides, coords),
        strides[variableIndex]);
  }
  ConstrainedIterator constrained_end(DimensionsType& coords,
                                      size_t variableIndex) {
    static_assert(coords.size() == Rank);
    return ConstrainedIterator(*this,
                               InternalUtils::calcFixedStartIndex(
                                   strides, coords, sizes[variableIndex]),
                               strides[variableIndex]);
  }

  ConstrainedConstIterator constrained_cbegin(DimensionsType& coords) {
    static_assert(coords.size() == Rank);
    return ConstrainedConstIterator(
        *this, InternalUtils::calcFixedStartIndex(strides, coords),
        strides[InternalUtils::findFixedIndex(coords)]);
  }
  ConstrainedConstIterator constrained_cend(DimensionsType& coords) {
    static_assert(coords.size() == Rank);
    const auto variableIndex = InternalUtils::findFixedIndex(coords);
    return ConstrainedConstIterator(*this,
                                    InternalUtils::calcFixedStartIndex(
                                        strides, coords, sizes[variableIndex]),
                                    strides[variableIndex]);
  }

  ConstrainedConstIterator constrained_cbegin(DimensionsType& coords,
                                              size_t variableIndex) {
    static_assert(coords.size() == Rank);
    return ConstrainedConstIterator(
        *this, InternalUtils::calcFixedStartIndex(strides, coords),
        strides[variableIndex]);
  }
  ConstrainedConstIterator constrained_cend(DimensionsType& coords,
                                            size_t variableIndex) {
    static_assert(coords.size() == Rank);
    return ConstrainedConstIterator(*this,
                                    InternalUtils::calcFixedStartIndex(
                                        strides, coords, sizes[variableIndex]),
                                    strides[variableIndex]);
  }

  //
  // Access Operators, both linear and with coordinates
  //

  ValueType& operator[](const size_t linearCoord) {
    return (*data)[linearCoord + offset];
  }
  ValueType& operator[](DimensionsType& coords) {
    static_assert(coords.size() == Rank);
    auto index = InternalUtils::coordsToIndex(coords, strides) + offset;
    return (*data)[index];
  }

  ValueType& at(const size_t linearCoord) {
    return (*data).at(linearCoord + offset);
  }
  ValueType& at(DimensionsType& coords) {
    static_assert(coords.size() == Rank);
    auto index = InternalUtils::coordsToIndex(coords, strides) + offset;
    return (*data).at(index);
  }

  const ValueType& c_at(const size_t linearCoord) const {
    return (*data).at(linearCoord + offset);
  }
  const ValueType& c_at(DimensionsType& coords) const {
    static_assert(coords.size() == Rank);
    auto index = InternalUtils::coordsToIndex(coords, strides) + offset;
    return (*data).at(index);
  }

  //
  // Manipulation functions
  //

  // Clones the tensor without sharing data
  Tensor<ValueType, Rank> clone() const {
    std::cout << "CALLED: CLONE" << std::endl;

    Tensor<ValueType, Rank> building;
    building.data = std::make_shared<std::vector<ValueType>>((*data).cbegin(),
                                                             (*data).cend());
    building.strides = strides;
    building.sizes = sizes;
    building.offset = offset;
    building._totalItems = _totalItems;
    return building;
  }

  // Clones the tensor sharing data
  Tensor<ValueType, Rank> share() const {
    std::cout << "CALLED: SHARE" << std::endl;

    Tensor<ValueType, Rank> building;
    building.data = data;
    building.strides = strides;
    building.sizes = sizes;
    building.offset = offset;
    building._totalItems = _totalItems;
    return building;
  }

  // Returns a slice of the tensor
  Tensor<ValueType, Rank - 1> slice(size_t dimensionIndex,
                                    size_t fixedDimensionValue) const {
    assert(dimensionIndex < rank() &&
           fixedDimensionValue <= sizes[dimensionIndex]);

    Tensor<ValueType, Rank - 1> sliced;
    sliced.data = data;
    sliced.offset = offset + (fixedDimensionValue * strides[dimensionIndex]);

    for (size_t index = 0; index < dimensionIndex; index++) {
      sliced.strides[index] = strides[index];
      sliced.sizes[index] = sizes[index];
    }
    for (size_t index = dimensionIndex + 1; index < strides.size(); index++) {
      sliced.strides[index - 1] = strides[index];
      sliced.sizes[index - 1] = sizes[index];
    }
    sliced._totalItems = InternalUtils::calcDataSize(sliced.sizes);

    return sliced;
  }

  // Merge Strides
  // We can't return a fixed rank vector, since we can't know the resulting rank
  Tensor<ValueType, DYNAMIC_TENSOR_TAG> flatten(size_t start,
                                                size_t end) const {
    assert(start <= strides.size() && end <= strides.size());
    assert(start <= sizes.size() && end <= sizes.size());

    Tensor<ValueType, DYNAMIC_TENSOR_TAG> flatted;
    flatted.data = data;
    flatted.offset = offset;
    flatted._totalItems = _totalItems;

    flatted.strides.insert(flatted.strides.end(), strides.begin(),
                           strides.begin() + start);
    flatted.strides.insert(flatted.strides.end(), strides.begin() + end,
                           strides.end());
    flatted.sizes.insert(flatted.sizes.end(), sizes.begin(),
                         sizes.begin() + start);
    flatted.sizes.insert(flatted.sizes.end(), sizes.begin() + end, sizes.end());

    for (size_t i = start; i != end; ++i) {
      flatted.sizes.at(start) *= sizes[i];
    }

    return flatted;
  }
};

}  // namespace TensorLib

#endif
