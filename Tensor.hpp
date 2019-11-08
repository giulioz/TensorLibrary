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

static inline size_t findFixedStride(const std::vector<size_t> strides,
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

static inline size_t findInitialPosition(const std::vector<size_t> strides,
                                         FixedDimensionsList indexList,
                                         const size_t width = 0) {
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

template <typename T>
struct is_callable {
 private:
  typedef char (&yes)[1];
  typedef char (&no)[2];

  struct Fallback {
    void operator()();
  };
  struct Derived : T, Fallback {};

  template <typename U, U>
  struct Check;

  template <typename>
  static yes test(...);

  template <typename C>
  static no test(Check<void (Fallback::*)(), &C::operator()>*);

 public:
  static const bool value = sizeof(test<Derived>(0)) == sizeof(yes);
};

template <typename ValueType>
class Tensor;

template <typename ITValueType>
struct IteratorTypeStandard {
  using ValueType = ITValueType;
  using ReturnType = ITValueType;
  using InternalTensorRef = Tensor<ValueType>&;

  static ValueType& getElementRef(InternalTensorRef& tensor, size_t pos) {
    return tensor.at(pos);
  }
};

template <typename ITValueType>
struct IteratorTypeConst {
  using ValueType = ITValueType;
  using ReturnType = const ITValueType;
  using InternalTensorRef = const Tensor<ValueType>&;

  static const ValueType& getElementRef(InternalTensorRef& tensor, size_t pos) {
    return tensor.c_at(pos);
  }
};

template <typename InternalTensorRef>
class IteratorIndexerConstrained {
  InternalTensorRef* tensor;
  size_t pos;

 public:
  IteratorIndexerConstrained(InternalTensorRef& tensor) : tensor(tensor) {}

  size_t operator()() { return tensor->c_at(pos); }
};

template <typename ValueType>
class Tensor {
 public:
  template <typename ITType = IteratorTypeStandard<ValueType>,
            typename ITIndexer =
                IteratorIndexerConstrained<typename ITType::InternalTensorRef>>
  class Iterator : public std::iterator<std::random_access_iterator_tag,
                                        typename ITType::ValueType, ITIndexer> {
    friend class Tensor;

    using iterator_type =
        typename std::iterator<std::random_access_iterator_tag,
                               typename ITType::ValueType, ITIndexer>;

   public:
    using pointer = typename iterator_type::pointer;
    using reference = typename iterator_type::reference;
    using difference_type = typename iterator_type::difference_type;
    using iterator_category = typename std::random_access_iterator_tag;

   private:
    typename ITType::InternalTensorRef tensor;
    ITIndexer currentPos;

    Iterator(typename ITType::InternalTensorRef tensor, ITIndexer startPos = 0)
        : tensor(tensor), currentPos(startPos) {}

   public:
    Iterator(const Iterator& copy)
        : tensor(copy.tensor), currentPos(copy.currentPos) {}

    Iterator(const Iterator&& move)
        : tensor(move.tensor), currentPos(move.currentPos) {}

    template <typename Func, typename... Args>
    typename std::enable_if<std::is_function<Func(Args...)>::value,
                            typename ITType::ReturnType>::type
    operator*() {
      return ITType::getElementRef(tensor, currentPos());
    }
    auto& operator*() { return ITType::getElementRef(tensor, currentPos); }

    template <typename Func, typename... Args>
    typename std::enable_if<std::is_function<Func(Args...)>::value,
                            typename ITType::ReturnType>::type
    operator->() {
      return &ITType::getElementRef(tensor, currentPos());
    }
    auto& operator-> () { return &ITType::getElementRef(tensor, currentPos); }

    template <typename Func, typename... Args>
    typename std::enable_if<std::is_function<Func(Args...)>::value,
                            typename ITType::ReturnType>::type
    operator[](const difference_type& n) {
      return ITType::getElementRef(tensor, n());
    }
    auto& operator[](const difference_type& n) {
      return ITType::getElementRef(tensor, n);
    }

#pragma region Seek Operators

    Iterator& operator++() {
      currentPos++;
      return *this;
    }
    Iterator& operator--() {
      currentPos--;
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
      return Iterator(tensor, currentPos + n);
    }

    Iterator& operator+=(const difference_type& n) {
      currentPos += n;
      return *this;
    }

    Iterator operator-(const difference_type& n) const {
      return Iterator(tensor, currentPos + n);
    }

    Iterator& operator-=(const difference_type& n) {
      currentPos -= n;
      return *this;
    }
#pragma endregion

#pragma region Comparison Operators

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
      return currentPos + other.currentPos;
    }

    difference_type operator-(const Iterator& other) const {
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

  using standard_iterator = Iterator<IteratorTypeStandard<ValueType>, size_t>;
  using const_iterator = Iterator<IteratorTypeConst<ValueType>, size_t>;
  // using constrained_iterator = Iterator<IteratorTypeStandard<ValueType>>;

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

  standard_iterator begin() { return standard_iterator(*this); }
  standard_iterator end() { return standard_iterator(*this, _totalItems); }
  const_iterator cbegin() { return const_iterator(*this); }
  const_iterator cend() { return const_iterator(*this, _totalItems); }

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

};  // namespace TensorLib

#endif
