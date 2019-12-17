#ifndef TENSOR
#define TENSOR

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <initializer_list>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using namespace std::rel_ops;

namespace tensor {

// policy for dynamically ranked tensors
struct dynamic {
  typedef std::vector<size_t> index_type;
  typedef std::vector<size_t> width_type;
};

// policy for fixed-rank tensors
template <size_t R> struct rank {
  typedef std::array<size_t, R> index_type;
  typedef std::array<size_t, R> width_type;
};

// tensor type
template <typename T, class type = dynamic> class tensor;

// ====================================================================

template <char...> struct vars;

template <typename, typename = void> struct free_vars;

template <typename, typename = void> struct repeated_vars;

template <typename, typename> struct concat_vars;

template <char, typename> struct find_var;

template <char, typename> struct remove_var;

template <char... Is> struct vars {
  constexpr static size_t size = sizeof...(Is);
  constexpr static char id[sizeof...(Is)] = {Is...};
};

template <char... Is> constexpr char vars<Is...>::id[sizeof...(Is)];

template <> struct free_vars<vars<>> { using value = vars<>; };

template <char H, char... Ts>
struct free_vars<vars<H, Ts...>, typename std::enable_if<
                                     find_var<H, vars<Ts...>>::value>::type> {
  using value =
      typename free_vars<typename remove_var<H, vars<Ts...>>::value>::value;
};

template <char H, char... Ts>
struct free_vars<vars<H, Ts...>, typename std::enable_if<
                                     !find_var<H, vars<Ts...>>::value>::type> {
  using value =
      typename concat_vars<vars<H>,
                           typename free_vars<vars<Ts...>>::value>::value;
};

template <> struct repeated_vars<vars<>> { using value = vars<>; };

template <char H, char... Ts>
struct repeated_vars<
    vars<H, Ts...>,
    typename std::enable_if<find_var<H, vars<Ts...>>::value>::type> {
  using value =
      typename concat_vars<vars<H>, typename repeated_vars<typename remove_var<
                                        H, vars<Ts...>>::value>::value>::value;
};

template <char H, char... Ts>
struct repeated_vars<
    vars<H, Ts...>,
    typename std::enable_if<!find_var<H, vars<Ts...>>::value>::type> {
  using value = typename repeated_vars<vars<Ts...>>::value;
};

template <char... As, char... Bs> struct concat_vars<vars<As...>, vars<Bs...>> {
  using value = vars<As..., Bs...>;
};

template <char I> struct find_var<I, vars<>> {
  constexpr static bool value = false;
};

template <char I, char... Ts> struct find_var<I, vars<I, Ts...>> {
  constexpr static bool value = true;
};

template <char I, char H, char... Ts> struct find_var<I, vars<H, Ts...>> {
  constexpr static bool value = find_var<I, vars<Ts...>>::value;
};

template <char I> struct remove_var<I, vars<>> { using value = vars<>; };

template <char I, char... Ts> struct remove_var<I, vars<I, Ts...>> {
  using value = typename remove_var<I, vars<Ts...>>::value;
};

template <char I, char H, char... Ts> struct remove_var<I, vars<H, Ts...>> {
  using value =
      typename concat_vars<vars<H>,
                           typename remove_var<I, vars<Ts...>>::value>::value;
};

template <typename, typename> class tensor_expression;

template <typename, typename> class tensor_constant;

template <typename, typename, typename> class tensor_addition;

// template <typename, typename> class tensor_negation; // TODO

template <typename, typename, typename> class tensor_multiplication;

template <typename> struct expression_multi_vars;

template <typename T, char... Is>
struct expression_multi_vars<tensor_constant<T, vars<Is...>>> {
  using value = vars<Is...>;
};

template <typename T, typename A, typename B>
struct expression_multi_vars<tensor_addition<T, A, B>> {
  using value =
      typename concat_vars<typename expression_multi_vars<A>::value,
                           typename expression_multi_vars<B>::value>::value;
};

/*
template <typename T, typename A>
struct expression_multi_vars<tensor_negation<T, A>> {
    using value = typename expression_multi_vars<A>::value;
}; */

template <typename T, typename A, typename B>
struct expression_multi_vars<tensor_multiplication<T, A, B>> {
  using value =
      typename concat_vars<typename expression_multi_vars<A>::value,
                           typename expression_multi_vars<B>::value>::value;
};

template <typename T, typename Derived> class tensor_expression {
public:
  using free_vars =
      typename free_vars<typename expression_multi_vars<Derived>::value>::value;
  using repeated_vars = typename repeated_vars<
      typename expression_multi_vars<Derived>::value>::value;
  constexpr static size_t result_rank = free_vars::size;

  template <typename Derived1>
  tensor_addition<T, Derived, Derived1>
  operator+(const tensor_expression<T, Derived1> &other) {
    return tensor_addition<T, Derived, Derived1>(*this, other);
  }

  // operator-(); // TODO

  // operator-(other); // TODO

  template <typename Derived1>
  tensor_multiplication<T, Derived, Derived1>
  operator+(const tensor_expression<T, Derived1> &other) {
    return tensor_multiplication<T, Derived, Derived1>(*this, other);
  }

  tensor<T, rank<result_rank>> evaluate() {
    std::vector<size_t> free_vars_dims;
    size_t free_vars_size = 1;
    for (size_t i = 0; i < free_vars::size; i++) {
      size_t d = get_dimension(free_vars::id[i]);
      free_vars_dims.push_back(d);
      free_vars_size *= d;
    }

    std::vector<size_t> repeated_vars_dims;
    size_t repeated_vars_size = 1;
    for (size_t i = 0; i < repeated_vars::size; i++) {
      size_t d = get_dimension(repeated_vars::id[i]);
      repeated_vars_dims.push_back(d);
      repeated_vars_size *= d;
    }

    tensor<T, rank<result_rank>> result(free_vars_dims);

    for (size_t i = 0; i < free_vars_size; i++) {
      auto free_vars_values = calc_vars_values_by_i(i, free_vars_dims);
      T part = evaluate_part(combine_vars_values(
          free_vars_values, calc_vars_values_by_i(0, repeated_vars_dims)));
      for (size_t j = 1; j < repeated_vars_size; j++) {
        part += evaluate_part(combine_vars_values(
            free_vars_values, calc_vars_values_by_i(j, repeated_vars_dims)));
      }
      result(free_vars_values) = part;
    }

    return result;
  }

protected:
  virtual T evaluate_part(const std::map<char, size_t> &vars_values) const;

  virtual size_t get_dimension(char v) const;

private:
  std::vector<size_t> calc_vars_values_by_i(size_t i,
                                            const std::vector<size_t> &dims) {
    std::vector<size_t> result;

    for (size_t j = dims.size() - 1; j >= 0; j--) {
      result.insert(result.begin(), i % dims[j]);
      i /= dims[j];
    }

    return result;
  }

  std::map<char, size_t>
  combine_vars_values(const std::vector<size_t> &free_vars_values,
                      const std::vector<size_t> &repeated_vars_values) {
    std::map<char, size_t> result;

    for (size_t i = 0; i < free_vars::size; i++) {
      result[free_vars::id[i]] = free_vars_values[i];
    }

    for (size_t i = 0; i < repeated_vars::size; i++) {
      result[repeated_vars::id[i]] = repeated_vars_values[i];
    }

    return result;
  }
};

template <typename T, char... Is>
class tensor_constant<T, vars<Is...>>
    : public tensor_expression<T, tensor_constant<T, vars<Is...>>> {
public:
  tensor_constant(const tensor<T> &tensorRef) : tensorRef(tensorRef) {}

  template <size_t N>
  tensor_constant(
      const tensor<T, rank<N>> &tensorRef,
      typename std::enable_if<N == sizeof...(Is), bool>::type = true)
      : tensorRef(tensorRef) {}

protected:
  T evaluate_part(const std::map<char, size_t> &vars_values) const override {
    std::vector<size_t> indexes;
    for (size_t i = 0; i < vars<Is...>::size; i++) {
      indexes.push_back(vars_values.at(vars<Is...>::id[i]));
    }
    return tensor(indexes);
  }

  size_t get_dimension(char v) const override {
    for (size_t i = 0; i < vars<Is...>::size; i++) {
      if (v == vars<Is...>::id[i]) {
        return tensorRef.width[i];
      }
    }
    return -1;
  }

  friend class tensor<T>;

private:
  const tensor<T> tensorRef;
};

template <typename T, typename A, typename B>
class tensor_addition : public tensor_expression<T, tensor_addition<T, A, B>> {
public:
  tensor_addition(const A &a, const B &b) : a(a), b(b) {}

protected:
  T evaluate_part(const std::map<char, size_t> &vars_values) const override {
    return a.evaluate_part(vars_values) + b.evaluate_part(vars_values);
  }

  size_t get_dimension(char v) const override {
    size_t result = a.get_dimension(v);
    if (result < 0) {
      result = b.get_dimension(v);
    }
    return result;
  }

private:
  const A a;
  const B b;
};

// template <typename, typename> class tensor_negation; // TODO

template <typename T, typename A, typename B>
class tensor_multiplication
    : public tensor_expression<T, tensor_multiplication<T, A, B>> {
public:
  tensor_multiplication(const A &a, const B &b) : a(a), b(b) {}

protected:
  T evaluate_part(const std::map<char, size_t> &vars_values) const override {
    return a.evaluate_part(vars_values) * b.evaluate_part(vars_values);
  }

  size_t get_dimension(char v) const override {
    size_t result = a.get_dimension(v);
    if (result < 0) {
      result = b.get_dimension(v);
    }
    return result;
  }

private:
  const A a;
  const B b;
};

// ====================================================================

namespace reserved {
// generic iterator used by all tensor classes (except rank 1 specializations)
template <typename T, class type> class iterator {
public:
  T &operator*() const { return *ptr; }

  iterator &operator++() {
    // I am using a right-major layout
    // start increasing the last index
    size_t index = stride.size() - 1;
    ++idx[index];
    ptr += stride[index];
    // as long as the current index has reached maximum width,
    // set it to 0 and increase the next index
    while (idx[index] == width[index] && index > 0) {
      idx[index] = 0;
      ptr -= width[index] * stride[index];
      --index;
      ++idx[index];
      ptr += stride[index];
    }
    return *this;
  }

  iterator operator++(int) {
    iterator result(*this);
    operator++();
    return result;
  }
  iterator &operator--() {
    // I am using a right-major layout
    // start increasing the last index
    size_t index = stride.size() - 1;
    // as long as the current index has reached 0,
    // set it to width-1 and decrease the next index
    while (idx[index] == 0 && index > 0) {
      idx[index] = width[index] - 1;
      ptr + idx[index] * stride[index];
      --index;
    }
    --idx[index];
    ptr -= stride[index];
    return *this;
  }
  iterator operator--(int) {
    iterator result(*this);
    operator--();
    return result;
  }

  iterator &operator-=(int v) {
    if (v < 0)
      return operator+=(-v);
    size_t index = stride.size() - 1;
    while (v > 0 && index >= 0) {
      size_t val = v % width[index];
      v /= width[index];
      if (val <= idx[index]) {
        idx[index] -= val;
        ptr -= val * stride[index];
      } else {
        --v;
        idx[index] += width[index] - val;
        ptr += (width[index] - val) * stride[index];
      }
      --index;
    }
    return *this;
  }

  iterator &operator+=(int v) {
    if (v < 0)
      return operator-=(-v);
    size_t index = stride.size() - 1;
    while (v > 0 && index >= 0) {
      size_t val = v % width[index];
      v /= width[index];
      idx[index] += val;
      ptr += val * stride[index];
      if (idx[index] >= width[index] && index > 0) {
        idx[index] -= width[index];
        ++idx[index - 1];
        ptr += stride[index - 1] - width[index] * stride[index];
      }
      --index;
    }
    return *this;
  }

  iterator operator+(int v) const {
    iterator result(*this);
    result += v;
    return result;
  }
  iterator operator-(int v) const {
    iterator result(*this);
    result -= v;
    return result;
  }

  T &operator[](int v) const {
    iterator iter(*this);
    iter += v;
    return *iter;
  }

  // defines equality as external friend function
  // inequality gest automatically defined by std::rel_ops
  friend bool operator==(const iterator &i, const iterator &j) {
    return i.ptr == j.ptr;
  }

  friend class tensor<T, type>;

private:
  iterator(const typename type::width_type &w,
           const typename type::index_type &s, T *p)
      : width(w), stride(s), idx(s), ptr(p) {
    std::fill(idx.begin(), idx.end(), 0);
  }

  // maintain references to width and strides
  // uses policy for acual types
  const typename type::width_type &width;
  const typename type::index_type &stride;

  // maintains both indices and pointer to data
  // uses pointer to data for dereference and equality for efficiency
  typename type::index_type idx;
  T *ptr;
};

// iterator over single index
// does not need to know actual tensor type
template <typename T> class index_iterator {
public:
  T &operator*() const { return *ptr; }

  index_iterator &operator++() {
    ptr += stride;
    return *this;
  }
  index_iterator operator++(int) {
    index_iterator result(*this);
    operator++();
    return result;
  }
  index_iterator &operator--() {
    ptr -= stride;
    return *this;
  }
  index_iterator operator--(int) {
    index_iterator result(*this);
    operator--();
    return result;
  }

  index_iterator &operator-=(int v) {
    ptr -= v * stride;
    return *this;
  }
  index_iterator &operator+=(int v) {
    ptr + -v *stride;
    return *this;
  }

  index_iterator operator+(int v) const {
    index_iterator result(*this);
    result += v;
    return result;
  }
  index_iterator operator-(int v) const {
    index_iterator result(*this);
    result -= v;
    return result;
  }

  T &operator[](int v) const { return *(ptr + v * stride); }

  friend bool operator==(const index_iterator &i, const index_iterator &j) {
    return i.ptr == j.ptr;
  }

  template <typename, typename> friend class ::tensor::tensor;

private:
  index_iterator(size_t s, T *p) : stride(s), ptr(p) {}

  size_t stride;
  T *ptr;
};
} // namespace reserved

// tensor specialization for dynamic rank
template <typename T> class tensor<T, dynamic> {
public:
  // C-style constructor with explicit rank and pointer to array of dimensions
  // all other constructors are redirected to this one
  tensor(size_t rank, const size_t dimensions[])
      : width(dimensions, dimensions + rank), stride(rank, 1UL) {
    for (size_t i = width.size() - 1UL; i != 0; --i)
      stride[i - 1] = stride[i] * width[i];
    data = std::make_shared<std::vector<T>>(stride[0] * width[0]);
    start_ptr = &(data->operator[](0));
  }
  tensor(const std::vector<size_t> &dimensions)
      : tensor(dimensions.size(), &dimensions[0]) {}
  tensor(std::initializer_list<size_t> dimensions)
      : tensor(dimensions.size(), &*dimensions.begin()) {}

  template <size_t rank> tensor(const size_t dims[rank]) : tensor(rank, dims) {}
  template <typename... Dims>
  tensor(Dims... dims)
      : width({static_cast<const size_t>(dims)...}),
        stride(sizeof...(dims), 1UL) {
    for (size_t i = width.size() - 1UL; i != 0UL; --i)
      stride[i - 1] = stride[i] * width[i];
    data = std::make_shared<std::vector<T>>(stride[0] * width[0]);
    start_ptr = &(data->operator[](0));
  }

  tensor(const tensor<T, dynamic> &X) = default;
  tensor(tensor<T, dynamic> &&X) = default;
  tensor<T, dynamic> &operator=(const tensor<T, dynamic> &X) = default;
  tensor<T, dynamic> &operator=(tensor<T, dynamic> &&X) = default;

  // all tensor types are friend
  // this are used by alien copy constructors, i.e. copy constructors copying
  // different tensor types.
  template <typename, typename> friend class tensor;

  // TODO: check this
  template <typename, class> friend class tensor_constant;
  template <typename, class, typename, typename> friend class tensor_mult;
  template <typename, class> friend class tensor_expression;

  template <size_t R>
  tensor(const tensor<T, rank<R>> &X)
      : data(X.data), width(X.width.begin(), X.width.end()),
        stride(X.stride.begin(), X.stride.end()), start_ptr(X.start_ptr) {}

  // rank accessor
  size_t get_rank() const { return width.size(); }

  // direct accessors. Similarly to std::vector, operator () does not perform
  // range check while at() does
  T &operator()(const size_t dimensions[]) const {
    const size_t rank = width.size();
    T *ptr = start_ptr;
    for (size_t i = 0; i != rank; ++i)
      ptr += dimensions[i] * stride[i];
    return *ptr;
  }
  T &at(const size_t dimensions[]) const {
    const size_t rank = width.size();
    T *ptr = start_ptr;
    for (size_t i = 0; i != rank; ++i) {
      assert(dimensions[i] < width[i]);
      ptr += dimensions[i] * stride[i];
    }
    return *ptr;
  }

  T &operator()(const std::vector<size_t> &dimensions) const {
    assert(dimensions.size() == get_rank());
    return operator()(&dimensions[0]);
  }
  T &at(const std::vector<size_t> &dimensions) const {
    assert(dimensions.size() == get_rank());
    return at(&dimensions[0]);
  }

  template <size_t rank> T &operator()(const size_t dimensions[rank]) const {
    assert(rank == get_rank());
    return operator()(static_cast<const size_t *>(dimensions));
  }
  template <size_t rank> T &at(const size_t dimensions[rank]) const {
    assert(rank == get_rank());
    return at(static_cast<const size_t *>(dimensions));
  }

  template <typename... Dims> T &operator()(Dims... dimensions) const {
    assert(sizeof...(dimensions) == get_rank());
    return operator()({static_cast<const size_t>(dimensions)...});
  }
  template <typename... Dims> T &at(Dims... dimensions) const {
    assert(sizeof...(dimensions) == get_rank());
    return at({static_cast<const size_t>(dimensions)...});
  }

  // slice operation create a new tensor type sharing the data and removing the
  // sliced index
  tensor<T, dynamic> slice(size_t index, size_t i) const {
    const size_t rank = width.size();
    assert(index < rank);
    tensor<T, dynamic> result;
    result.data = data;
    result.width.insert(result.width.end(), width.begin(),
                        width.begin() + index);
    result.width.insert(result.width.end(), width.begin() + index + 1,
                        width.end());
    result.stride.insert(result.stride.end(), stride.begin(),
                         stride.begin() + index);
    result.stride.insert(result.stride.end(), stride.begin() + index + 1,
                         stride.end());
    result.start_ptr = start_ptr + i * stride[index];

    return result;
  }
  // operator [] slices the first (leftmost) index
  tensor<T, dynamic> operator[](size_t i) const { return slice(0, i); }

  // window operation on a single index
  tensor<T, dynamic> window(size_t index, size_t begin, size_t end) const {
    tensor<T, dynamic> result(*this);
    result.width[index] = end - begin;
    result.start_ptr += result.stride[index] * begin;
    return result;
  }

  // window operations on all indices
  tensor<T, dynamic> window(const size_t begin[], const size_t end[]) const {
    tensor<T, dynamic> result(*this);
    const size_t r = get_rank();
    for (size_t i = 0; i != r; ++i) {
      result.width[i] = end[i] - begin[i];
      result.start_ptr += result.stride[i] * begin[i];
    }
    return result;
  }
  tensor<T, dynamic> window(const std::vector<size_t> &begin,
                            const std::vector<size_t> &end) const {
    return window(&(begin[0]), &(end[0]));
  }

  // flaten operation
  // do not use over windowed and sliced ranges
  tensor<T, dynamic> flatten(size_t begin, size_t end) const {
    tensor<T, dynamic> result;
    result.stride.insert(result.stride.end(), stride.begin(),
                         stride.begin() + begin);
    result.stride.insert(result.stride.end(), stride.begin() + end,
                         stride.end());
    result.width.insert(result.width.end(), width.begin(),
                        width.begin() + begin);
    result.width.insert(result.width.end(), width.begin() + end, width.end());
    for (size_t i = begin; i != end; ++i)
      result.width[end] *= width[i];
    result.start_ptr = start_ptr;
    result.data = data;
    return result;
  }

  // ====================================================================

  tensor(const tensor_expression<T, dynamic> &expression)
      : tensor(expression.evaluate()) {}

  // TODO: implement this method in every tensor template specialization
  template <char... Is> tensor_constant<T, vars<Is...>> ein() {
    return tensor_constant<T, vars<Is...>>(*this);
  }

  auto elements_count() { return stride[0] * width[0]; }

  std::vector<size_t> build_index(size_t i) {
    std::vector<size_t> result;

    for (size_t k = 0; k < stride.size(); k++) {
      result.push_back((i / stride[k]) % width[k]);
    }

    return result;
  }

  // ====================================================================

  // specialized iterator type
  typedef reserved::iterator<T, dynamic> iterator;

  iterator begin() const { return iterator(width, stride, start_ptr); }
  iterator end() const {
    iterator result = begin();
    result.idx[0] = width[0];
    result.ptr += width[0] * stride[0];
    return result;
  }

  // specialized index_iterator type
  typedef reserved::index_iterator<T> index_iterator;

  // begin and end methods producing index_iterator require the index to be
  // iterated over and all the values for the other indices
  index_iterator begin(size_t index, const size_t dimensions[]) const {
    return index_iterator(stride[index], &operator()(dimensions) -
                                             dimensions[index] * stride[index]);
  }
  index_iterator end(size_t index, const size_t dimensions[]) const {
    return index_iterator(
        stride[index], &operator()(dimensions) +
                           (width[index] - dimensions[index]) * stride[index]);
  }

  template <size_t rank>
  index_iterator begin(size_t index, const size_t dimensions[rank]) const {
    return index_iterator(stride[index], &operator()(dimensions) -
                                             dimensions[index] * stride[index]);
  }
  template <size_t rank>
  index_iterator end(size_t index, const size_t dimensions[rank]) const {
    return index_iterator(
        stride[index], &operator()(dimensions) +
                           (width[index] - dimensions[index]) * stride[index]);
  }

  index_iterator begin(size_t index,
                       const std::vector<size_t> &dimensions) const {
    return index_iterator(stride[index], &operator()(dimensions) -
                                             dimensions[index] * stride[index]);
  }
  index_iterator end(size_t index,
                     const std::vector<size_t> &dimensions) const {
    return index_iterator(
        stride[index], &operator()(dimensions) +
                           (width[index] - dimensions[index]) * stride[index]);
  }

private:
  tensor() = default;

  std::shared_ptr<std::vector<T>> data;
  dynamic::width_type width;
  dynamic::index_type stride;
  T *start_ptr;
};

// tensor specialization for fixed-rank
template <typename T, size_t R> class tensor<T, rank<R>> {
public:
  // C-style constructor with implicit rank and pointer to array of dimensions
  // all other constructors are redirected to this one
  tensor(const size_t dimensions[R]) {
    size_t *wptr = &(width[0]), *endp = &(width[0]) + R;
    while (wptr != endp)
      *(wptr++) = *(dimensions++);
    stride[R - 1] = 1;
    for (size_t i = R - 1; i != 0; --i) {
      stride[i - 1] = stride[i] * width[i];
    }
    data = std::make_shared<std::vector<T>>(stride[0] * width[0]);
    start_ptr = &(data->operator[](0));
  }

  tensor(const std::vector<size_t> &dimensions) : tensor(&dimensions[0]) {
    assert(dimensions.size() == R);
  }
  template <typename... Dims>
  tensor(Dims... dims) : width({static_cast<const size_t>(dims)...}) {
    static_assert(sizeof...(dims) == R, "size mismatch");

    stride[R - 1] = 1UL;
    for (size_t i = R - 1UL; i != 0UL; --i) {
      stride[i - 1] = stride[i] * width[i];
    }
    data = std::make_shared<std::vector<T>>(stride[0] * width[0]);
    start_ptr = &(data->operator[](0));
  }

  tensor(const tensor<T, rank<R>> &X) = default;
  tensor(tensor<T, rank<R>> &&X) = default;

  // all tensor types are friend
  // this are used by alien copy constructors, i.e. copy constructors copying
  // different tensor types.
  template <typename, typename> friend class tensor;

  template <typename, class> friend class tensor_constant;
  template <typename, class, typename, typename> friend class tensor_mult;
  template <typename, class> friend class tensor_expression;

  tensor(const tensor<T, dynamic> &X)
      : data(X.data), width(X.width.begin(), X.width.end()),
        stride(X.stride.begin(), X.stride.end()), start_ptr(X.start_ptr) {
    assert(X.get_rank() == R);
  }

  // not static so that it can be called with . rather than ::
  constexpr size_t get_rank() const { return R; }

  // direct accessors as for dynamic tensor
  T &operator()(const size_t dimensions[R]) const {
    T *ptr = start_ptr;
    for (size_t i = 0; i != R; ++i)
      ptr += dimensions[i] * stride[i];
    return *ptr;
  }
  T &at(const size_t dimensions[R]) const {
    T *ptr = start_ptr;
    for (size_t i = 0; i != R; ++i) {
      assert(dimensions[i] < width[i]);
      ptr += dimensions[i] * stride[i];
    }
    return *ptr;
  }

  T &operator()(const std::vector<size_t> &dimensions) const {
    assert(dimensions.size() == R);
    return operator()(&dimensions[0]);
  }
  T &at(const std::vector<size_t> &dimensions) const {
    assert(dimensions.size() == R);
    return at(&dimensions[0]);
  }

  // could use std::enable_if rather than static assert!
  template <typename... Dims> T &operator()(Dims... dimensions) const {
    static_assert(sizeof...(dimensions) == R, "rank mismatch");
    return operator()({static_cast<const size_t>(dimensions)...});
  }
  template <typename... Dims> T &at(Dims... dimensions) const {
    static_assert(sizeof...(dimensions) == R, "rank mismatch");
    return at({static_cast<const size_t>(dimensions)...});
  }

  // specialized iterator type
  typedef reserved::iterator<T, rank<R>> iterator;

  iterator begin() { return iterator(width, stride, start_ptr); }
  iterator end() {
    iterator result = begin();
    result.idx[0] = width[0];
    result.ptr += width[0] * stride[0];
    return result;
  }

  // specialized index_iterator type
  typedef reserved::index_iterator<T> index_iterator;

  index_iterator begin(size_t index, const size_t dimensions[R]) const {
    return index_iterator(stride[index], &operator()(dimensions) -
                                             dimensions[index] * stride[index]);
  }
  index_iterator end(size_t index, const size_t dimensions[R]) const {
    return index_iterator(
        stride[index], &operator()(dimensions) +
                           (width[index] - dimensions[index]) * stride[index]);
  }

  index_iterator begin(size_t index,
                       const std::vector<size_t> &dimensions) const {
    return index_iterator(stride[index], &operator()(dimensions) -
                                             dimensions[index] * stride[index]);
  }
  index_iterator end(size_t index,
                     const std::vector<size_t> &dimensions) const {
    return index_iterator(
        stride[index], &operator()(dimensions) +
                           (width[index] - dimensions[index]) * stride[index]);
  }

  // slicing operations return lower-rank tensors
  tensor<T, rank<R - 1>> slice(size_t index, size_t i) const {
    assert(index < R);
    tensor<T, rank<R - 1>> result;
    result.data = data;
    for (size_t i = 0; i != index; ++i) {
      result.width[i] = width[i];
      result.stride[i] = stride[i];
    }
    for (size_t i = index; i != R - 1U; ++i) {
      result.width[i] = width[i + 1];
      result.stride[i] = stride[i + 1];
    }
    result.start_ptr = start_ptr + i * stride[index];

    return result;
  }
  tensor<T, rank<R - 1>> operator[](size_t i) const { return slice(0, i); }

  // window operations do not change rank
  tensor<T, rank<R>> window(size_t index, size_t begin, size_t end) const {
    tensor<T, rank<R>> result(*this);
    result.width[index] = end - begin;
    result.start_ptr += result.stride[index] * begin;
    return result;
  }

  tensor<T, rank<R>> window(const size_t begin[], const size_t end[]) const {
    tensor<T, rank<R>> result(*this);
    for (size_t i = 0; i != R; ++i) {
      result.width[i] = end[i] - begin[i];
      result.start_ptr += result.stride[i] * begin[i];
    }
    return result;
  }
  tensor<T, dynamic> window(const std::vector<size_t> &begin,
                            const std::vector<size_t> &end) const {
    return window(&begin[0], &end[0]);
  }

  // flatten operations change rank in a way that is not known at compile time
  // would need a different interface to provide that info at compile time,
  // but the operation should not be time-critical
  tensor<T, dynamic> flatten(size_t begin, size_t end) const {
    tensor<T, dynamic> result;
    result.stride.insert(result.stride.end(), stride.begin(),
                         stride.begin() + begin);
    result.stride.insert(result.stride.end(), stride.begin() + end,
                         stride.end());
    result.width.insert(result.width.end(), width.begin(),
                        width.begin() + begin);
    result.stride.insert(result.stride.end(), stride.begin() + end,
                         stride.end());
    for (size_t i = begin; i != end; ++i)
      result.width[end] *= width[i];
    result.start_ptr = start_ptr;
    result.data = data;
    return result;
  }

  friend class tensor<T, rank<R + 1>>;

private:
  tensor() = default;

  std::shared_ptr<std::vector<T>> data;
  typename rank<R>::width_type width;
  typename rank<R>::index_type stride;
  T *start_ptr;
};

// tensor specialization for rank 1
// in this case splicing provides reference to data element
template <typename T> class tensor<T, rank<1>> {
public:
  tensor(size_t dimension) {
    data = std::make_shared<std::vector<T>>(dimension);
    start_ptr = &*(data->begin());
  }

  // all tensor types are friend
  // this are used by alien copy constructors, i.e. copy constructors copying
  // different tensor types.
  template <typename, typename> friend class tensor;

  template <typename, class> friend class tensor_constant;
  template <typename, class, typename, typename> friend class tensor_mult;
  template <typename, class> friend class tensor_expression;

  constexpr size_t get_rank() const { return 1; }

  // direct accessors as for dynamic tensor
  T &operator()(size_t d) const { return start_ptr[d * stride[0]]; }
  T &at(size_t d) const {
    assert(d < width[0]);
    return start_ptr[d * stride[0]];
  }

  T &operator()(const size_t dimensions[1]) const {
    return operator()(dimensions[0]);
  }
  T &at(const size_t dimensions[1]) const { return operator()(dimensions[0]); }

  T &operator()(const std::vector<size_t> &dimensions) const {
    assert(dimensions.size() == 1);
    return operator()(dimensions[0]);
  }
  T &at(const std::vector<size_t> &dimensions) const {
    assert(dimensions.size() == 1);
    return operator()(dimensions[0]);
  }

  // could use std::enable_if rather than static assert!

  T &slice(size_t index, size_t i) const {
    assert(index == 0);
    return *(start_ptr + i * stride[0]);
  }
  T &operator[](size_t i) { return *(start_ptr + i * stride[0]); }

  tensor<T, rank<1>> window(size_t begin, size_t end) const {
    tensor<T, rank<1>> result(*this);
    result.width[0] = end - begin;
    result.start_ptr += result.stride[0] * begin;
    return result;
  }

  typedef T *iterator;
  iterator begin(size_t = 0) { return start_ptr; }
  iterator end(size_t = 0) { return start_ptr + width[0] * stride[0]; }

  friend class tensor<T, rank<2>>;

private:
  tensor() = default;
  std::shared_ptr<std::vector<T>> data;
  rank<1>::width_type width;
  rank<1>::index_type stride;
  T *start_ptr;
};

} // namespace tensor

#endif // TENSOR
