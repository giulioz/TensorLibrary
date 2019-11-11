# tensor-library [![Build Status](https://travis-ci.org/giulioz/TensorLibrary.svg?branch=master)](https://travis-ci.org/giulioz/TensorLibrary)

Assignment for Advanced algorithms and programming methods [CM0470] course.

This template library allows to use a special "Tensor" data structure, useful for image processing, machine learning, simulations and machine learning.

## Creation, Copy and Access

The Tensor object constructor accepts the inner value type as template, and the sizes of the dimensions as a varidic. Otherwise, you can set the dimensions with an array-like iterable:

```cpp
TensorLib::Tensor<int> t1(2, 2);

auto dims = std::array{2, 2};
TensorLib::Tensor<int> t2(dims);
```

The tensor can be accessed with the square brackets operator, passing an initializer list (with full dimensions indices) or a single number (linear index).

```cpp
TensorLib::Tensor<int> tensor(2, 2);
tensor[{0,1}] = 1;
std::cout << tensor[{0,1}]; // prints 1

tensor[0] = 2;
std::cout << tensor[0]; // prints 2
```

Dimensions are fixed, and cannot be changed without recreating the tensor. For efficency and index safety, you can also set a fixed rank at compile time:

```cpp
TensorLib::Tensor<int, TensorLib::TensorTypeFixedRank<2>> t1(2, 2);
std::cout << t1[{2, 1}]; // ok
std::cout << t1[{2, 1, 5}]; // compiler error

// compiler error
TensorLib::Tensor<int, TensorLib::TensorTypeFixedRank<2>> t2(2, 2, 5);
```

While rank can be checked in compile-time (using `TensorTypeFixedRank`), dimensions aren't known, so you can only find out of bounds errors in runtime (they are asserted and checked, when using the iterator or method `at`).

A Tensor automatically manages memory, behaving like a std::vector, deallocating data on the end of the stack object lifetime. Copy, assignment and `clone()` method produces clones of the Tensor, which does not share data. If you want to have data shared, you can use the `share()` method, which create a copy of the tensor that shares the data;

```cpp
TensorLib::Tensor<int> t1(1);
std::fill(t1.begin(), t1.end(), 0);

auto t2 = t1;
auto t3 = t2.share();
t2[0] = 100;
assert(t1[0] == 0); // true
assert(t3[0] == 100); // true
```

## Sizes

You can access tensor sizes using these methods:

```cpp
TensorLib::Tensor<int> tensor(10,5);
tensor.rank(); // returns the rank
tensor.size(); // returns the total count (10*5)
tensor.sizeAt(0); // returns the size at a given index (10)
```

## Iterators

You can operate on a Tensor using a linear Iterator:

```cpp
TensorLib::Tensor<int> t1(2, 2, 2);
std::fill(t1.begin(), t1.end(), 0);
for (auto& i : t1) {
  i = 10;
  k++;
}
```

You may also iterate on a single dimension, specifying the base coordinates and the variable index (using the `VARIABLE_INDEX` macro):

```cpp
TensorLib::Tensor<int> tensor(2, 2, 2);
auto it = tensor.constrained_cbegin({VARIABLE_INDEX, 2, 0});
auto end = tensor.constrained_cend({VARIABLE_INDEX, 2, 0});
while (it < end) {
  std::cout << *it << std::endl;
  it++;
}

// You may also specify the variable index on the last parameter, without the macro
auto it = tensor.constrained_begin({0, 2, 0}, 0);
```

Every iterator has a constant version, with the prefix `c`.

## Slicing

You can also slice a tensor using the method `slice(dimension, value)`, which returns a new tensor (that shares data with the previous):

```cpp
Tensor<int> t1(2, 2);
int k = 0;
for (auto& i : t1) {
  i = k;
  k++;
}
// Tensor data: 1, 2, 3, 4

Tensor<int> t2 = t1.slice(1, 1);
// Will contain (2, 3)

Tensor<int> t3 = t1.slice(1, 0);
// Will contain (0, 1)

Tensor<int> t4 = t1.slice(0, 0);
// Will contain (0, 2)
```

The rank of the resulting tensor is equal to the rank of the previous tensor -1. Since we can make 

Strides are kept in left-most manner.

## Flattening

You can flatten a tensor using the method `flatten(start, end)`, which take 2 parameters: `start` and `end`, that represent the indexes of the dimensions that bounds the region that you want to flatten (you can only flatten consecutive dimensions), for example:

```cpp
Tensor<int> t1(2, 2, 2);

Tensor<int> t2 = t1.flatten(1, 2);
// Will flatten to a tensor with dimensions: (2, 4)

Tensor<int> t3 = t1.flatten(0, 1);
// Will flatten to a tensor with dimensions: (4, 2)
```

The flattened tensor does not maintain the rank of the derivating tensor but it shares the same data.
