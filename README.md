# tensor-library [![Build Status](https://travis-ci.org/giulioz/TensorLibrary.svg?branch=master)](https://travis-ci.org/giulioz/TensorLibrary)

Assignment for Advanced algorithms and programming methods [CM0470] course.

This template library allows to use a special "Tensor" data structure, useful for image processing, machine learning, simulations and machine learning.

The Tensor object constructor accepts the inner value type as template, and the sizes of the dimensions as a varidic. Otherwise, you can set the dimensions with an array-like iterable:

```cpp
Tensor<int> t1(2, 2);

auto dims = std::array{2, 2};
Tensor<int> t2(dims);
```

Dimensions are fixed, and cannot be changed without recreating the tensor. For efficency and index safety, you can also set a fixed rank at compile time:

```cpp
Tensor<int, TensorTypeFixedRank<2>> t1(2, 2);
std::cout << t1[{2, 1}]; // ok
std::cout << t1[{2, 1, 5}]; // compiler error

Tensor<int, TensorTypeFixedRank<2>> t2(2, 2, 5); // compiler error
```

While rank can be checked in compile-time (using `TensorTypeFixedRank`), dimensions aren't known, so you can only find out of bounds errors in runtime (they are asserted and checked, when using the it).

The tensor can be accessed with the square brackets operator, passing an initializer list (with full dimensions indices) or a single number (linear index).

```cpp
Tensor<int> tensor(2, 2);
tensor[{0,1}] = 1;
std::cout << tensor[{0,1}]; // prints 1

tensor[0] = 2;
std::cout << tensor[0]; // prints 2
```

A Tensor automatically manages memory, behaving like a std::vector, deallocating data on the end of the stack object lifetime. Copy and assignment produces clones of the Tensor, that does not share data.

```cpp
Tensor<int> t1(1);
std::fill(t1.begin(), t1.end(), 0);

auto t2 = t1;
t2[0] = 100;
assert(t1[0] == 0); // true
```

operations like copy, slice and flatten) create tensors that share the data, thus referring to the same memory region. A modified cell will be seen on all the other derived tensors. If you want to create a deep copy of the tensor you can call the `clone()` method.



The rank of the 
