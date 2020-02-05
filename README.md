# tensor-library

Third Assignment for Advanced algorithms and programming methods [CM0470] course.

We made some changes to an existing Tensor Library to allow operations using the Einstein Notation, distributing the workload over multiple threads. This allows better performances when performing calculations on very large tensors.

We decided to keep the API unchanged, while changing the behaviour underneath. This way it's possible to adopt the parallel version without any cost. Any call to the assignment operator with einstein expressions spawns several threads of execution.

```cpp
tensor<size_t, rank<2>> t1(SIZE, SIZE), t2(SIZE, SIZE);

auto i = new_index;
auto j = new_index;
t2(j, i) = t1(i, j);
```
