#ifndef TENSOR_H
#define TENSOR_H

#include <vector>

template <typename ValueType>
class Tensor {
  std::vector<ValueType> data;

 public:
  ValueType operator()() { return data[indices[0]]; }
};

#endif
