#include <iostream>
#include <sstream>
#include <string>

#include "Tensor.hpp"

template <typename T>
void fill_tensor_i(T &tensor, int i = 0) {
  for (auto &ai : tensor) {
    ai = i;
    i++;
  }
}

template <class TensorType>
void printTensor(const TensorType &t, std::ostream &stream = std::cout) {
  for (auto iterator = t.begin(); iterator != t.end(); iterator++) {
    stream << *iterator << ", ";
  }

  stream << std::endl;
}

template <class TensorType>
void assertTensorValues(const TensorType &tensor, std::string expected) {
  std::stringstream buffer;
  printTensor(tensor, buffer);
  assert(buffer.str().compare(expected) == 0);
}

void tensorMultiplicationTests() {
  std::cout << "Tensor Multiplication Tests:" << std::endl;

  tensor::tensor<int> a({4, 3, 4});
  fill_tensor_i(a, 0);
  tensor::tensor<int> b({3});
  fill_tensor_i(b, 0);
  auto exp = a.ein<'i', 'j', 'k'>() * b.ein<'j'>();
  tensor::tensor<int> c = exp.evaluate<'i', 'k'>();
  printTensor(c);
  assertTensorValues(c,
                     "20, 23, 26, 29, 56, 59, 62, 65, 92, 95, 98, 101, 128, "
                     "131, 134, 137, \n");

  std::cout << std::endl;

  tensor::tensor<int> a1({4});
  fill_tensor_i(a1, 0);
  tensor::tensor<int> b1({4});
  fill_tensor_i(b1, 0);
  auto exp2 = a1.ein<'i'>() * b1.ein<'i'>();
  tensor::tensor<int> c1 = exp2.evaluate<>();
  printTensor(c1);
  assertTensorValues(c1, "14, \n");

  std::cout << std::endl;
}

void sameTensorMultiplicationTests() {
  std::cout << "Same Tensor Multiplication Test:" << std::endl;

  tensor::tensor<int> a({3});
  fill_tensor_i(a, 0);
  auto exp = a.ein<'i'>() * a.ein<'i'>();
  tensor::tensor<int> c = exp.evaluate<>();
  printTensor(c);
  assertTensorValues(c, "5, \n");

  tensor::tensor<int> d({3, 3});
  fill_tensor_i(d, 0);
  auto exp2 = d.ein<'i', 'j'>() * d.ein<'i', 'j'>();
  tensor::tensor<int> e = exp2.evaluate<>();
  printTensor(e);
  assertTensorValues(e, "204, \n");

  std::cout << std::endl;
}

void traceTests() {
  std::cout << "Trace Tests:" << std::endl;

  tensor::tensor<int> f({3, 3});
  fill_tensor_i(f, 0);
  auto exp3 = f.ein<'i', 'i'>();
  tensor::tensor<int> g = exp3.evaluate<>();
  printTensor(g);
  assertTensorValues(g, "12, \n");

  std::cout << std::endl;
}

void tensorAdditionTests() {
  std::cout << "Tensor Addition Tests:" << std::endl;

  tensor::tensor<int> a({4});
  fill_tensor_i(a, 0);
  tensor::tensor<int> b({4});
  fill_tensor_i(b, 0);
  auto exp = a.ein<'i'>() + b.ein<'i'>();
  tensor::tensor<int> c = exp.evaluate<'i'>();
  printTensor(c);
  assertTensorValues(c, "0, 2, 4, 6, \n");

  tensor::tensor<int> a1({2, 2, 2});
  fill_tensor_i(a1, 0);
  tensor::tensor<int> b1({2, 2, 2});
  fill_tensor_i(b1, 0);
  auto exp1 = a1.ein<'i', 'j', 'k'>() + b1.ein<'i', 'j', 'k'>();
  tensor::tensor<int> c1 = exp1.evaluate<'i', 'j', 'k'>();
  printTensor(c1);
  assertTensorValues(c1, "0, 2, 4, 6, 8, 10, 12, 14, \n");

  std::cout << std::endl;
}

void operationConcatTests() {
  std::cout << "Tensor Operation Concat Tests:" << std::endl;

  // R_i = (A_iij + B_j) * C_ij

  tensor::tensor<int> a({4, 4, 4});
  fill_tensor_i(a, 0);
  tensor::tensor<int> b({4});
  fill_tensor_i(b, 0);
  tensor::tensor<int> c({4, 4});
  fill_tensor_i(c, 0);
  auto exp = (a.ein<'i', 'i', 'j'>() + b.ein<'j'>()) * c.ein<'i', 'j'>();
  tensor::tensor<int> d = exp.evaluate<'i'>();
  printTensor(d);
  assertTensorValues(d, "790, 2830, 4870, 6910, \n");

  std::cout << std::endl;
}

void transposeTests() {
  std::cout << "Tensor Transpose Test:" << std::endl;
  tensor::tensor<int> a({2, 2});
  fill_tensor_i(a, 1);
  auto exp = a.ein<'i', 'j'>();
  tensor::tensor<int> r = exp.evaluate<'j', 'i'>();
  printTensor(r);
  assertTensorValues(r, "1, 3, 2, 4, \n");
  std::cout << std::endl;
}

int main() {
  tensorMultiplicationTests();
  sameTensorMultiplicationTests();
  traceTests();
  tensorAdditionTests();
  operationConcatTests();
  transposeTests();

  return 0;
}
