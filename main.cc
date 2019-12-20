#include <chrono>
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

  tensor::tensor<int> a(4, 3, 4);
  fill_tensor_i(a, 0);
  tensor::tensor<int> b(3);
  fill_tensor_i(b, 0);
  auto exp = a.ein("ijk") * b.ein("j");
  tensor::tensor<int> c = exp.evaluate();
  printTensor(c);
  assertTensorValues(c,
                     "20, 23, 26, 29, 56, 59, 62, 65, 92, 95, 98, 101, 128, "
                     "131, 134, 137, \n");

  std::cout << std::endl;

  tensor::tensor<int> a1(4);
  fill_tensor_i(a1, 0);
  tensor::tensor<int> b1(4);
  fill_tensor_i(b1, 0);
  auto exp2 = a1.ein("i") * b1.ein("i");
  tensor::tensor<int> c1 = exp2.evaluate();
  printTensor(c1);
  assertTensorValues(c1, "14, \n");

  std::cout << std::endl;

  tensor::tensor<int> a2(4, 4);
  fill_tensor_i(a2, 0);
  tensor::tensor<int> b2(4, 4);
  fill_tensor_i(b2, 0);
  auto exp3 = a2.ein("ij") * b2.ein("ik");
  tensor::tensor<int> d2 = exp3.evaluate();
  printTensor(d2);
}

void sameTensorMultiplicationTests() {
  std::cout << "Same Tensor Multiplication Test:" << std::endl;

  tensor::tensor<int> a(3);
  fill_tensor_i(a, 0);
  auto exp = a.ein("i") * a.ein("i");
  tensor::tensor<int> c = exp.evaluate();
  printTensor(c);
  assertTensorValues(c, "5, \n");

  tensor::tensor<int> d(3, 3);
  fill_tensor_i(d, 0);
  auto exp2 = d.ein("ij") * d.ein("ij");
  tensor::tensor<int> e = exp2.evaluate();
  printTensor(e);
  assertTensorValues(e, "204, \n");

  std::cout << std::endl;
}

void traceTests() {
  std::cout << "Trace Tests:" << std::endl;

  tensor::tensor<int> f(3, 3);
  fill_tensor_i(f, 0);
  auto exp3 = f.ein("ii");
  tensor::tensor<int> g = exp3.evaluate();
  printTensor(g);
  assertTensorValues(g, "12, \n");

  tensor::tensor<int> h(3, 3, 3);
  fill_tensor_i(h, 0);
  auto exp4 = h.ein("iii");
  tensor::tensor<int> i = exp4.evaluate();
  printTensor(i);
  assertTensorValues(i, "39, \n");

  std::cout << std::endl;
}

void tensorAdditionTests() {
  std::cout << "Tensor Addition Tests:" << std::endl;

  tensor::tensor<int> a(4);
  fill_tensor_i(a, 0);
  tensor::tensor<int> b(4);
  fill_tensor_i(b, 2);
  auto exp = a.ein("i") + b.ein("i");
  tensor::tensor<int> c = exp.evaluate();
  printTensor(c);
  assertTensorValues(c, "2, 4, 6, 8, \n");

  tensor::tensor<int> a1(2, 2, 2);
  fill_tensor_i(a1, 0);
  tensor::tensor<int> b1(2, 2, 2);
  fill_tensor_i(b1, 0);
  auto exp1 = a1.ein("ijk") + b1.ein("ijk");
  tensor::tensor<int> c1 = exp1.evaluate();
  printTensor(c1);
  assertTensorValues(c1, "0, 2, 4, 6, 8, 10, 12, 14, \n");

  tensor::tensor<int> a2(3, 3);
  fill_tensor_i(a2, 0);
  auto exp2 = a2.ein("ij") + a2.ein("ji");
  tensor::tensor<int> c2 = exp2.evaluate();
  printTensor(c2);
  // assertTensorValues(c2, "2, 4, 6, 8, \n");

  std::cout << std::endl;
}

void operationConcatTests() {
  std::cout << "Tensor Operation Concat Tests:" << std::endl;

  tensor::tensor<int> a(4, 4);
  fill_tensor_i(a, 0);
  tensor::tensor<int> b(4, 4);
  fill_tensor_i(b, 0);
  tensor::tensor<int> c(4);
  fill_tensor_i(c, 0);
  auto exp = a.ein("ij") * b.ein("ik") * c.ein("j");
  tensor::tensor<int> d = exp.evaluate();
  printTensor(d);

  // tensor::tensor<int> a(4);
  // fill_tensor_i(a, 0);
  // tensor::tensor<int> b(4);
  // fill_tensor_i(b, 0);
  // tensor::tensor<int> c(4);
  // fill_tensor_i(c, 0);
  // auto exp = a.ein("i") * b.ein("i") * c.ein("i");
  // tensor::tensor<int> d = exp.evaluate();
  // printTensor(d);
  // assertTensorValues(d, "12, \n");

  // tensor::tensor<int> a1(2,2,2);
  // fill_tensor_i(a1, 0);
  // tensor::tensor<int> b1(2,2,2);
  // fill_tensor_i(b1, 0);
  // auto exp1 = a1.ein("ijk") + b1.ein("ijk");
  // tensor::tensor<int> c1 = exp1.evaluate();
  // printTensor(c1);
  // assertTensorValues(c1, "56, \n");

  std::cout << std::endl;
}

int main() {
  tensorMultiplicationTests();
  sameTensorMultiplicationTests();
  traceTests();
  tensorAdditionTests();
  operationConcatTests();

  return 0;
}
