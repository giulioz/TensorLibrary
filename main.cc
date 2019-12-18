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

void test1() {
  std::cout << "Test1:" << std::endl;

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
}

void test2() {
  std::cout << "Test2:" << std::endl;

  tensor::tensor<int> a(4);
  fill_tensor_i(a, 0);
  tensor::tensor<int> b(4);
  fill_tensor_i(b, 0);
  auto exp = a.ein("i") * b.ein("i");
  tensor::tensor<int> c = exp.evaluate();
  printTensor(c);
  assertTensorValues(c, "14, \n");

  std::cout << std::endl;
}

void test3() {
  std::cout << "Test3:" << std::endl;

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

  tensor::tensor<int> f(3, 3);
  fill_tensor_i(f, 0);
  auto exp3 = f.ein("ii");
  tensor::tensor<int> g = exp3.evaluate();
  printTensor(g);
  assertTensorValues(g, "36, \n");

  std::cout << std::endl;
}

void test4() {
  // std::cout << "Test4:" << std::endl;

  // tensor::tensor<int> a(6);
  // fill_tensor_i(a, 0);
  // auto exp = a.ein<'i', 'i'>();
  // tensor::tensor<int> c = exp.evaluate();
  // printTensor(c);

  // std::cout << std::endl;
}

int main() {
  test1();
  test2();
  test3();
  test4();

  return 0;
}
