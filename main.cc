#include <chrono>
#include <iostream>
#include <sstream>
#include <string>

#include "Tensor.hpp"

template <typename T> void fill_tensor_i(T &tensor, int i = 0) {
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

  tensor::tensor<int> a(4, 4, 4);
  fill_tensor_i(a, 0);
  tensor::tensor<int> b(4);
  fill_tensor_i(b, 100);
  auto exp = a.ein<'i', 'j', 'k'>() * b.ein<'j'>();
  tensor::tensor<int> c = exp.evaluate();
  printTensor(c);

  std::cout << std::endl;
}

void test2() {
  std::cout << "Test2:" << std::endl;

  tensor::tensor<int> a(4);
  fill_tensor_i(a, 0);
  tensor::tensor<int> b(4);
  fill_tensor_i(b, 100);
  auto exp = a.ein<'i'>() * b.ein<'i'>();
  tensor::tensor<int> c = exp.evaluate();
  printTensor(c);

  std::cout << std::endl;
}

void test3() {
  std::cout << "Test3:" << std::endl;

  tensor::tensor<int> a(6);
  fill_tensor_i(a, 0);
  auto exp = a.ein<'i'>() * a.ein<'i'>();
  tensor::tensor<int> c = exp.evaluate();
  printTensor(c);

  std::cout << std::endl;
}

void test4() {
  std::cout << "Test4:" << std::endl;

  tensor::tensor<int> a(6);
  fill_tensor_i(a, 0);
  auto exp = a.ein<'i', 'i'>();
  tensor::tensor<int> c = exp.evaluate();
  printTensor(c);

  std::cout << std::endl;
}

int main() {
  test1();
  test2();
  test3();
  test4();

  return 0;
}
