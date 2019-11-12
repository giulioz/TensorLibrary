#include <iostream>
#include <sstream>
#include <string>

#include "Tensor.hpp"

using namespace TensorLib;

template <class TensorType>
void printTensor(TensorType& t, std::ostream& stream = std::cout) {
  for (auto iterator = t.cbegin(); iterator < t.cend(); iterator++) {
    stream << *iterator << ", ";
  }

  stream << std::endl;
}

template <class TensorType>
void assertTensorValues(TensorType tensor, std::string expected) {
  std::stringstream buffer;
  printTensor(tensor, buffer);
  assert(buffer.str().compare(expected) == 0);
}

void fixedRankTest() {
  std::cout << "Fixed Rank Test: " << std::endl;
  Tensor<int, 2> tensor(2, 2);
  tensor[{0, 0}] = 11;
  tensor[{1, 0}] = 21;
  tensor[{0, 1}] = 12;
  tensor[{1, 1}] = 22;
  printTensor(tensor);
  assertTensorValues(tensor, "11, 21, 12, 22, \n");

  auto it = tensor.constrained_begin({1, VARIABLE_INDEX});
  auto end = tensor.constrained_end({1, VARIABLE_INDEX});
  while (it < end) {
    *it += 100;
    it++;
  }
  printTensor(tensor);
  assertTensorValues(tensor, "11, 121, 12, 122, \n");

  std::cout << std::endl;
}

void sharingTest() {
  std::cout << "Sharing Test: " << std::endl;
  Tensor<int> t1(2, 2);
  std::fill(t1.begin(), t1.end(), 9);
  Tensor<int> t2 = t1.clone();

  t2[{0, 1}] = 12;
  printTensor(t1);
  assertTensorValues(t1, "9, 9, 9, 9, \n");
  printTensor(t2);
  assertTensorValues(t2, "9, 9, 12, 9, \n");

  Tensor t3 = t1;
  t3[{0, 1}] = 12;
  printTensor(t1);
  // assertTensorValues(t1, "9, 9, 9, 9, \n");
  printTensor(t3);
  assertTensorValues(t3, "9, 9, 12, 9, \n");

  std::cout << std::endl;
}

void sliceTest() {
  std::cout << "Slicing Test: " << std::endl;
  Tensor<int> t1(2, 2);
  std::fill(t1.begin(), t1.end(), 0);
  int k = 0;
  for (auto& i : t1) {
    i = k;
    k++;
  }
  std::cout << "[0,0]: " << t1[{0, 0}] << std::endl;
  std::cout << "[0,1]: " << t1[{0, 1}] << std::endl;
  std::cout << "[1,0]: " << t1[{1, 0}] << std::endl;
  std::cout << "[1,1]: " << t1[{1, 1}] << std::endl;
  printTensor(t1);
  assertTensorValues(t1, "0, 1, 2, 3, \n");

  std::cout << "Slice (1,1): ";
  Tensor<int> t2 = t1.slice(1, 1);
  assertTensorValues(t2, "2, 3, \n");
  printTensor(t2);

  std::cout << "Slice (1,0): ";
  Tensor<int> t3 = t1.slice(1, 0);
  assertTensorValues(t3, "0, 1, \n");
  printTensor(t3);

  std::cout << "Slice (0,0): ";
  Tensor<int> t4 = t1.slice(0, 0);
  assertTensorValues(t4, "0, 2, \n");
  printTensor(t4);

  std::cout << "Slice (0,1): ";
  Tensor<int> t5 = t1.slice(0, 1);
  assertTensorValues(t5, "1, 3, \n");
  printTensor(t5);
  std::cout << std::endl << "Higher Rank" << std::endl;

  Tensor<int> th1(2, 2, 2);
  std::fill(th1.begin(), th1.end(), 0);
  int k1 = 0;
  for (auto& j : th1) {
    j = k1;
    k1++;
  }

  std::cout << "Slice (0,0): ";
  Tensor<int> th2 = th1.slice(0, 0);
  assertTensorValues(th2, "0, 2, 4, 6, \n");
  printTensor(th2);

  std::cout << std::endl;

  // TODO: test with double sliced tensor
}

void flattenTest() {
  std::cout << "Flatten Test: " << std::endl;
  Tensor<int> t1(2, 2, 2);
  std::fill(t1.begin(), t1.end(), 0);
  int k = 0;
  for (auto& i : t1) {
    i = k;
    k++;
  }
  printTensor(t1);
  assertTensorValues(t1, "0, 1, 2, 3, 4, 5, 6, 7, \n");

  std::cout << "Flatten (0,1): " << std::endl;
  Tensor<int> t2 = t1.flatten(0, 1);
  // printTensor(t2);
  // std::cout << "Sizes t1: ";
  // for (auto&& size : t1.sizes) {
  //   std::cout << size << ", ";
  // }
  // std::cout << std::endl;
  // std::cout << "Sizes t2: ";
  // for (auto&& size : t2.sizes) {
  //   std::cout << size << ", ";
  // }
  std::cout << std::endl;

  Tensor<int> t3(2, 2, 2, 2);
  std::fill(t3.begin(), t3.end(), 0);
  int k2 = 0;
  for (auto& j : t3) {
    j = k2;
    k2++;
  }
  printTensor(t3);

  std::cout << "Flatten (1,2): " << std::endl;
  Tensor<int> t4 = t3.flatten(1, 2);
  // std::cout << "Sizes t3: ";
  // for (auto&& size : t3.sizes) {
  //   std::cout << size << ", ";
  // }
  // std::cout << std::endl;
  // std::cout << "Sizes t4: ";
  // for (auto&& size : t4.sizes) {
  //   std::cout << size << ", ";
  // }
  std::cout << std::endl;
  printTensor(t4);
}

int main() {
  auto dims = std::array{2, 2};
  Tensor<int> t0(dims);

  Tensor<int> tensor({2, 4, 2});

  tensor[{0, 0, 0}] = 111;
  tensor[{1, 0, 0}] = 211;
  tensor[{0, 1, 0}] = 121;
  tensor[{1, 1, 0}] = 221;
  tensor[{0, 2, 0}] = 131;
  tensor[{1, 2, 0}] = 231;
  tensor[{0, 3, 0}] = 141;
  tensor[{1, 3, 0}] = 241;
  tensor[{0, 0, 1}] = 112;
  tensor[{1, 0, 1}] = 212;
  tensor[{0, 1, 1}] = 122;
  tensor[{1, 1, 1}] = 222;
  tensor[{0, 2, 1}] = 132;
  tensor[{1, 2, 1}] = 232;
  tensor[{0, 3, 1}] = 142;
  tensor[{1, 3, 1}] = 242;

  Tensor<int> tensorCopy = tensor;

  std::cout << "Iterator read test: " << std::endl;
  auto iterator = tensor.begin();
  while (iterator < tensor.end()) {
    std::cout << *iterator << ", ";
    iterator++;
  }
  std::cout << std::endl;

  std::cout << "Write test: ";
  tensor[0] = 42;
  tensor[{1, 0, 0}] = 43;
  tensor.begin()[2] = 44;
  printTensor(tensor);
  std::cout << std::endl;

  std::cout << "Copy test: " << std::endl;
  Tensor<int> tensor2 = tensor;
  tensor.begin()[0] = 42;
  tensor.begin()[0] = 43;
  std::fill(tensor.begin(), tensor.end(), 42);
  std::cout << "Tensor1: ";
  printTensor(tensor);
  std::cout << "Tensor2: ";
  printTensor(tensor2);
  std::cout << std::endl;

  std::cout << "Constrained Test: " << std::endl;
  auto it = tensorCopy.constrained_cbegin({VARIABLE_INDEX, 2, 0});
  auto end = tensorCopy.constrained_cend({VARIABLE_INDEX, 2, 0});
  while (it < end) {
    std::cout << *it << std::endl;
    it++;
  }
  std::cout << std::endl;

  tensorCopy.constrained_begin({VARIABLE_INDEX, 0, 0})[1] = 1000;
  printTensor(tensorCopy);
  std::cout << std::endl;

  fixedRankTest();
  sharingTest();
  sliceTest();
  flattenTest();

  return 0;
}
