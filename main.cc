#include <iostream>
#include <sstream>
#include <string>

#include "Tensor.hpp"

using namespace TensorLib;

template <class TensorType>
void printTensor(const TensorType& t, std::ostream& stream = std::cout) {
  for (auto iterator = t.cbegin(); iterator < t.cend(); iterator++) {
    stream << *iterator << ", ";
  }

  stream << std::endl;
}

template <class TensorType>
void assertTensorValues(const TensorType& tensor, std::string expected) {
  std::stringstream buffer;
  printTensor(tensor, buffer);
  assert(buffer.str().compare(expected) == 0);
}

void fixedRankTest() {
  std::cout << "Fixed Rank Test: " << std::endl;
  Tensor<int, 2> tensor = Tensor<int>::buildTensor(2, 2);
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
  Tensor t1 = Tensor<int>::buildTensor(2, 2);
  std::fill(t1.begin(), t1.end(), 9);
  std::cout << "Tensor<int> t2 = t1.clone()" << std::endl;

  Tensor t2 = t1.clone();
  std::cout << "t2 = t1" << std::endl;
  t2 = t1;

  t2[{0, 1}] = 12;
  printTensor(t1);
  assertTensorValues(t1, "9, 9, 9, 9, \n");
  printTensor(t2);
  assertTensorValues(t2, "9, 9, 12, 9, \n");

  Tensor t3 = t1;
  t3[{0, 1}] = 12;
  printTensor(t1);
  assertTensorValues(t1, "9, 9, 9, 9, \n");
  printTensor(t3);
  assertTensorValues(t3, "9, 9, 12, 9, \n");

  Tensor t4 = t3.share();
  *t4.begin() = 100;
  printTensor(t3);
  assertTensorValues(t3, "100, 9, 12, 9, \n");
  printTensor(t4);
  assertTensorValues(t4, "100, 9, 12, 9, \n");
  assertTensorValues(t1, "9, 9, 9, 9, \n");

  std::cout << std::endl;
}

void sliceTest() {
  std::cout << "Slicing Test: " << std::endl;
  Tensor t1 = Tensor<int>::buildTensor(2, 2);
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
  Tensor t2 = t1.slice(1, 1);
  assertTensorValues(t2, "2, 3, \n");
  printTensor(t2);

  std::cout << "Slice (1,0): ";
  Tensor t3 = t1.slice(1, 0);
  assertTensorValues(t3, "0, 1, \n");
  printTensor(t3);

  std::cout << "Slice (0,0): ";
  Tensor t4 = t1.slice(0, 0);
  assertTensorValues(t4, "0, 2, \n");
  printTensor(t4);

  std::cout << "Slice (0,1): ";
  Tensor t5 = t1.slice(0, 1);
  assertTensorValues(t5, "1, 3, \n");
  printTensor(t5);
  std::cout << std::endl << "Higher Rank" << std::endl;

  Tensor th1 = Tensor<int>::buildTensor(2, 2, 2);
  std::fill(th1.begin(), th1.end(), 0);
  int k1 = 0;
  for (auto& j : th1) {
    j = k1;
    k1++;
  }

  std::cout << "Slice (0,0): ";
  Tensor th2 = th1.slice(0, 0);
  assertTensorValues(th2, "0, 2, 4, 6, \n");
  printTensor(th2);

  std::cout << std::endl;

  std::cout << "Constrained Test: " << std::endl;
  auto it = th2.constrained_cbegin({VARIABLE_INDEX, 1});
  auto end = th2.constrained_cend({VARIABLE_INDEX, 1});
  while (it < end) {
    std::cout << *it << " ";
    it++;
  }
  std::cout << std::endl;
}

void flattenTest() {
  std::cout << "Flatten Test: " << std::endl;
  Tensor<int> t1 = Tensor<int>::buildTensor(2, 2, 2);
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
  printTensor(t2);
  std::cout << std::endl;

  {
    std::cout << "Constrained Test: " << std::endl;
    auto it = t2.constrained_cbegin({VARIABLE_INDEX, 1});
    auto end = t2.constrained_cend({VARIABLE_INDEX, 1});
    while (it < end) {
      std::cout << *it << " ";
      it++;
    }
  }

  Tensor<int> t3 = Tensor<int>::buildTensor(2, 2, 2, 2);
  std::fill(t3.begin(), t3.end(), 0);
  k = 0;
  for (auto& j : t3) {
    j = k;
    k++;
  }
  printTensor(t3);

  std::cout << "Flatten (1,2): " << std::endl;
  Tensor<int> t4 = t3.flatten(1, 2);
  printTensor(t4);
  std::cout << std::endl;

  {
    std::cout << "Constrained Test: " << std::endl;
    auto it = t4.constrained_cbegin({0, VARIABLE_INDEX, 1});
    auto end = t4.constrained_cend({0, VARIABLE_INDEX, 1});
    while (it < end) {
      std::cout << *it << " ";
      it++;
    }
  }

  std::cout << std::endl;
}

void creationTest() {
  std::cout << "Dynamic" << std::endl;
  Tensor<int> t0({2, 4, 6});

  std::cout << "Copy Dynamic" << std::endl;
  Tensor<int> t1 = t0;
  t1[0] = 100;
  assert(t0[0] != 100);

  std::cout << "Sized" << std::endl;
  Tensor t100 = Tensor<int>::buildTensor(2, 4, 6);
  std::cout << "Copy Sized-Dynamic" << std::endl;
  Tensor<int> t101 = t100;
  t101[0] = 100;
  assert(t100[0] != 100);

  std::cout << "Move Dynamic" << std::endl;
  Tensor<int> t2 = std::move(t0);

  std::cout << "Dynamic" << std::endl;
  Tensor<int> t3({2, 4, 6});
  Tensor<int> t30({2, 4, 6});
  std::cout << "Copy Dynamic-Dynamic" << std::endl;
  t3 = t30;
  std::cout << "Build fixed, move fixed" << std::endl;
  Tensor t4 = Tensor<int>::buildTensor(2, 4, 6);

  std::cout << "Fixed Initializer" << std::endl;
  Tensor t5 = Tensor<int, 3>({2, 4, 6});

  std::cout << std::endl;
}

int main() {
  creationTest();

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
