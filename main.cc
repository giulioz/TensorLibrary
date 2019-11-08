#include <iostream>

#include "Tensor.hpp"

using namespace TensorLib;

template <class TensorType>
void printTensor(TensorType& t) {
  for (auto iterator = t.cbegin(); iterator < t.cend(); iterator++) {
    std::cout << *iterator << ", ";
  }

  std::cout << std::endl;
}

template <class TensorType>
void printTensor(TensorType& t, DimensionsList& indexList) {
  for (auto iterator = t.constrained_begin(indexList);
       iterator < t.constrained_end(indexList); iterator++) {
    std::cout << *iterator << ", ";
  }

  std::cout << std::endl;
}

int main() {
  Tensor<int> tensor({2, 4, 2});

  tensor[{0, 0, 0}] = 111;
  tensor[{1, 0, 0}] = 211;
  tensor[{0, 1, 0}] = 121; //
  tensor[{1, 1, 0}] = 221; //
  tensor[{0, 2, 0}] = 131; //
  tensor[{1, 2, 0}] = 231; //
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

  std::cout << "Iterator read test: " << std::endl;
  auto iterator = tensor.begin();
  while (iterator < tensor.end()) {
    std::cout << *iterator << ", ";
    iterator++;
  }
  std::cout << std::endl;

  // auto iteratorFixed = tensor.begin({0, 0, VARIABLE_INDEX});
  // std::cout << "Iteratore Fisso: " << std::endl;
  // auto end = tensor.end({0, 0, VARIABLE_INDEX});
  // while (iteratorFixed < end) {
  //   std::cout << *iteratorFixed << ", ";
  //   iteratorFixed++;
  // }
  // std::cout << std::endl;

  // printTensor(tensor);

  // std::cout << "0,3: " << tensor[{0, 3}] << std::endl;
  // std::cout << "1,2: " << tensor[{1, 2}] << std::endl;

  // std::cout << std::endl;

  // std::cout << "Write test: ";
  // tensor[0] = 42;
  // tensor[{1, 0, 0}] = 43;
  // tensor.begin()[2] = 44;
  // printTensor(tensor);
  // std::cout << std::endl;

  // std::cout << "Copy test: " << std::endl;
  // Tensor<int> tensor2 = tensor;
  // tensor.begin()[0] = 42;
  // tensor.begin()[0] = 43;
  // std::fill(tensor.begin(), tensor.end(), 42);
  // std::cout << "Tensor1: ";
  // printTensor(tensor);
  // std::cout << "Tensor2: ";
  // printTensor(tensor2);
  // std::cout << std::endl;

  std::cout << "Constrained Test: " << std::endl;
  auto it2 = tensor.constrained_begin({VARIABLE_INDEX, 1, 0});
  auto end2 = tensor.constrained_end({VARIABLE_INDEX, 1, 0});
  auto it = tensor.constrained_begin({VARIABLE_INDEX, 2, 0});
  auto end = tensor.constrained_end({VARIABLE_INDEX, 2, 0});
  while (it < end) {
    std::cout << *it << std::endl;
    it++;
  }
  std::cout << std::endl;

  // tensor2.constrained_end({1, 0, 0})[1] = 1000;
  // printTensor(tensor2);

  return 0;
}
