#include <iostream>

#include "Tensor.hpp"

template <class TensorType>
void printTensor(TensorType& t) {
  auto iterator = t.begin();
  
  std::cout << std::endl << "Data:" << std::endl;
  for (; iterator < t.end(); iterator++) {
    std::cout << *iterator << ", ";
  }

  std::cout << std::endl;
}

int main() {
  Tensor<int> tensor({2, 4, 2});

  tensor[{0, 0, 0}] = 111;
  tensor[{0, 1, 0}] = 121;
  tensor[{0, 2, 0}] = 131;
  tensor[{0, 3, 0}] = 141;
  tensor[{1, 0, 0}] = 211;
  tensor[{1, 1, 0}] = 221;
  tensor[{1, 2, 0}] = 231;
  tensor[{1, 3, 0}] = 241;
  tensor[{0, 0, 1}] = 112;
  tensor[{0, 1, 1}] = 122;
  tensor[{0, 2, 1}] = 132;
  tensor[{0, 3, 1}] = 142;
  tensor[{1, 0, 1}] = 212;
  tensor[{1, 1, 1}] = 222;
  tensor[{1, 2, 1}] = 232;
  tensor[{1, 3, 1}] = 242;

  auto iterator = tensor.begin();
  std::cout << "Iteratore: " << std::endl;
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

  std::fill(tensor.begin(), tensor.end(), 69);
  // printTensor(tensor);

  // auto iteratorFill = tensor.begin();
  // while (iteratorFill < tensor.end()) {
  //   *iteratorFill = 666;
  //   iteratorFill++;
  // }

  printTensor(tensor);

  // for example a rank 3 tensor of size (3,4,5) represented in right-major
  // order will have strides (20,5,1) and width (3,4,5). Entry (i,j,k) will be
  // at index (20*i+5*j+k*1) in the flat storage.
  return 0;
}
