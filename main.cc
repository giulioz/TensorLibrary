#include <iostream>

#include "Tensor.hpp"

int main() {
  Tensor<int> tensor({2, 4, 2});
  tensor[{0, 0, 0}] = 1;
  tensor[{0, 1, 0}] = 2;
  tensor[{0, 2, 0}] = 3;
  tensor[{0, 3, 0}] = 4;
  tensor[{1, 0, 0}] = 1;
  tensor[{1, 1, 0}] = 2;
  tensor[{1, 2, 0}] = 3;
  tensor[{1, 3, 0}] = 4;
  tensor[{0, 0, 1}] = 11;
  tensor[{0, 1, 1}] = 22;
  tensor[{0, 2, 1}] = 33;
  tensor[{0, 3, 1}] = 44;
  tensor[{1, 0, 1}] = 11;
  tensor[{1, 1, 1}] = 22;
  tensor[{1, 2, 1}] = 33;
  tensor[{1, 3, 1}] = 44;
  // tensor[{0}] = 10;

  // std::cout << tensor[{0}] << std::endl;
  tensor.printTensor();
  return 0;
}
