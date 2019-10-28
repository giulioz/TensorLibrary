#include <iostream>

#include "Tensor.hpp"

int main() {
  Tensor<int> tensor({2, 4});
  tensor[{0, 0}] = 1;
  tensor[{0, 1}] = 2;
  tensor[{0, 2}] = 3;
  tensor[{0, 3}] = 4;
  tensor[{1, 0}] = 1;
  tensor[{1, 1}] = 2;
  tensor[{1, 2}] = 3;
  tensor[{1, 3}] = 4;
  // tensor[{0}] = 10;

  // std::cout << tensor[{0}] << std::endl;
  tensor.printTensor();
  return 0;
}
