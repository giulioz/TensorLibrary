#include <iostream>

#include "Tensor.hpp"

int main() {
  Tensor<int> tensor({5, 2});
  tensor[{0}] = 10;

  std::cout << tensor[{0}] << std::endl;
  return 0;
}
