#include <iostream>

#include "Tensor.hpp"

int main() {
  Tensor<int> tensor({2, 4, 2});

  tensor[{0, 0, 0}] = 100;
  tensor[{0, 1, 0}] = 200;
  tensor[{0, 2, 0}] = 300;
  tensor[{0, 3, 0}] = 400;
  tensor[{1, 0, 0}] = 110;
  tensor[{1, 1, 0}] = 210;
  tensor[{1, 2, 0}] = 310;
  tensor[{1, 3, 0}] = 410;
  tensor[{0, 0, 1}] = 101;
  tensor[{0, 1, 1}] = 201;
  tensor[{0, 2, 1}] = 301;
  tensor[{0, 3, 1}] = 401;
  tensor[{1, 0, 1}] = 111;
  tensor[{1, 1, 1}] = 211;
  tensor[{1, 2, 1}] = 311;
  tensor[{1, 3, 1}] = 411;
  
  auto iterator = tensor.begin();
  iterator++;
  std::cout << "Iteratore: " << *iterator << std::endl;

  tensor.printTensor();

  std::cout << "0,3: " << tensor[{0, 3}] << std::endl;
  std::cout << "1,2: " << tensor[{1, 2}] << std::endl;

  std::fill(tensor.begin(), tensor.end(), 69);
  tensor.printTensor();

  auto iteratorFill = tensor.begin();
  while (iteratorFill < tensor.end()) {
    *iteratorFill = 666;
    iteratorFill++;
  }
  tensor.printTensor();

  //for example a rank 3 tensor of size (3,4,5) represented in right-major order will have strides (20,5,1) and width (3,4,5). Entry (i,j,k) will be at index (20*i+5*j+k*1) in the flat storage.
  return 0;
}
