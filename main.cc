#include <chrono>
#include <iostream>

#include "Tensor.hpp"

int main() {
  // tensor::tensor<int> td(2, 2, 3);
  // tensor::tensor<int, tensor::rank<3>> tr(2, 2, 3);

  // int count = 0;
  // tensor::tensor<int> t2 = td;
  // for (auto iter = t2.begin(); iter != t2.end(); ++iter) *iter = count++;

  // t2 = tr;
  // for (auto iter = t2.begin(); iter != t2.end(); ++iter) *iter = count++;

  // for (auto iter = td.begin(); iter != td.end(); ++iter)
  //   std::cout << *iter << ' ';
  // std::cout << '\n';

  // for (auto iter = tr.begin(); iter != tr.end(); ++iter)
  //   std::cout << *iter << ' ';
  // std::cout << '\n';

  // for (auto iter = tr.begin(2, {0, 0, 1}); iter != tr.end(2, {0, 0, 1});
  // ++iter)
  //   std::cout << *iter << ' ';
  // std::cout << '\n';

  // for (auto iter = td.begin(1, {0, 0, 1}); iter != td.end(1, {0, 0, 1});
  // ++iter)
  //   std::cout << *iter << ' ';
  // std::cout << '\n';

  // t2 = td.window(2, 0, 2);
  // for (auto iter = t2.begin(); iter != t2.end(); ++iter)
  //   std::cout << *iter << ' ';
  // std::cout << '\n';

  // ====================================================================

  int i = 0;

  tensor::tensor<int> a(4, 4, 4);
  for (auto &ai : a) {
    ai = i;
    i++;
  }

  i = 100;
  tensor::tensor<int> b(4);
  for (auto &bi : b) {
    bi = i;
    i++;
  }

  auto exp = a.ein("ijk") * b.ein("j");

  tensor::tensor<int> c = exp.evaluate();
  std::cout << "Values: ";
  for (auto &&i : c) {
    std::cout << i << ", ";
  }
  std::cout << std::endl;
}
