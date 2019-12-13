#include <chrono>
#include <iostream>

#include "Tensor.hpp"

int main() {
  tensor::tensor<int> td(2, 2, 3);
  tensor::tensor<int, tensor::rank<3>> tr(2, 2, 3);

  int count = 0;
  tensor::tensor<int> t2 = td;
  for (auto iter = t2.begin(); iter != t2.end(); ++iter) *iter = count++;

  t2 = tr;
  for (auto iter = t2.begin(); iter != t2.end(); ++iter) *iter = count++;

  for (auto iter = td.begin(); iter != td.end(); ++iter)
    std::cout << *iter << ' ';
  std::cout << '\n';

  for (auto iter = tr.begin(); iter != tr.end(); ++iter)
    std::cout << *iter << ' ';
  std::cout << '\n';

  for (auto iter = tr.begin(2, {0, 0, 1}); iter != tr.end(2, {0, 0, 1}); ++iter)
    std::cout << *iter << ' ';
  std::cout << '\n';

  for (auto iter = td.begin(1, {0, 0, 1}); iter != td.end(1, {0, 0, 1}); ++iter)
    std::cout << *iter << ' ';
  std::cout << '\n';

  t2 = td.window(2, 0, 2);
  for (auto iter = t2.begin(); iter != t2.end(); ++iter)
    std::cout << *iter << ' ';
  std::cout << '\n';

  // ====================================================================

  tensor::tensor<int> a(4, 4, 4);
  tensor::tensor<int> b(4);

  // const char i = 'i';
  // const char j = 'j';
  // const char k = 'k';
  // a.ein<i, j, k>();
  // a.ein<'i'>();
  a.ein("ijk");

  // tensor::tensor<int> c = a.ein<"ijk">() * b.ein("j");

  // auto indexA = 1;
  // auto indexB = 0;

  // cik = Σj aijk bj
  // tensor::tensor<int, tensor::rank<2>> c(4, 4);
  // for (size_t i = 0; i < a.count(); i++) {

  // }

  // for (size_t i = 0; i < 4; i++) {
  //   for (size_t j = 0; j < 4; j++) {
  //     for (size_t k = 0; k < 4; k++) {
  //       c(i, k) += a(i, j, k) * b(j);
  //     }
  //   }
  // }

  // tensor::tensor<int> c(4, 4);
  // auto tr = c.sumIndices("ii");

  // std::cout << "Trace:" << tr << std::endl;
}
