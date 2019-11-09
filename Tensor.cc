#include "Tensor.hpp"

const size_t TensorLib::VARIABLE_INDEX = -1;

size_t TensorLib::coordsToIndex(DimensionsList& coords,
                                const std::vector<size_t>& strides) {
  size_t index = 0;
  size_t strideIndex = 0;
  for (auto&& coord : coords) {
    index += coord * (strides.at(strideIndex));
    strideIndex++;
  }

  return index;
}

size_t TensorLib::calcDataSize(DimensionsList dimensions) {
  return std::accumulate(std::begin(dimensions), std::end(dimensions), 1,
                         std::multiplies<double>());
}

std::vector<size_t> TensorLib::calcStrides(DimensionsList dimensions) {
  std::vector<size_t> strides;

  size_t stride = 1;
  for (auto&& dim : dimensions) {
    strides.push_back(stride);
    stride *= dim;
  }
  return strides;
}

size_t TensorLib::calcFixedStartIndex(const std::vector<size_t>& strides,
                                      DimensionsList& indexList,
                                      const size_t& width) {
  size_t index = 0;
  size_t position = 0;
  for (auto&& fixedValue : indexList) {
    if (fixedValue != VARIABLE_INDEX) {
      position += fixedValue * strides[index];
    } else if (width != 0) {
      position += width * strides[index];
    }
    index++;
  }

  return position;
}

size_t TensorLib::findFixedIndex(DimensionsList& indexList) {
  size_t index = 0;
  for (auto&& fixedValue : indexList) {
    if (fixedValue == VARIABLE_INDEX) {
      return index;
    }
    index++;
  }

  return index;
}
