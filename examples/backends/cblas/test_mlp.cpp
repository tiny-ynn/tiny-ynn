/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#include <iostream>
#include <random>

#include "tiny_ynn/tiny_ynn.h"

const int SIZE = 100;

template <typename N>
void construct_net(N &nn, tiny_ynn::core::backend_t backend_type) {
  using relu    = tiny_ynn::relu_layer;
  using fc      = tiny_ynn::fully_connected_layer;
  using softmax = tiny_ynn::softmax_layer;

  nn << fc(SIZE, SIZE, false, backend_type) << relu()
     << fc(SIZE, SIZE, false, backend_type) << softmax();
}

int main(int argc, char **argv) {
  try {
    tiny_ynn::network<tiny_ynn::sequential> nn_internal;
    construct_net(nn_internal, tiny_ynn::core::backend_t::internal);

    tiny_ynn::network<tiny_ynn::sequential> nn_cblas;
    construct_net(nn_cblas, tiny_ynn::core::backend_t::cblas);

    tiny_ynn::vec_t input;
    for (size_t i = 0; i < SIZE; i++) {
      input.push_back(rand_r() / RAND_MAX);
    }
    auto output_internal = nn_internal.fprop(input);
    auto output_cblas    = nn_cblas.fprop(input);

    for (size_t i = 0; i < SIZE; i++) {
      std::cout << output_internal[i] << "|" << output_cblas[i] << "\t";
    }
    std::cout << std::endl;
  } catch (tiny_ynn::nn_error &err) {
    std::cerr << "Exception: " << err.what() << std::endl;
  }
}
