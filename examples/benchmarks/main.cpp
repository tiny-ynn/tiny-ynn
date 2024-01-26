/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/

#include <iostream>

#include "tiny_ynn/tiny_ynn.h"

int main() {
  tiny_ynn::models::alexnet nn;

  // change all layers at once
  nn.weight_init(tiny_ynn::weight_init::constant(2.0));
  nn.bias_init(tiny_ynn::weight_init::constant(2.0));
  nn.init_weight();

  tiny_ynn::vec_t in(224 * 224 * 3);

  // generate random variables
  tiny_ynn::uniform_rand(in.begin(), in.end(), 0, 1);

  tiny_ynn::timer t;  // start the timer

  // predict
  auto res = nn.predict(in);

  double elapsed_s = t.elapsed();
  t.stop();

  std::cout << "Elapsed time(s): " << elapsed_s << std::endl;
}
