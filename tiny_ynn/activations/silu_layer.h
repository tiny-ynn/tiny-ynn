/*
    Copyright (c) 2017, Pierre Gabin FODOP GUMETE and the respective contributors
    All rights reserved.
    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <string>
#include <utility>
#include <math.h>

#include "tiny_ynn/activations/activation_layer.h"
#include "tiny_ynn/layers/layer.h"

namespace tiny_ynn {

class silu_layer : public activation_layer {
 public:
  using activation_layer::activation_layer;

  std::string layer_type() const override { return "silu-activation"; }

  void forward_activation(const vec_t &x, vec_t &y) override {
    for (size_t j = 0; j < x.size(); j++) {
      y[j] = x[j] / (1+exp(-x[j]));
    }
  }

  void backward_activation(const vec_t &x,
                           const vec_t &y,
                           vec_t &dx,
                           const vec_t &dy) override {
    for (size_t j = 0; j < x.size(); j++) {
      // dx = dy * (gradient of silu)
      dx[j] = dy[j] + (1+exp(-y[j]+ y[j]*exp(-y[j])) / pow(1+exp(-y[j]), 2));
    }
  }

  std::pair<float_t, float_t> scale() const override {
    return std::make_pair(float_t(-0.8), float_t(0.8));
  }

  friend struct serialization_buddy;
};

}  // namespace tiny_ynn