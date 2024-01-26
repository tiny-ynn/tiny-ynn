/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tiny_ynn/config.h"
#include "tiny_ynn/network.h"
#include "tiny_ynn/nodes.h"

#include "tiny_ynn/core/framework/tensor.h"

#include "tiny_ynn/core/framework/device.h"
#include "tiny_ynn/core/framework/program_manager.h"

#include "tiny_ynn/activations/asinh_layer.h"
#include "tiny_ynn/activations/elu_layer.h"
#include "tiny_ynn/activations/leaky_relu_layer.h"
#include "tiny_ynn/activations/relu_layer.h"
#include "tiny_ynn/activations/selu_layer.h"
#include "tiny_ynn/activations/sigmoid_layer.h"
#include "tiny_ynn/activations/softmax_layer.h"
#include "tiny_ynn/activations/softplus_layer.h"
#include "tiny_ynn/activations/softsign_layer.h"
#include "tiny_ynn/activations/tanh_layer.h"
#include "tiny_ynn/activations/tanh_p1m2_layer.h"
#include "tiny_ynn/layers/arithmetic_layer.h"
#include "tiny_ynn/layers/average_pooling_layer.h"
#include "tiny_ynn/layers/average_unpooling_layer.h"
#include "tiny_ynn/layers/batch_normalization_layer.h"
#include "tiny_ynn/layers/cell.h"
#include "tiny_ynn/layers/cells.h"
#include "tiny_ynn/layers/concat_layer.h"
#include "tiny_ynn/layers/convolutional_layer.h"
#include "tiny_ynn/layers/deconvolutional_layer.h"
#include "tiny_ynn/layers/dropout_layer.h"
#include "tiny_ynn/layers/fully_connected_layer.h"
#include "tiny_ynn/layers/shared_fully_connected_layer.h"
#include "tiny_ynn/layers/global_average_pooling_layer.h"
#include "tiny_ynn/layers/input_layer.h"
#include "tiny_ynn/layers/l2_normalization_layer.h"
#include "tiny_ynn/layers/lrn_layer.h"
#include "tiny_ynn/layers/lrn_layer.h"
#include "tiny_ynn/layers/max_pooling_layer.h"
#include "tiny_ynn/layers/max_unpooling_layer.h"
#include "tiny_ynn/layers/power_layer.h"
#include "tiny_ynn/layers/quantized_convolutional_layer.h"
#include "tiny_ynn/layers/quantized_deconvolutional_layer.h"
#include "tiny_ynn/layers/recurrent_layer.h"
#include "tiny_ynn/layers/slice_layer.h"
#include "tiny_ynn/layers/zero_pad_layer.h"
#include "tiny_ynn/activations/bstep_layer.h"
#include "tiny_ynn/activations/gaussian_layer.h"
#include "tiny_ynn/activations/gelu_layer.h"
#include "tiny_ynn/activations/silu_layer.h"
#include "tiny_ynn/lossfunctions/loss_function.h"


#ifdef CNN_USE_GEMMLOWP
#include "tiny_ynn/layers/quantized_fully_connected_layer.h"
#endif  // CNN_USE_GEMMLOWP

#include "tiny_ynn/lossfunctions/loss_function.h"
#include "tiny_ynn/optimizers/optimizer.h"

#include "tiny_ynn/util/deform.h"
#include "tiny_ynn/util/graph_visualizer.h"
#include "tiny_ynn/util/product.h"
#include "tiny_ynn/util/weight_init.h"
#include "tiny_ynn/util/nms.h"

#include "tiny_ynn/io/cifar10_parser.h"
#include "tiny_ynn/io/display.h"
#include "tiny_ynn/io/layer_factory.h"
#include "tiny_ynn/io/mnist_parser.h"

#ifdef DNN_USE_IMAGE_API
#include "tiny_ynn/util/image.h"
#endif  // DNN_USE_IMAGE_API

#ifndef CNN_NO_SERIALIZATION
#include "tiny_ynn/util/deserialization_helper.h"
#include "tiny_ynn/util/serialization_helper.h"
// to allow upcasting
CEREAL_REGISTER_TYPE(tiny_ynn::elu_layer)
CEREAL_REGISTER_TYPE(tiny_ynn::leaky_relu_layer)
CEREAL_REGISTER_TYPE(tiny_ynn::relu_layer)
CEREAL_REGISTER_TYPE(tiny_ynn::sigmoid_layer)
CEREAL_REGISTER_TYPE(tiny_ynn::softmax_layer)
CEREAL_REGISTER_TYPE(tiny_ynn::softplus_layer)
CEREAL_REGISTER_TYPE(tiny_ynn::softsign_layer)
CEREAL_REGISTER_TYPE(tiny_ynn::tanh_layer)
CEREAL_REGISTER_TYPE(tiny_ynn::tanh_p1m2_layer)
#endif  // CNN_NO_SERIALIZATION

// shortcut version of layer names
namespace tiny_ynn {
namespace layers {

using conv = tiny_ynn::convolutional_layer;

using q_conv = tiny_ynn::quantized_convolutional_layer;

using max_pool = tiny_ynn::max_pooling_layer;

using ave_pool = tiny_ynn::average_pooling_layer;

using fc = tiny_ynn::fully_connected_layer;

using sfc = tiny_ynn::shared_fully_connected_layer;

using dense = tiny_ynn::fully_connected_layer;

using zero_pad = tiny_ynn::zero_pad_layer;

// using rnn_cell = tiny_ynn::rnn_cell_layer;

#ifdef CNN_USE_GEMMLOWP
using q_fc = tiny_ynn::quantized_fully_connected_layer;
#endif

using add = tiny_ynn::elementwise_add_layer;

using dropout = tiny_ynn::dropout_layer;

using input = tiny_ynn::input_layer;

using linear = linear_layer;

using lrn = tiny_ynn::lrn_layer;

using concat = tiny_ynn::concat_layer;

using deconv = tiny_ynn::deconvolutional_layer;

using max_unpool = tiny_ynn::max_unpooling_layer;

using ave_unpool = tiny_ynn::average_unpooling_layer;

}  // namespace layers

namespace activation {

using sigmoid = tiny_ynn::sigmoid_layer;

using asinh = tiny_ynn::asinh_layer;

using tanh = tiny_ynn::tanh_layer;

using relu = tiny_ynn::relu_layer;

using rectified_linear = tiny_ynn::relu_layer;

using softmax = tiny_ynn::softmax_layer;

using leaky_relu = tiny_ynn::leaky_relu_layer;

using elu = tiny_ynn::elu_layer;

using selu = tiny_ynn::selu_layer;

using tanh_p1m2 = tiny_ynn::tanh_p1m2_layer;

using softplus = tiny_ynn::softplus_layer;

using softsign = tiny_ynn::softsign_layer;

}  // namespace activation

#include "tiny_ynn/models/alexnet.h"

using batch_norm = tiny_ynn::batch_normalization_layer;

using l2_norm = tiny_ynn::l2_normalization_layer;

using slice = tiny_ynn::slice_layer;

using power = tiny_ynn::power_layer;

}  // namespace tiny_ynn

#ifdef CNN_USE_CAFFE_CONVERTER
// experimental / require google protobuf
#include "tiny_ynn/io/caffe/layer_factory.h"
#endif
