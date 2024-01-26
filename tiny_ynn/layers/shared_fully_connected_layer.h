/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/

#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tiny_ynn/layers/layer.h"

namespace tiny_ynn {

/**
 * compute fully-connected(matmul) operation
 **/
class shared_fully_connected_layer : public layer {
public:
    shared_fully_connected_layer(size_t x_size, size_t y_size,size_t z_size,bool has_bias=true,bool parallel=true)
        :layer(std_input_order(has_bias), // x, W and b
    {vector_type::data}),
          x_size_(x_size),
          y_size_(y_size),
          z_size_(z_size),
          has_bias_(has_bias),
          layer_parallelize(parallel)
    {}
    std::string layer_type() const override {
        return "shared-fully-connected";
    }
    std::vector<shape3d> in_shape() const override {
        if(has_bias_){
            // return input shapes
            // order of shapes must be equal to argument of layer constructor
            return { shape3d(x_size_, 1, z_size_), // x
                        shape3d(x_size_, y_size_, 1), // W
                        shape3d(y_size_, 1, 1) }; // b
        }else
        {
            return { shape3d(x_size_, 1, z_size_), // x
                        shape3d(x_size_, y_size_, 1)}; // W
        }
    }
    std::vector<shape3d> out_shape() const override {
        return { shape3d(y_size_, 1, z_size_) }; // y
    }
    void forward_propagation(const std::vector<tensor_t *> &in_data,
                             std::vector<tensor_t *> &out_data) override{
        const tensor_t &in = *in_data[0];
        const vec_t &W     = (*in_data[1])[0];
        const vec_t *b     = (has_bias_?&(*in_data[2])[0]:nullptr);
        tensor_t &out      = *out_data[0];

        for(size_t sample=0;sample<in.size();sample++){
            const vec_t &x = in[sample];
            vec_t &y      = out[sample];
            std::fill(y.begin(), y.end(), 0.0);
            // y = Wx+b for each channel;
          for_(layer_parallelize, 0, z_size_, [&](const blocked_range &ch_range) {
            for (size_t ch = ch_range.begin(); ch <ch_range.end(); ch++){
                for (size_t r = 0; r < y_size_; r++) {
                    for (size_t c = 0; c < x_size_; c++)
                        y[ch*y_size_+r] += W[c*y_size_+r]*x[ch*x_size_+c];
                    if(has_bias_){
                        y[ch*y_size_+r] += (*b)[r];
                    }
                }
            }
          });
        }
    }
    void back_propagation(const std::vector<tensor_t *> &in_data,
                          const std::vector<tensor_t *> &out_data,
                          std::vector<tensor_t *> &out_grad,
                          std::vector<tensor_t *> &in_grad) override {
        // incoming/outcoming data
        const tensor_t& curr_delta = *out_grad[0]; // dE/dy (already calculated in nextlayer)
        const tensor_t& prev_out = *in_data[0];
        //        const tensor_t& x = *in_data[0];
        const vec_t& W = (*in_data[1])[0];
        tensor_t& prev_delta = *in_grad[0]; // dE/dx (passed into previous layer)
        tensor_t& dW         = *in_grad[1]; // dE/dW

        for(size_t sample=0;sample<prev_out.size();sample++){
          for_(layer_parallelize, 0, z_size_, [&](const blocked_range &ch_range) {
            for (size_t ch = ch_range.begin();ch < ch_range.end(); ch++){
                for (size_t c = 0; c < x_size_; c++)
                    for (size_t r = 0; r < y_size_; r++)
                        prev_delta[sample][ch*x_size_+c] += curr_delta[sample][ch*y_size_+r] * W[c*y_size_+r];
            // accumulate weight difference
                for (size_t r = 0; r < y_size_; r++)
                    for (size_t c = 0; c < x_size_; c++)
                        dW[sample][c*y_size_+r] += curr_delta[sample][ch*y_size_+r] * prev_out[sample][ch*x_size_+c];
            // accumulate bias difference
                if(has_bias_){
                    tensor_t& db         = *in_grad[2]; // dE/db
                    // propagate delta to prev-layer
                    for (size_t r = 0; r < y_size_; r++)
                        db[sample][r] += curr_delta[sample][ch*y_size_+r];
                }
            }
          });
        }
    }
private:
    size_t x_size_;
    size_t y_size_;
    size_t z_size_;
    bool has_bias_;
    bool layer_parallelize;
};

}  // namespace tiny_ynn

