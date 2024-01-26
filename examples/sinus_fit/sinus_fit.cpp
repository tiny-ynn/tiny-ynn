/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.
    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/

// this example shows how to use tiny-ynn library to fit data, by learning a
// sinus function.

// please also see:
// https://github.com/tiny-ynn/tiny-ynn/blob/master/docs/how_tos/How-Tos.md

#include <iostream>

#include "tiny_ynn/tiny_ynn.h"

int main() {
  // create a simple network with 2 layer of 10 neurons each
  // input is x, output is sin(x)
  tiny_ynn::network<tiny_ynn::sequential> net;
  net << tiny_ynn::fully_connected_layer(1, 10);
  net << tiny_ynn::tanh_layer();
  net << tiny_ynn::fully_connected_layer(10, 10);
  net << tiny_ynn::tanh_layer();
  net << tiny_ynn::fully_connected_layer(10, 1);

  // create input and desired output on a period
  std::vector<tiny_ynn::vec_t> X;
  std::vector<tiny_ynn::vec_t> sinusX;
  for (float x = -3.1416f; x < 3.1416f; x += 0.2f) {
    tiny_ynn::vec_t vx    = {x};
    tiny_ynn::vec_t vsinx = {sinf(x)};

    X.push_back(vx);
    sinusX.push_back(vsinx);
  }

  // set learning parameters
  size_t batch_size = 16;    // 16 samples for each network weight update
  int epochs        = 2000;  // 2000 presentation of all samples
  tiny_ynn::adamax opt;

  // this lambda function will be called after each epoch
  int iEpoch              = 0;
  auto on_enumerate_epoch = [&]() {
    // compute loss and disp 1/100 of the time
    iEpoch++;
    if (iEpoch % 100) return;

    double loss = net.get_loss<tiny_ynn::mse>(X, sinusX);
    std::cout << "epoch=" << iEpoch << "/" << epochs << " loss=" << loss
              << std::endl;
  };

  // learn
  std::cout << "learning the sinus function with 2000 epochs:" << std::endl;
  net.fit<tiny_ynn::mse>(opt, X, sinusX, batch_size, epochs, []() {},
                         on_enumerate_epoch);

  std::cout << std::endl
            << "Training finished, now computing prediction results:"
            << std::endl;

  // compare prediction and desired output
  float fMaxError = 0.f;
  for (float x = -3.1416f; x < 3.1416f; x += 0.2f) {
    tiny_ynn::vec_t xv = {x};
    float fPredicted   = net.predict(xv)[0];
    float fDesired     = sinf(x);

    std::cout << "x=" << x << " sinX=" << fDesired
              << " predicted=" << fPredicted << std::endl;

    // update max error
    float fError = fabs(fPredicted - fDesired);

    if (fMaxError < fError) fMaxError = fError;
  }

  std::cout << std::endl << "max_error=" << fMaxError << std::endl;

  return 0;
}
