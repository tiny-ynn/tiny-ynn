/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#include <iostream>

#include "tiny_ynn/tiny_ynn.h"

// rescale output to 0-100
template <typename Activation>
double rescale(double x) {
  Activation a(1);
  return 100.0 * (x - a.scale().first) / (a.scale().second - a.scale().first);
}

void convert_image(const std::string &imagefilename,
                   double minv,
                   double maxv,
                   int w,
                   int h,
                   tiny_ynn::vec_t &data) {
  tiny_ynn::image<> img(imagefilename, tiny_ynn::image_type::rgb);
  tiny_ynn::image<> resized = resize_image(img, w, h);
  data.resize(resized.width() * resized.height() * resized.depth());
  for (size_t c = 0; c < resized.depth(); ++c) {
    for (size_t y = 0; y < resized.height(); ++y) {
      for (size_t x = 0; x < resized.width(); ++x) {
        data[c * resized.width() * resized.height() + y * resized.width() + x] =
          (maxv - minv) * (resized[y * resized.width() + x + c]) / 255.0 + minv;
      }
    }
  }
}

template <typename N>
void construct_net(N &nn) {
  using conv    = tiny_ynn::convolutional_layer;
  using pool    = tiny_ynn::max_pooling_layer;
  using fc      = tiny_ynn::fully_connected_layer;
  using relu    = tiny_ynn::relu_layer;
  using softmax = tiny_ynn::softmax_layer;

  const size_t n_fmaps  = 32;  ///< number of feature maps for upper layer
  const size_t n_fmaps2 = 64;  ///< number of feature maps for lower layer
  const size_t n_fc = 64;  ///< number of hidden units in fully-connected layer

  nn << conv(32, 32, 5, 3, n_fmaps, tiny_ynn::padding::same)  // C1
     << pool(32, 32, n_fmaps, 2)                              // P2
     << relu(16, 16, n_fmaps)                                 // activation
     << conv(16, 16, 5, n_fmaps, n_fmaps, tiny_ynn::padding::same)  // C3
     << pool(16, 16, n_fmaps, 2)                                    // P4
     << relu(8, 8, n_fmaps)                                        // activation
     << conv(8, 8, 5, n_fmaps, n_fmaps2, tiny_ynn::padding::same)  // C5
     << pool(8, 8, n_fmaps2, 2)                                    // P6
     << relu(4, 4, n_fmaps2)                                       // activation
     << fc(4 * 4 * n_fmaps2, n_fc)                                 // FC7
     << fc(n_fc, 10) << softmax(10);                               // FC10
}

void recognize(const std::string &dictionary, const std::string &src_filename) {
  tiny_ynn::network<tiny_ynn::sequential> nn;

  construct_net(nn);

  // load nets
  std::ifstream ifs(dictionary.c_str());
  ifs >> nn;

  // convert imagefile to vec_t
  tiny_ynn::vec_t data;
  convert_image(src_filename, -1.0, 1.0, 32, 32, data);

  // recognize
  auto res = nn.predict(data);
  std::vector<std::pair<double, int>> scores;

  // sort & print top-3
  for (int i = 0; i < 10; i++)
    scores.emplace_back(rescale<tiny_ynn::tanh_layer>(res[i]), i);

  sort(scores.begin(), scores.end(), std::greater<std::pair<double, int>>());

  for (int i = 0; i < 3; i++)
    std::cout << scores[i].second << "," << scores[i].first << std::endl;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cout << "please specify image file";
    return 0;
  }
  recognize("cifar-weights", argv[1]);
}
