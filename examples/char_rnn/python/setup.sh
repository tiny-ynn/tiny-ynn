cd tiny_ynn
swig -c++ -python tiny_char_rnn.i
python3 setup.py build_ext --inplace
