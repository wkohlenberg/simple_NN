#ifndef MNIST_H
#define MNIST_H

#include <stdint.h>
#include <vector>

using namespace std;

uint32_t endianSwap(uint32_t a);
int readLabels(vector<int>&vLabels, bool training=true);
int readImages(vector<vector<int> >&vImages, bool training=true);
int getInput(vector<int> &input, vector<vector<int> > training, int index);
int getOutput(vector<int> &output, vector<int> labels, int index);

#endif
