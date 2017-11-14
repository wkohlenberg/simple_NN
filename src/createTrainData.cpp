#include <iostream>
#include <fstream>
#include <random>

#define TRAINING_INPUT        2000            // 2000 inputs and target outputs

using namespace std;

int main()
{
  ofstream trainData ("input/xor_train_data.txt", ios::out);
  if (trainData.is_open())
  {
    // Real random
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, 1);

    int input1, input2;

    for (int i = 0; i < TRAINING_INPUT; i++)
    {
      input1 = dis(gen);
      input2 = dis(gen);
      trainData << "in: " << input1 << " " << input2 << endl;
      trainData << "out: " << (input1^input2) << endl;
    }

    trainData.close();
  }
  else
  {
    cout << "Could not open file..." << endl;
  }

  return 0;
}
