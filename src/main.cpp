#include <iostream>
#include <cmath>
#include <random>
#include <vector>


#define INPUT     2
#define HIDDEN    2
#define OUTPUT    2

using namespace std;

class Neuron;
typedef vector<Neuron> Layer;

struct Connection
{
  double weight;
  double deltaWeight;
};

// ======================================================
// Class - Neuron
// ======================================================
class Neuron
{
public:
  Neuron(unsigned outputs, unsigned index);
  void setOutputVal(double value) {m_outputValue = value;}
  double getOutputVal() const {return m_outputValue;}
  void feedForward(const Layer &prevLayer);
  void calculateOutputGradient(double target, Layer &prevLayer);
  void calculateHiddenGradient(const Layer &nextLayer, Layer &prevLayer);
  void setWeight(unsigned index, double weight);
  void updateWeights();
  void setDeltaWeight(double delta) {m_outputWeight[m_index].deltaWeight = delta;}

private:
  double sigmoid(double value);
  double derivativeSigmoid(double value);

  unsigned m_index;
  double m_outputValue;
  double m_gradient;
  double m_target;
  static double eta;           // Learning rate

  vector<Connection> m_outputWeight;
};

double Neuron::eta = 0.5;

void Neuron::updateWeights()
{
  for (unsigned index = 0; index < m_outputWeight.size(); index++)
  m_outputWeight[index].weight = m_outputWeight[index].weight - m_outputWeight[index].deltaWeight;
}

double Neuron::sigmoid(double value)
{
  return 1/(1 + exp(-value));
}

double Neuron::derivativeSigmoid(double value)
{
  return value*(1-value);
}

void Neuron::setWeight(unsigned index, double weight)
{
  m_outputWeight[index].weight = weight;
}

void Neuron::calculateOutputGradient(double target, Layer &prevLayer)
{
  for (unsigned n = 0; n < prevLayer.size()-1; n++)
  {
    m_target = target;
    double delta = -(m_target - m_outputValue);
    m_gradient = delta * derivativeSigmoid(m_outputValue) * prevLayer[n].m_outputValue;
    prevLayer[n].m_outputWeight[m_index].deltaWeight = m_gradient * eta;
  }
}

void Neuron::calculateHiddenGradient(const Layer &nextLayer, Layer &prevLayer)
{
  double sum = 0.;
  double delta = 0.;

  for (unsigned n = 0; n < nextLayer.size() - 1; n++)   // Exclude the bias node
  {
    delta = -(nextLayer[n].m_target - nextLayer[n].m_outputValue);
    sum += m_outputWeight[n].weight * derivativeSigmoid(nextLayer[n].m_outputValue) * delta;
  }

  for (unsigned n = 0; n < prevLayer.size()-1; n++)
  {
    m_gradient = sum * Neuron::derivativeSigmoid(m_outputValue) * prevLayer[n].m_outputValue;
    prevLayer[n].m_outputWeight[m_index].deltaWeight = m_gradient*eta;
  }
}

void Neuron::feedForward(const Layer &prevLayer)
{
  double sum = 0.0;
  // calculate the output*weights of neurons from previous Layer
  for (unsigned i = 0; i < prevLayer.size(); i++)
  {
    sum += prevLayer[i].getOutputVal() * prevLayer[i].m_outputWeight[m_index].weight;
  }

  m_outputValue = Neuron::sigmoid(sum);
}

Neuron::Neuron(unsigned outputs, unsigned index)
{
  // Real random
  random_device rd;
  mt19937 gen(rd());
  uniform_real_distribution<> dis(0, 1);

  for (unsigned i = 0; i < outputs; i++)
  {
    m_outputWeight.push_back(Connection());
    m_outputWeight.back().weight = dis(gen);
    cout << m_outputWeight.back().weight << endl;
  }

  m_index = index;
}

// ======================================================
// Class - Net
// ======================================================
class Net
{
public:
  Net(const vector<unsigned> &topology);
  void feedForward(const vector<double> &input);
  void backPropagation(const vector<double> &target);
  double printTotalError(const vector<double> &target);
  void init();

private:
  vector<Layer> m_layers;

};

void Net::feedForward(const vector<double> &input)
{
  // Assign input values to input neurons
  for (unsigned i = 0; i < input.size(); i++)
  {
    m_layers[0][i].setOutputVal(input[i]);
  }

  // Start @ the first hidden layer after the input layer
  // Per neuron calculate the output*weights of the last layer
  for (unsigned layerNum = 1; layerNum < m_layers.size(); layerNum++)
  {
    Layer &prevLayer = m_layers[layerNum - 1];
    for (unsigned n = 0; n < m_layers[layerNum].size()-1; n++)
    {
      m_layers[layerNum][n].feedForward(prevLayer);
    }
  }
}

void Net::backPropagation(const vector<double> &target)
{
  for (unsigned n = 0; n < m_layers.back().size()-1; n++)
  {
    Layer &prevLayer = m_layers[1];
    m_layers.back()[n].calculateOutputGradient(target[n], prevLayer);
  }

  // Calculate the derivative of the hidden neurons
  Layer &hiddenLayer = m_layers[m_layers.size()-2];
  for (unsigned n = 0; n < m_layers[1].size()-1; n++) // Exclude the bias node
  {
    hiddenLayer[n].calculateHiddenGradient(m_layers.back(), m_layers[0]);
  }

  for (unsigned n = 0; n < m_layers.size()-1; n++)  // No output layer opdate...
  {
    for (unsigned i = 0; i < m_layers[n].size()-1; i++) // Exclude bias nodes
    {
      m_layers[n][i].updateWeights();
    }
  }
}

double Net::printTotalError(const vector<double> &target)
{
  double sum = 0.;
  for (unsigned n = 0; n < m_layers.back().size()-1; n++)
  {
    sum += -(target[n] - m_layers.back()[n].getOutputVal());
  }

  return sum;
}

void Net::init()
{
  m_layers[0][0].setWeight(0, 0.15);
  m_layers[0][0].setWeight(1, 0.25);
  m_layers[0][1].setWeight(0, 0.2);
  m_layers[0][1].setWeight(1, 0.3);
  m_layers[0][2].setWeight(0, 0.35);
  m_layers[0][2].setWeight(1, 0.35);

  m_layers[1][0].setWeight(0, 0.4);
  m_layers[1][0].setWeight(1, 0.5);
  m_layers[1][1].setWeight(0, 0.45);
  m_layers[1][1].setWeight(1, 0.55);
  m_layers[1][2].setWeight(0, 0.6);
  m_layers[1][2].setWeight(1, 0.6);
}

Net::Net(const vector<unsigned> &topology)
{
  // Get the number of layers
  unsigned numLayers = topology.size();
  for (unsigned layerNum = 0; layerNum < numLayers; layerNum++)
  {
    m_layers.push_back(Layer());
    // Get the number of outputs per neuron
    unsigned numOutputs = layerNum == topology.size() ? 0 : topology[layerNum+1];

    // Create Neuron with neuron outputs and index
    for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; neuronNum++)    // = is adding a bias node
    {
      m_layers.back().push_back(Neuron(numOutputs, neuronNum));
    }

    // Set output of the bias node to 1.0
    m_layers.back().back().setOutputVal(1.0);
  }
}

// ======================================================
// Main
// ======================================================
int main()
{
  // Topology
  vector<unsigned> topology;
  topology.push_back(INPUT);
  topology.push_back(HIDDEN);
  topology.push_back(OUTPUT);

  // Create the neural net
  Net NN(topology);
  NN.init();

  // Input value
  vector<double> inputVals;
  inputVals.push_back(0.05);
  inputVals.push_back(0.1);

  // Feed forward
  NN.feedForward(inputVals);

  // Target values
  vector<double> targetVals;
  targetVals.push_back(0.09);
  targetVals.push_back(0.91);

  // Backward propogation
  cout << "Total error is " << NN.printTotalError(targetVals) << endl;
  cout << endl << "backPropagation" << endl;
  for (unsigned cnt = 0; cnt < 10000; cnt++)
  {
    NN.backPropagation(targetVals);
    NN.feedForward(inputVals);
  }

  cout << "Total error is " << NN.printTotalError(targetVals) << endl;

  return 0;
}
