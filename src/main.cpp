#include <iostream>
#include <vector>
#include <cmath>

#include "mnist.h"

#define INPUTS		784
#define HIDDEN		30
#define OUTPUTS		10


using namespace std;

class Neuron;
// Create layer map
typedef vector<Neuron> Layer; 	// Neurons in the layers
vector<Layer> m_layers;			// input, hidden and output layer

struct Connection
{
	double weight;
	double deltaWeight;
};

class Neuron
{
public:
	Neuron(unsigned index, unsigned numOutputs);
	void setOutputVal(double value) { m_outputVal = value; }
	double getOutputVal() const { return m_outputVal; }
	unsigned getIndex() { return m_index; }
	void feedForward(const Layer &prevLayer);	
	void calcOutputGradients(double targetVal);
	void calcHiddenGradients(const Layer &nextLayer);
	void updateInputWeights(Layer &prevLayer);
	
private:
	double randomWeight() { return rand() / double(RAND_MAX);}	

	double m_outputVal;
	vector<Connection> m_outputWeights;
	unsigned m_index;
	double m_gradient;
	static double eta;
	static double alpha;
}; 

double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;

void Neuron::updateInputWeights(Layer &prevLayer)
{
	// The weights to be updated are in the Connection container
	for (unsigned n = 0; n < prevLayer.size(); n++)
	{
		Neuron &neuron = prevLayer[n];
		double oldDeltaWeight = neuron.m_outputWeights[m_index].deltaWeight;
		double newDeltaWeight = eta * neuron.getOutputVal() * m_gradient + alpha  *oldDeltaWeight;

		neuron.m_outputWeights[m_index].deltaWeight = newDeltaWeight;
		neuron.m_outputWeights[m_index].weight += newDeltaWeight;

		// cout << neuron.m_outputWeights[m_index].weight << endl;
	}
	// cout << endl;
}

void Neuron::feedForward(const Layer &prevLayer)
{
	double sum = 0.0;

	// Sum the previous layer's outputs (which are our inputs)
	for (unsigned n = 0; n < prevLayer.size(); n++)
	{
		sum += prevLayer[n].getOutputVal() * prevLayer[n].m_outputWeights[m_index].weight;
	}

	m_outputVal = tanh(sum);
}

void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
	double dow;
	
	for (unsigned n = 0; n < nextLayer.size() - 1; n++)
	{
		dow += m_outputWeights[n].weight * nextLayer[n].m_gradient;
	}
	
	m_gradient = dow * (1.0 - m_outputVal * m_outputVal);
}

void Neuron::calcOutputGradients(double targetVal)
{
	double delta = targetVal - m_outputVal;
	m_gradient = delta * (1.0 - m_outputVal * m_outputVal);
}

Neuron::Neuron(unsigned index, unsigned numOutputs)
{
	// Create a weight for every output
	for (unsigned outputNum = 0; outputNum < numOutputs; outputNum++)
	{
		m_outputWeights.push_back(Connection());
		m_outputWeights.back().weight = randomWeight();
		// cout << m_outputWeights.back().weight << endl;
	}			

	m_index = index;
}

int main()
{
	//////////////////////////////////////////////////
    // initialization
	////////////////////////////////////////////////// 

	// Load training data
	vector<int> trainingLabels;
	readLabels(trainingLabels);

	// Topology e.g. (2, 4, 1)
	vector<unsigned> topology;
	topology.push_back(INPUTS);
	topology.push_back(HIDDEN);
	topology.push_back(OUTPUTS);

	// Input vector
	vector<int> inputVals;
	
	// Output vector
	vector<int> targetVals (10);

	// Settings
	double m_recentAverageSmoothingFactor = 100.0;

	// Fill each layer with neurons
	unsigned numLayers = topology.size();
	for (unsigned layerNum = 0; layerNum < numLayers; layerNum++)
	{
		m_layers.push_back(Layer());
		
		// Outputs depends on the number of neurons in the next layer
		unsigned numOutputs = 0;
		if (layerNum != (topology.size() - 1))
		{	
			numOutputs = topology[layerNum+1];
		}

		for(unsigned neuronNum = 0; neuronNum <= topology[layerNum]; neuronNum++)
		{
			m_layers.back().push_back(Neuron(neuronNum, numOutputs));
			// cout << "Made a neuron! - Num: " << neuronNum << " Output: " << numOutputs << endl;
		}	
		
		// Set output of the bias neuron to 1.0
		m_layers.back().back().setOutputVal(1.0);
	}

	// Load dataset here 
	vector<int> labels;
	vector<vector<int> > images;
	
	readLabels(labels);
	readImages(images);
	cout << "Loaded the images and labels..." << endl;

	//////////////////////////////////////////////////
	// Loop through data
	//////////////////////////////////////////////////
for (unsigned k = 0; k < 60000; k++)
{
	// Load the first image from the dataset
	getInput(inputVals, images, k);
	getOutput(targetVals, labels, k);
	
	for (unsigned i = 0; i < inputVals.size(); i++)
	{
		m_layers[0][i].setOutputVal(inputVals[i]);		
	} 
		
	// Feedforward - count the inputs from the previous layer
	for (unsigned layerNum = 1; layerNum < m_layers.size(); layerNum++)
	{
		Layer &prevLayer = m_layers[layerNum - 1];
		for (unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n )
		{
			m_layers[layerNum][n].feedForward(prevLayer);
		}
	}	

	// Get the output results
	vector<double> resultVals;
	for (unsigned n = 0; n < m_layers.back().size() - 1; n++)
	{
		resultVals.push_back(m_layers.back()[n].getOutputVal());
	}
	
	// Backpropagation by calculating the overall net error
	// (RMS of output neuron errors)
	Layer &outputLayer = m_layers.back();
	double m_error = 0.0;

	for (unsigned n = 0; n < outputLayer.size() - 1; n++)	
	{
		double delta = targetVals[n] - outputLayer[n].getOutputVal();
		m_error += (delta * delta);
	}

	m_error /= (outputLayer.size() - 1);
	m_error = sqrt(m_error);

	// Implement a recent average measurement
	double m_recentAverageError = 0.0;
	m_recentAverageError =  (m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
				/ (m_recentAverageSmoothingFactor + 1.0);

	// Calculate output layer gradients
	for (unsigned n = 0; n < outputLayer.size() - 1; n++)
	{
		outputLayer[n].calcOutputGradients(targetVals[n]);
	}

	// Calc hidden layer gradients
	for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; layerNum--)
	{
		Layer &hiddenLayer = m_layers[layerNum];
		Layer &nextLayer = m_layers[layerNum + 1];

		for (unsigned n = 0; n < hiddenLayer.size(); n++)
		{
			hiddenLayer[n].calcHiddenGradients(nextLayer);
		}
	}

	// For all layers from outputs to first hidden layer,
	// update connection weights
	for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; layerNum--)
	{
		Layer &layer = m_layers[layerNum];
		Layer &prevLayer = m_layers[layerNum - 1];

		for (unsigned n = 0; n < layer.size() - 1; n++)
		{
			layer[n].updateInputWeights(prevLayer);
		}
	}

	// Show output
	cout << endl << k << " Label: " << endl;
	vector<int>::iterator it;
	for (it = targetVals.begin(); it < targetVals.end(); it++)
	{
		cout << *it << " ";
	}
	
	cout << endl << "Output: " << endl;
	vector<double>::iterator itr;
	for (itr = resultVals.begin(); itr < resultVals.end(); itr++)
	{
		cout << *itr << " ";
	}
	cout << endl;
}
	return 0;
}
