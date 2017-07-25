#include <iostream>
#include <vector>

using namespace std;

class Neuron
{
public:
	Neuron(unsigned index, unsigned numOutputs);

private:

}; 

Neuron::Neuron(unsigned index, unsigned numOutputs)
{
	
}

int main()
{
	// Topology e.g. (2, 4, 1)
	vector<unsigned> topology;
	topology.push_back(2);
	topology.push_back(4);
	topology.push_back(1);

	// Create layer map 2d vector
	typedef vector<Neuron> Layer;			// Neurons in the layer
	vector<Layer> m_layers;			// Input, hidden and output layer

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

		for(unsigned neuronNum = 0; neuronNum < topology[layerNum]; neuronNum++)
		{
			m_layers.back().push_back(Neuron(neuronNum, numOutputs));
			cout << "Made a neuron! - Num: " << neuronNum << " Output: " << numOutputs << endl;
		}	
	}

	// Set the input values 

	// Feedforward - count the inputs from the previous layer
	

	return 0;
}
