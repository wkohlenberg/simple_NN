#include <iostream>
#include <fstream>
#include <stdint.h>
#include <vector>
#include <string> 

using namespace std;

// Endiannes
int endianSwap(uint32_t a)
{
	return (a<<24) | ((a<<8) & 0x00FF0000) | ((a>>8) & 0x0000FF00) | (a>>24);
}

/*
	Return values:
	- 0: no error
	- 1: cannot find or open file
	- 2: no matching header for given file
*/
int readLabels(vector<int>&vLabels, bool training)
{
	string	filePath = "../database/";
	int samples = 0;

	if (training)
	{
		filePath.append("train-labels-idx1-ubyte");
		samples = 60000;
	}
	else
	{
		filePath.append("t10k-labels-idx1-ubyte");
		samples = 10000;
	}

	// Open the label file, training or test 
	ifstream labelFile; 
	labelFile.open(filePath.c_str());
	if (!labelFile.is_open())
	{
		cout << "Couldn't open or find the file." << endl;
		return 1;
	}
	
	// Read the header
	uint32_t magic;
	uint32_t nLabels;
	uint8_t label;
	labelFile.read(reinterpret_cast<char *>(&magic), sizeof(magic));
	labelFile.read(reinterpret_cast<char *>(&nLabels), sizeof(nLabels));

	if (!((endianSwap(magic) == 2049) && (endianSwap(nLabels) == samples)))
	{
		cout << "No corresponding header..." << endl;
		return 2;
	}

	for (int i = 0; i < samples; i++)
	{
		labelFile.read(reinterpret_cast<char *>(&label), sizeof(label));
		vLabels.push_back(int(label));
	}		

	return 0;
}

int readImages(vector<vector<int> >&vImages, bool training=true)
{
	string filePath = "../database/";
	int samples = 0;
	
	if (training)
	{
		filePath.append("train-images-idx3-ubyte");
		samples = 60000;
	}
	else
	{
		filePath.append("t10k-images-idx3-ubyte");
		samples = 10000;
	}

	// Open the image files, training or test set
	ifstream imageFile;
	imageFile.open(filePath.c_str());
	if (!imageFile.is_open())
	{
		cout << "Couldn't open or find the file." << endl;
		return 1;
	}

	// Read the header
	uint32_t magic;
	uint32_t nImages;
	uint32_t rows;
	uint32_t columns;
	uint8_t pixel;
	imageFile.read(reinterpret_cast<char *>(&magic), sizeof(magic));
	imageFile.read(reinterpret_cast<char *>(&nImages), sizeof(nImages));
	imageFile.read(reinterpret_cast<char *>(&rows), sizeof(rows));
	imageFile.read(reinterpret_cast<char *>(&columns), sizeof(columns));	

	if(!((endianSwap(magic) == 2051) && (endianSwap(nImages) == samples)))
	{
		cout << "No corresponding header..." << endl;
		return 2;
	}

	
	vector<int> temp;
	int pixelsImg = endianSwap(rows) * endianSwap(columns);
	
	for (int y = 0; y < samples; y++)
	{
		for (int x = 0; x < pixelsImg; x++)
		{
			imageFile.read(reinterpret_cast<char *>(&pixel), sizeof(pixel));
			temp.push_back(int(pixel));
		}
		vImages.push_back(temp);
		temp.clear();
	}

	return 0;
}

// Get input vector for the input neurons
int getInput(vector<int> &input, vector<vector<int> > training, int index)
{
	for (unsigned n = 0; n < training[index].size(); n++)
	{
		input.push_back(training[index].at(n));
	}	

	return 0; 
}

// Get output vector for the output neurons
int getOutput(vector<int> &output, vector<int> labels, int index)
{
	// Load vector ouput with zero's
	fill (output.begin(), output.end(), 0);
	
	int label = labels.at(index);
	for (unsigned i = 0; i < labels.size(); i++)
	{
		if (int(i)== label)
		{
			output.at(i) = 1;
		}
	}

	return 0;
}
