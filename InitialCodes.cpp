#include<bits/stdc++.h>
using namespace std;

class Neuron {
private:
    double input;
    double output;
public:
    Neuron(double input) {
        this->input = input;
        activate();
    }

    void activate() {
        this->output = this->input / (1 + abs(this->input));
    }

    void setInput(double input) {
        this->input = input;
        activate();
    }

    double getOutput() {
        return this->output;
    }
};

class Matrix {
private:
    int numOfRows;
    int numOfCols;
    vector<vector<double>> entries;

    double generateRandomNumber(double min, double max) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(min, max);
        return dis(gen);
    }

public:
    Matrix(int numOfRows, int numOfCols, bool initRandom) {
        this->numOfRows = numOfRows;
        this->numOfCols = numOfCols;

        for(int i=0; i<this->numOfRows; i++) {
            vector<double> row(this->numOfCols, 0);
            if(initRandom) {
                for(int j=0; j<this->numOfCols; j++) {
                    row[j] = this->generateRandomNumber(0, 1);
                }
            }
            entries.push_back(row);
        }
    }

    int getNumOfRows() {
        return this->numOfRows;
    }

    int getNumOfCols() {
        return this->numOfCols;
    }

    double getEntryAt(int i, int j) {
        return this->entries[i][j];
    }

    void setEntryAt(int i, int j, double v) {
        this->entries[i][j] = v;
    }

    Matrix* transpose() {
        Matrix* tMatrix = new Matrix(this->numOfCols, this->numOfRows, false);
        for(int i=0; i<this->numOfCols; i++) {
            for(int j=0; j<this->numOfRows; j++) {
                tMatrix->setEntryAt(i, j, this->getEntryAt(j, i));
            }
        }
        return tMatrix;
    }

    // this * other
    Matrix* multiply(Matrix* other) {
        if(this->numOfCols != other->numOfRows) {
            throw invalid_argument("Matrix cannot be multiplied!");
        }

        Matrix* mul = new Matrix(this->numOfRows, other->numOfCols, false);
        for (int i=0; i < this->numOfRows; i++) {
            for (int j=0; j < other->numOfCols; j++) {
                for (int k=0; k < this->numOfCols; k++) {
                    mul->setEntryAt(i, j, mul->getEntryAt(i, j) + this->getEntryAt(i, k) * other->getEntryAt(k, j));
                }
            }
        }
        return mul;
    }
};

class Layer {
private:
    int numOfNeurons;
    vector<Neuron*> neurons;

public:
    Layer(int numOfNeurons) {
        this->numOfNeurons = numOfNeurons;

        for(int i=0; i<this->numOfNeurons; i++) {
            neurons.push_back(new Neuron(0));
        }
    }

    int getLayerSize() {
        return neurons.size();
    }

    /*
        Multiplies the inputs with weights and then returns the activation values from the products.
        The return value can be directly fed to the next layer.
    */
    Matrix* getOutputVector(Matrix* input, Matrix* weight) {
        try{
            Matrix* outputs = input->multiply(weight);
            for(int i=0; i<this->numOfNeurons; i++) {
                this->neurons[i]->setInput(outputs->getEntryAt(0, i));
                outputs->setEntryAt(0, i, this->neurons[i]->getOutput());
            }
            return outputs;
        } catch(const invalid_argument& e) {
            throw invalid_argument("Inputs and weights are not compatiable with each other.");
        }
    }
};

class NeuralNetwork {
private:
    vector<Layer*> layers;
    vector<Matrix*> layerWeights;

public:
    NeuralNetwork(vector<int> topology) {
        for(int i=0; i<topology.size(); i++) {
            layers.push_back(new Layer(topology[i]));
        }
        for(int i=0; i<topology.size()-1; i++) {
            layerWeights.push_back(new Matrix(topology[i], topology[i+1], true));
        }
    }

    int getNumOfLayers() {
        return layers.size();
    }

    int getSizeOfLayerAt(int i) {
        return layers[i]->getLayerSize();
    }

    Matrix* forwardPassOutput(Matrix* input) {
        if(input->getNumOfCols() != layers[0]->getLayerSize()) {
            throw invalid_argument("Invalid input to the network.");
        }
        Matrix* weightForZerothLayer = new Matrix(input->getNumOfCols(), input->getNumOfCols(), false);
        for(int i=0; i<input->getNumOfCols(); i++) {
            weightForZerothLayer->setEntryAt(i, i, 1);
        }
        Matrix* output = layers[0]->getOutputVector(input, weightForZerothLayer);
        for(int i=1; i<layers.size(); i++) {
            output = layers[i]->getOutputVector(output, layerWeights[i-1]);
        }
        return output;
    }
};

int main() {
    vector<int> topology{2, 3, 2};
    NeuralNetwork* neuralNetwork = new NeuralNetwork(topology);
    Matrix* input = new Matrix(1, 2, false);
    input->setEntryAt(0, 0, 0.9);
    input->setEntryAt(0, 1, 0.01);
    cout << "Inputs:\n";
    for(int i=0; i<input->getNumOfCols(); i++) {
        cout << input->getEntryAt(0, i) << endl;
    }
    Matrix* output = neuralNetwork->forwardPassOutput(input);
    cout << "\nOutputs:\n";
    for(int i=0; i<output->getNumOfCols(); i++) {
        cout << output->getEntryAt(0, i) << endl;
    }
}
