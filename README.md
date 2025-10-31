#include <iostream>
#include <vector>
#include <cmath>
#include <random>

using namespace std;

struct Layer {
    vector<vector<double>> weights;
    vector<double> biases;
    vector<double> outputs;
};

class NeuralNetwork {
private:
    vector<Layer> layers;
    double learning_rate;

    double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
    double sigmoid_derivative(double x) { return x * (1 - x); }

public:
    NeuralNetwork(const vector<int>& structure, double lr = 0.1) : learning_rate(lr) {
        random_device rd; mt19937 gen(rd()); uniform_real_distribution<> dis(-1.0, 1.0);
        for (size_t i = 1; i < structure.size(); ++i) {
            Layer layer;
            layer.weights.resize(structure[i], vector<double>(structure[i-1]));
            layer.biases.resize(structure[i]);
            for (auto& row : layer.weights) for (auto& w : row) w = dis(gen);
            for (auto& b : layer.biases) b = dis(gen);
            layers.push_back(layer);
        }
    }

    vector<double> forward(const vector<double>& input) {
        vector<double> activations = input;
        for (auto& layer : layers) {
            vector<double> next(layer.biases.size());
            for (size_t i = 0; i < next.size(); ++i) {
                next[i] = layer.biases[i];
                for (size_t j = 0; j < activations.size(); ++j)
                    next[i] += activations[j] * layer.weights[i][j];
                next[i] = sigmoid(next[i]);
            }
            layer.outputs = next;
            activations = next;
        }
        return activations;
    }
};

int main() {
    NeuralNetwork nn({2, 3, 1});
    vector<double> input = {0.0, 1.0};
    vector<double> output = nn.forward(input);
    cout << "Output: " << output[0] << endl;
    return 0;
}
