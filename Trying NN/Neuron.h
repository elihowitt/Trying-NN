#pragma once
#include<vector>
#include<cmath>

struct Connection
{
	double weight;
	double deltaWeight;
};
class Neuron;
typedef std::vector<Neuron> Layer;

class Neuron
{
public:
	Neuron(unsigned, unsigned);

	void setOutputVal(double val) { m_outputVal = val; }
	double getOutputVal(void) const { return m_outputVal; }

	void updateInputWeights(Layer&);

	void calcOutputGradients(double);
	void calcHiddenGradients(const Layer&);

	void feedForward(Layer&);

	~Neuron();
private:
	static double eta;
	static double alpha;
	static double transferFunction(double);
	static double transferFunctionDerivative(double);
	static double randomWeight() { return rand() / double(RAND_MAX); }
	double sumDOW(const Layer&) const;
	double m_outputVal;
	std::vector<Connection> m_outputWeights;
	unsigned m_myIndex;
	double m_gradient;


};

