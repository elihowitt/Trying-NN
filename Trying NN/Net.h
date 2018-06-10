#pragma once

#include<fstream>
#include<cassert>
#include<iostream>

#include"olc.h"
#include"Neuron.h"

class Net
{
public:
	Net();
	Net(const std::vector<unsigned> , std::ofstream&);

	void feedForward(const std::vector<double>&);
	void backProp(const std::vector<double>&)   ;
	void getResults(std::vector<double>&) const;
	double getRecentAverageError(void) const { return m_recentAgerageError; }

	~Net();
private:
	std::vector<Layer> m_layers;
	double m_error;
	double m_recentAgerageError;
	double m_recentAverageSmoothingFactor;
};

