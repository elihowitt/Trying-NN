#include <fstream>
#include<Windows.h>
#include"Net.h"

void showVectorVals(char* label, std::vector<double> &v, std::ofstream& f)
{
	f << label << " ";
	for (unsigned i = 0; i < v.size(); ++i) {
		f << v[i] << " ";
	}

	f << std::endl;
}

void genTrainingSetAbc(std::vector<double>& in, std::vector<double>& out)
{
	double a = rand() % 2;
	double b = rand() % 2;
	double c = rand() % 2;
	in.push_back(a);
	in.push_back(b);
	in.push_back(c);
	out.push_back(!a && b && c);
}
void genTrainingSetAXORB(std::vector<double>& in, std::vector<double>& out)
{
	double a = rand() % 2;
	double b = rand() % 2;
	in.push_back(a);
	in.push_back(b);
	out.push_back(!(a == b));

}

int main()
{
	std::ofstream fileLogResults("logResults.txt");

	std::vector<unsigned> topology;
	//Test
	//topology.push_back(3);
	//topology.push_back(3);
	//topology.push_back(2);
	//topology.push_back(1);
	topology.push_back(3);
	topology.push_back(3);
	topology.push_back(1);

	Net net(topology);

	std::vector<double> inputVals ;
	std::vector<double> targetVals ;
	std::vector<double> resultVals;

	bool bPause = false;
	int i = 0;
	while (!GetAsyncKeyState(VK_SPACE) && i++ <3000)
	{
		if (!bPause)
		{
			//Gen set.
			inputVals .clear();
			targetVals.clear();
			genTrainingSetAbc(inputVals, targetVals);

			//Inputing.
			showVectorVals((char*)": Inputs:", inputVals, fileLogResults);
			net.feedForward(inputVals);

			// Collect the net's actual output results:
			net.getResults(resultVals);
			showVectorVals((char*)"Outputs:", resultVals, fileLogResults);

			// Train the net what the outputs should have been:
			showVectorVals((char*)"Targets:", targetVals, fileLogResults);
			net.backProp(targetVals);

			// Report how well the training is working, average over recent samples:
			fileLogResults << "Net recent average error: "
				<< net.getRecentAverageError() << std::endl;
		}
		if (GetAsyncKeyState(VK_UP))bPause = (bPause) ? 0 : 1;
	}

	std::cin.get();
	return 0;
}