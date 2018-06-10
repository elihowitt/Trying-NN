#include<fstream>
#include<Windows.h>
#include<ctime>

#include"Net.h"

void showVectorVals(char* label, std::vector<double> &v, std::ofstream& f)
{
	f << label << " ";
	for (unsigned i = 0; i < v.size(); ++i) {
		f << v[i] << " ";
	}

	f << std::endl;
}
void showVectorVals(char* label, int t, std::ofstream& f)
{
	f << label << " ";
	
		f << t << " ";
	

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
void genTrainingSetAXORB_BA(std::vector<double>& in, std::vector<double>& out)
{
	double a = rand() % 2;
	double b = rand() % 2;
	in.push_back(a);
	in.push_back(b);
	out.push_back(!(a == b));
	out.push_back(a*b);

}

struct ImageTrainingSet
{
	int tag;
	olcSprite spr;
};
std::vector<olcSprite> vecInts;

int genTrainingSetImage(std::vector<double>& in, std::vector<double>& out)
{
	int t = rand() % 10;
	ImageTrainingSet set = { t, vecInts[t] };
	for (int x = 0; x < set.spr.nWidth; x++)
		for (int y = 0; y < set.spr.nHeight; y++)
			in.push_back((set.spr.GetColour(x, y)) != FG_BLACK);
	for (int i = 0; i < 10; i++)out.push_back(0.0);
	out[t] = 1.0;
	return set.tag;
}


int main()
{
	srand(time(0));

	olcSprite spriteImageZero(L"NN_image_zero.spr");
	olcSprite spriteImageOne(L"NN_image_one.spr");
	olcSprite spriteImageTwo(L"NN_image_two.spr");
	olcSprite spriteImageThree(L"NN_image_three.spr");
	olcSprite spriteImageFour(L"NN_image_four.spr");
	olcSprite spriteImageFive(L"NN_image_five.spr");
	olcSprite spriteImageSix(L"NN_image_six.spr");
	olcSprite spriteImageSeven(L"NN_image_seven.spr");
	olcSprite spriteImageEight(L"NN_image_eight.spr");
	olcSprite spriteImageNine(L"NN_image_nine.spr");

	vecInts.push_back(spriteImageZero);
	vecInts.push_back(spriteImageOne);
	vecInts.push_back(spriteImageTwo);
	vecInts.push_back(spriteImageThree);
	vecInts.push_back(spriteImageFour);
	vecInts.push_back(spriteImageFive);
	vecInts.push_back(spriteImageSix);
	vecInts.push_back(spriteImageSeven);
	vecInts.push_back(spriteImageEight);
	vecInts.push_back(spriteImageNine);



	std::ofstream fileLogResults("logResults.txt");

	std::vector<unsigned> topology;
	topology.push_back(100);
	topology.push_back(16);
	topology.push_back(16);
	topology.push_back(16);
	topology.push_back(16);
	topology.push_back(10);

	Net net(topology, fileLogResults);

	std::vector<double> inputVals;
	std::vector<double> targetVals;
	std::vector<double> resultVals;

	bool bPause = false;
	int i = 0;
	int currentTag;
	while (!GetAsyncKeyState(VK_SPACE) && i++ < 30000)
	{
		if (!bPause)
		{
			//Gen set.
			inputVals.clear();
			targetVals.clear();
			currentTag = genTrainingSetImage(inputVals, targetVals);

			//Inputing.
			showVectorVals((char*)"[Inputs]:", currentTag, fileLogResults);
			net.feedForward(inputVals);

			// Collect the net's actual output results:
			net.getResults(resultVals);
			showVectorVals((char*)"[Outputs]:", resultVals, fileLogResults);

			// Train the net what the outputs should have been:
			showVectorVals((char*)"[Targets]:", targetVals, fileLogResults);
			net.backProp(targetVals);

			// Report how well the training is working, average over recent samples:
			fileLogResults << "Net recent average error: "
				<< net.getRecentAverageError() << std::endl;
		}
		if (GetAsyncKeyState(VK_UP))bPause = (bPause) ? 0 : 1;
	}
	fileLogResults.close();

	std::cin.get();
	return 0;
}