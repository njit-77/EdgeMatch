#pragma once

#ifdef _WIN64

#if _DEBUG
#pragma comment (lib,"opencv_world420d.lib")
#else
#pragma comment (lib,"opencv_world420.lib")
#endif

#else
#endif

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <objbase.h>
#include <vector>


#define PryNumber 3

struct EdgeModelBaseInfo
{
	/*x方向梯度*/
	float grad_x;

	/*y方向梯度*/
	float grad_y;

	/*√(grad_x²+grad_y²)*/
	float magnitude;

	/*1/magnitude*/
	float magnitudeN;

	cv::Point contour;

	EdgeModelBaseInfo()
	{
		grad_x = 0;
		grad_y = 0;
		magnitude = 0;
		magnitudeN = 0;
		contour = cv::Point(0, 0);
	}
};

struct EdgeModelInfo
{
	GUID ModelID;

	int MinGray;
	int MaxGray;
	int PryNum;
	std::vector<std::vector<EdgeModelBaseInfo>> EdgeModelBaseInfos;

	EdgeModelInfo()
	{
		MinGray = 0;
		MaxGray = 0;
		PryNum = PryNumber;
	}
};

class EdgeMatch
{
private:
	EdgeModelInfo* ModelInfo;

private:
	EdgeMatch()
	{
		ModelInfo = nullptr;
	}

	~EdgeMatch()
	{
		if (ModelInfo != nullptr)
		{
			for (size_t i = 0; i < ModelInfo->EdgeModelBaseInfos.size(); i++)
			{
				for (size_t j = 0; j < ModelInfo->EdgeModelBaseInfos[i].size(); j++)
				{
					ModelInfo->EdgeModelBaseInfos[i].clear();
				}
				ModelInfo->EdgeModelBaseInfos.clear();
			}
			delete ModelInfo;
			ModelInfo = nullptr;
		}
	}

public:
	int create_edge_model_path(const char* picPath, const char* modelID, int minGray, int maxGray, int pryNum);
	void pryImage(cv::Mat srcImage, std::vector<cv::Mat>& pryImage, int pryNum);
	void createGrad(cv::Mat pryCanny, cv::Mat prySobelX, cv::Mat prySobelY, std::vector<EdgeModelBaseInfo>& data);

public:

	static EdgeMatch& GetInstance()
	{
		static EdgeMatch m_instance;
		return m_instance;
	}
};
