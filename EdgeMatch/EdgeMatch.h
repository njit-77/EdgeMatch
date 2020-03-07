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

#include <guiddef.h>
#include <vector>


#define PryNum 3

#define CONVERT_STR_2_GUID(str, guid) do\
{\
	sscanf_s(str, "%8x-%4x-%4x-%2x%2x-%2x%2x%2x%2x%2x%2x",\
	&(guid.Data1),&(guid.Data2),&(guid.Data3),\
	&(guid.Data4[0]),&(guid.Data4[1]),&(guid.Data4[2]),&(guid.Data4[3]),\
	&(guid.Data4[4]),&(guid.Data4[5]),&(guid.Data4[6]),&(guid.Data4[7]));\
}while (0);


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

	/*边缘点*/
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
	int PryNumber;
	float Score;
	float StartAngle;
	float EndAngle;
	float StepAngle;
	float Greediness;
	int* EdgeModelBaseInfoSize;
	EdgeModelBaseInfo** EdgeModelBaseInfos;

	EdgeModelInfo()
	{
		ModelID = { 0 };
		MinGray = 0;
		MaxGray = 0;
		PryNumber = PryNum;
		Score = 0.5;
		StartAngle = -45;
		EndAngle = 45;
		StepAngle = 1;
		Greediness = 0.9;
		EdgeModelBaseInfoSize = nullptr;
		EdgeModelBaseInfos = nullptr;
	}

	~EdgeModelInfo()
	{
		DeleteData();
	}

	void DeleteData()
	{
		if (EdgeModelBaseInfoSize != nullptr)
		{
			delete[] EdgeModelBaseInfoSize;
			EdgeModelBaseInfoSize = nullptr;
		}
		if (EdgeModelBaseInfos != nullptr)
		{
			for (size_t i = 0; i < PryNumber; i++)
			{
				delete[] EdgeModelBaseInfos[i];
				EdgeModelBaseInfos[i] = nullptr;
			}
			delete[] EdgeModelBaseInfos;
			EdgeModelBaseInfos = nullptr;
		}
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
			ModelInfo->DeleteData();
			delete ModelInfo;
			ModelInfo = nullptr;
		}
	}

private:
	
	/** @brief 构建金字塔模型

		@anchor pryImage

		@param srcImage 原始图像
		@param pryImage 金字塔模型
		@param pryNum 金字塔层数
	*/
	void pryImage(cv::Mat srcImage, std::vector<cv::Mat>& pryImage, int pryNum);
	
	/** @brief 生成梯度信息

		@anchor createGradInfo

		@param pryCanny canny处理后图像
		@param prySobelX Sobel X方向处理后图像
		@param prySobelY Sobel Y方向处理后图像
		@param gradData 梯度信息集合
	*/
	void createGradInfo(cv::Mat pryCanny, cv::Mat prySobelX, cv::Mat prySobelY, std::vector<EdgeModelBaseInfo>& gradData);

	/** @brief 剔除轮廓点

		@anchor deleteContourPoints

		@param gradData 梯度信息集合
	*/
	void deleteContourPoints(std::vector<EdgeModelBaseInfo>& gradData);

public:

	int create_edge_model_path(const char* picPath, const char* modelID, int minGray, int maxGray, int pryNum,
		float score, float startAngle, float endAngle, float stepAngle, float greediness);
	

public:

	static EdgeMatch& GetInstance()
	{
		static EdgeMatch m_instance;
		return m_instance;
	}

};
