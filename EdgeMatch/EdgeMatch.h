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
#include <opencv2/highgui.hpp>

#include <guiddef.h>
#include <vector>


#define DrawContours
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

struct EdgeModelSearchInfo
{
	float Score;
	float Angle;
	float CenterX;
	float CenterY;

	EdgeModelSearchInfo()
	{
		Score = 0;
		Angle = 0;
		CenterX = 0;
		CenterY = 0;
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
		@param cannyImage canny 金字塔模型
		@param sobelXImage soble x方向 金字塔模型
		@param sobelYImage soble y方向 金字塔模型
	*/
	void getPryImage(cv::Mat srcImage, std::vector<cv::Mat>& cannyImage, std::vector<cv::Mat>& sobelXImage, std::vector<cv::Mat>& sobelYImage);

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

	/** @brief 归一化轮廓点

		@anchor normalContourPoints

		@param gradData 梯度信息集合
	*/
	void normalContourPoints(std::vector<EdgeModelBaseInfo>& gradData);

	/** @brief 旋转模型信息

		@anchor rotateGradInfo

		@param angle 旋转角度
		@param modelInfo 模型参数
		@param length 模型参数数量
		@param modelGradX 旋转后x方向梯度
		@param modelGradY 旋转后y方向梯度
		@param modelCenterX 旋转后 轮廓点x
		@param modelCenterY 旋转后 轮廓点y
	*/
	void rotateGradInfo(float angle, EdgeModelBaseInfo*& modelInfo, int length, float*& modelGradX, float*& modelGradY, float*& modelContourX, float*& modelContourY);

	/** @brief 搜索最优模版信息

		@anchor searchMatchModel

		@param dstSobleX 目标Sobel x方向图像
		@param dstSobleY 目标Sobel y方向图像
		@param minScore 最小相似度
		@param greediness 贪婪度
		@param angle 模版旋转角度
		@param length 模版总数量
		@param modelGradX 模版x方向梯度
		@param modelGradY 模版y方向梯度
		@param modelContourX 模版轮廓点x坐标
		@param modelContourY 模版轮廓点y坐标
		@param searchInfo 搜索结果
	*/
	void searchMatchModel(cv::Mat& dstSobleX, cv::Mat& dstSobleY,
		float minScore, float greediness, float angle, int length,
		float*& modelGradX, float*& modelGradY, float*& modelContourX, float*& modelContourY,
		EdgeModelSearchInfo& searchInfo);

	void drawContours(cv::Mat& dst, int length, float*& modelContourX, float*& modelContourY, EdgeModelSearchInfo& searchInfo);

public:

	/** @brief 基于图片创建边缘模版

		返回值 1-创建成功

		@anchor create_edge_model_path

		@param picPath 图片路径
		@param modelID 模型ID
		@param minGray 最下灰度值
		@param maxGray 最大灰度值
		@param pryNum 金字塔层数
		@param score 最小相似度
		@param startAngle 搜索起始角度
		@param endAngle 搜索终止角度
		@param stepAngle 搜索步进角度
		@param greediness 贪婪值
	*/
	int create_edge_model_path(const char* picPath, const char* modelID, int minGray, int maxGray, int pryNum,
		float score, float startAngle, float endAngle, float stepAngle, float greediness);

	/** @brief 基于图片查找模版

		返回值 1-查找成功;0-失败

		@anchor find_edge_model_path

		@param picPath 图片路径
		@param modelID 模型ID
	*/
	int find_edge_model_path(const char* picPath, const char* modelID);

public:

	static EdgeMatch& GetInstance()
	{
		static EdgeMatch m_instance;
		return m_instance;
	}

};
