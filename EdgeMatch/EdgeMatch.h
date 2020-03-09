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
#include <Windows.h>


//#define DrawContours
#define PyrNum 3
#define TestCount 10


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
	cv::Point2f contour;

	EdgeModelBaseInfo()
	{
		grad_x = 0;
		grad_y = 0;
		magnitude = 0;
		magnitudeN = 0;
		contour = cv::Point2f(0, 0);
	}
};

struct EdgeModelInfo
{
	GUID ModelID;

	int MinGray;
	int MaxGray;
	int PyrNumber;
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
		PyrNumber = PyrNum;
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
			for (size_t i = 0; i < PyrNumber; i++)
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

		@anchor getPyrImage

		@param srcImage 原始图像
		@param pyrImage 金字塔模型图像
		@param pryNum 金字塔层数
	*/
	void getPyrImage(IN cv::Mat srcImage, IN int pryNum, OUT std::vector<cv::Mat>& pyrImage);

	/** @brief 生成梯度信息

		@anchor createGradInfo

		@param pyrImage 原始图像
		@param pyrSobelX Sobel X方向处理后图像
		@param pyrSobelY Sobel Y方向处理后图像
		@param gradData 梯度信息集合
	*/
	void createGradInfo(IN cv::Mat pyrImage, IN cv::Mat pyrSobelX, IN cv::Mat pyrSobelY, OUT std::vector<EdgeModelBaseInfo>& gradData);

	/** @brief 剔除轮廓点

		@anchor deleteContourPoints

		@param gradData 梯度信息集合
	*/
	void deleteContourPoints(IN OUT std::vector<EdgeModelBaseInfo>& gradData);

	/** @brief 归一化轮廓点

		@anchor normalContourPoints

		@param gradData 梯度信息集合
	*/
	void normalContourPoints(IN OUT std::vector<EdgeModelBaseInfo>& gradData);

	/** @brief 旋转模型信息

		@anchor rotateGradInfo

		@param modelInfo 模型参数
		@param length 模型参数数量
		@param angle 旋转角度
		@param modelGradX 旋转后x方向梯度
		@param modelGradY 旋转后y方向梯度
		@param modelCenterX 旋转后 轮廓点x
		@param modelCenterY 旋转后 轮廓点y
	*/
	void rotateGradInfo(IN EdgeModelBaseInfo*& modelInfo, IN int length, IN float angle,
		OUT float*& modelGradX, OUT float*& modelGradY, OUT float*& modelContourX, OUT float*& modelContourY);

	/** @brief 搜索最优模版信息

		@anchor searchMatchModel

		@param dstSobleX 目标Sobel x方向图像
		@param dstSobleY 目标Sobel y方向图像
		@param center 搜索范围
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
	void searchMatchModel(IN cv::Mat& dstSobleX, IN cv::Mat& dstSobleY, IN int* center,
		IN float minScore, IN float greediness, IN float angle, IN int length,
		IN float*& modelGradX, IN float*& modelGradY, IN float*& modelContourX, IN float*& modelContourY,
		OUT EdgeModelSearchInfo& searchInfo);

	/** @brief 绘制轮廓信息

		@anchor drawContours

		@param img 图像
		@param modelContourX 轮廓点x坐标集合
		@param modelContourY 轮廓点y坐标集合
		@param length 轮廓点数量
		@param searchInfo 搜索结果
	*/
	void drawContours(IN cv::Mat& img, IN float*& modelContourX, IN float*& modelContourY, IN int length, IN EdgeModelSearchInfo& searchInfo);

public:

	/** @brief 基于图片创建边缘模版

		返回值 1-创建成功

		@anchor create_edge_model_path

		@param picPath 图片路径
		@param modelID 模型ID
		@param minGray 最下灰度值
		@param maxGray 最大灰度值
		@param pyrNum 金字塔层数
		@param score 最小相似度
		@param startAngle 搜索起始角度
		@param endAngle 搜索终止角度
		@param stepAngle 搜索步进角度
		@param greediness 贪婪值
	*/
	int create_edge_model_path(IN const char* picPath, IN const char* modelID, IN int minGray, IN int maxGray, IN int pyrNum,
		IN float score, IN float startAngle, IN float endAngle, IN float stepAngle, IN float greediness);

	/** @brief 基于图片查找模版

		返回值 1-查找成功;0-失败

		@anchor find_edge_model_path

		@param picPath 图片路径
		@param modelID 模型ID
	*/
	int find_edge_model_path(IN const char* picPath, IN const char* modelID);

public:

	static EdgeMatch& GetInstance()
	{
		static EdgeMatch m_instance;
		return m_instance;
	}

};
