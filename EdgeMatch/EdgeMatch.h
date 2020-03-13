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
#include <iostream>

#define Paraller_Rotate
#define Paraller_Search
//#define DrawContours
#define SSE
//#define SavePNG
#define PyrNum 3
#define TestCount 1
#define SSEStep 8


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
	void rotateGradInfo(IN EdgeModelBaseInfo*& modelInfo, IN uint length, IN float angle,
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
	void searchMatchModel(IN cv::Mat& dstSobleX, IN cv::Mat& dstSobleY, IN uint* center,
		IN float minScore, IN float greediness, IN float angle, IN uint length,
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

#ifdef Paraller_Rotate

class Paraller_RotateGradInfo :public cv::ParallelLoopBody
{
private:
	EdgeModelBaseInfo*& modelInfo;
	uint& length;
	float& angle;
	float*& modelGradX;
	float*& modelGradY;
	float*& modelContourX;
	float*& modelContourY;

public:
	Paraller_RotateGradInfo(IN EdgeModelBaseInfo*& _modelInfo,
		IN uint _length, IN float _angle,
		OUT float*& _modelGradX, OUT float*& _modelGradY,
		OUT float*& _modelContourX, OUT float*& _modelContourY)
		: modelInfo(_modelInfo), length(_length), angle(_angle),
		modelGradX(_modelGradX), modelGradY(_modelGradY),
		modelContourX(_modelContourX), modelContourY(_modelContourY)
	{
		if (modelGradX != nullptr)
		{
			delete[]modelGradX;
			modelGradX = nullptr;
		}
		modelGradX = new float[length];

		if (modelGradY != nullptr)
		{
			delete[]modelGradY;
			modelGradY = nullptr;
		}
		modelGradY = new float[length];

		if (modelContourX != nullptr)
		{
			delete[]modelContourX;
			modelContourX = nullptr;
		}
		modelContourX = new float[length];

		if (modelContourY != nullptr)
		{
			delete[]modelContourY;
			modelContourY = nullptr;
		}
		modelContourY = new float[length];
	}

	virtual void operator()(const cv::Range& r) const
	{
		float rotRad = angle * M_PI / 180;
		float sinA = sin(rotRad);
		float cosA = cos(rotRad);
		for (int i = r.start; i != r.end; i++)
		{
			modelGradX[i] = modelInfo[i].grad_x * cosA - modelInfo[i].grad_y * sinA;
			modelGradY[i] = modelInfo[i].grad_y * cosA + modelInfo[i].grad_x * sinA;
			modelContourX[i] = modelInfo[i].contour.x * cosA - modelInfo[i].contour.y * sinA;
			modelContourY[i] = modelInfo[i].contour.y * cosA + modelInfo[i].contour.x * sinA;
		}
	}
};

#endif


#ifdef Paraller_Search

#include <thread>

class Paraller_SearchMatchModel :public cv::ParallelLoopBody
{
private:
	cv::Mat& dstSobleX;
	cv::Mat& dstSobleY;
	uint*& center;
	float& minScore;
	float& greediness;
	float& angle;
	uint& length;
	float*& modelGradX;
	float*& modelGradY;
	float*& modelContourX;
	float*& modelContourY;
	EdgeModelSearchInfo& searchInfo;


public:
	Paraller_SearchMatchModel(IN cv::Mat& _dstSobleX, IN cv::Mat& _dstSobleY, IN uint* _center,
		IN float _minScore, IN float _greediness, IN float _angle, IN uint _length,
		IN float*& _modelGradX, IN float*& _modelGradY,
		IN float*& _modelContourX, IN float*& _modelContourY,
		OUT EdgeModelSearchInfo& _searchInfo)
		: dstSobleX(_dstSobleX), dstSobleY(_dstSobleY), center(_center),
		minScore(_minScore), greediness(_greediness), angle(_angle), length(_length),
		modelGradX(_modelGradX), modelGradY(_modelGradY),
		modelContourX(_modelContourX), modelContourY(_modelContourY),
		searchInfo(_searchInfo) {}

	virtual void operator()(const cv::Range& r) const
	{
		assert(dstSobleX.size() == dstSobleY.size());

		float NormGreediness = (1 - greediness * minScore) / (1 - greediness) / length;
		float NormMinScore = minScore / length;

		float* pSobleX = nullptr;
		float* pSobleY = nullptr;
		if (dstSobleX.isContinuous() && dstSobleY.isContinuous())
		{
			pSobleX = (float*)dstSobleX.ptr();
			pSobleY = (float*)dstSobleY.ptr();
		}

		for (uint y = r.start; y < r.end; y++)
		{
			for (uint x = center[1]; x < center[3]; x++)
			{
				float partialScore = 0;
				float score = 0;
				uint sum = 0;

#ifdef SSE
				__m256 _x = _mm256_set1_ps(x);
				__m256 _y = _mm256_set1_ps(y);
				__m256i _curX = _mm256_setzero_si256();
				__m256i _curY = _mm256_setzero_si256();
				__m256 _gx = _mm256_set1_ps(0.0f);
				__m256 _gy = _mm256_set1_ps(0.0f);

				int count = length / SSEStep;
				int count2 = length & (SSEStep - 1);
				for (uint index = 0; index < length - count2; index += SSEStep)
				{
					sum += SSEStep;

					_curX = _mm256_cvttps_epi32(_mm256_add_ps(_x, _mm256_load_ps(modelContourX + index)));
					_curY = _mm256_cvttps_epi32(_mm256_add_ps(_y, _mm256_load_ps(modelContourY + index)));

					__m256i l = _mm256_add_epi32(_curX, _mm256_mullo_epi32(_curY, _mm256_set1_epi32(dstSobleX.cols)));
					for (uchar k = 0; k < SSEStep; k++)
					{
						if (_curX.m256i_i32[k] < 0 || _curX.m256i_i32[k] > dstSobleX.cols - 1
							|| _curY.m256i_i32[k] < 0 || _curY.m256i_i32[k] > dstSobleX.rows - 1)
						{
							_gx.m256_f32[k] = 0;
							_gy.m256_f32[k] = 0;
						}
						else
						{
							if (pSobleX != nullptr)
							{
								_gx.m256_f32[k] = pSobleX[l.m256i_i32[k]];
								_gy.m256_f32[k] = pSobleY[l.m256i_i32[k]];
							}
							else
							{
								_gx.m256_f32[k] = dstSobleX.at<float>(_curY.m256i_i32[k], _curX.m256i_i32[k]);
								_gy.m256_f32[k] = dstSobleY.at<float>(_curY.m256i_i32[k], _curX.m256i_i32[k]);
							}
						}
					}

					__m256 _graddot = _mm256_add_ps(_mm256_mul_ps(_gx, _mm256_load_ps(modelGradX + index)), _mm256_mul_ps(_gy, _mm256_load_ps(modelGradY + index)));
					__m256 _grad = _mm256_sqrt_ps(_mm256_add_ps(_mm256_mul_ps(_gx, _gx), _mm256_mul_ps(_gy, _gy)));

					for (uchar k = 0; k < SSEStep; k++)
					{
						if (abs(_gx.m256_f32[k]) > 1e-7 || abs(_gy.m256_f32[k]) > 1e-7)
						{
							partialScore += _graddot.m256_f32[k] / _grad.m256_f32[k];

							score = partialScore / (sum + k - (SSEStep - 1));
							if (score < NormMinScore * (index + 1))
								goto Next;
							//if (score < (min((minScore - 1) + NormGreediness * sum, NormMinScore * sum)))
							//	break;
						}
					}
				}
				for (uint index = length - count2; index < length; index++)
				{
					sum++;
					uint curX = x + modelContourX[index];
					uint curY = y + modelContourY[index];

					if (curX > dstSobleX.cols - 1 || curY > dstSobleX.rows - 1)
						continue;

					float gx = 0;
					float gy = 0;
					if (pSobleX != nullptr)
					{
						uint l = curY * dstSobleX.cols + curX;
						gx = pSobleX[l];
						gy = pSobleY[l];
					}
					else
					{
						gx = dstSobleX.at<float>(curY, curX);
						gy = dstSobleY.at<float>(curY, curX);
					}

					if (abs(gx) > 1e-7 || abs(gy) > 1e-7)
					{
						float grad = sqrt(gx * gx + gy * gy);
						partialScore += ((gx * modelGradX[index] + gy * modelGradY[index])) / grad;

						score = partialScore / sum;
						if (score < NormMinScore * (index + 1))
							break;
						//if (score < (min((minScore - 1) + NormGreediness * sum, NormMinScore * sum)))
						//	break;
					}
				}
			Next:
#else
				for (size_t index = 0; index < length; index++)
				{
					sum++;
					int curX = x + modelContourX[index];
					int curY = y + modelContourY[index];

					if (curX < 0 || curY < 0 || curX > dstSobleX.cols - 1 || curY > dstSobleX.rows - 1)
						continue;

					float gx = 0;
					float gy = 0;
					if (pSobleX != nullptr)
					{
						gx = pSobleX[curY * dstSobleX.cols + curX];
						gy = pSobleY[curY * dstSobleX.cols + curX];
					}
					else
					{
						gx = dstSobleX.at<float>(curY, curX);
						gy = dstSobleY.at<float>(curY, curX);
					}

					if (gx != 0 || gy != 0)
					{
						float grad = sqrt(gx * gx + gy * gy);
						float n_gx = gx / grad;
						float n_gy = gy / grad;
						partialScore += (n_gx * modelGradX[index] + n_gy * modelGradY[index]);

						score = partialScore / sum;
						if (score < NormMinScore * (index + 1))
							break;
						/*if (score < (min((minScore - 1) + NormGreediness * sum, NormMinScore * sum)))
							break;*/
					}
				}
#endif

				if (score > searchInfo.Score)
				{
					static std::mutex mtx;
					std::lock_guard<std::mutex> lock(mtx);
					if (score > searchInfo.Score)
					{
						searchInfo.Score = score;
						searchInfo.Angle = angle;
						searchInfo.CenterX = x;
						searchInfo.CenterY = y;
					}
				}
			}
		}
	}
};

#endif
