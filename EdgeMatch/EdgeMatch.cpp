﻿// EdgeMatch.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#define _USE_MATH_DEFINES

#include "EdgeMatch.h"
#include <math.h>
#include <map>

#pragma comment(lib, "Rpcrt4.lib")

int main()
{
	const char* str = "D:\\Project\\Git\\EdgeMatch\\bin\\x64\\Release\\";
	double time = 0;
	LARGE_INTEGER nFreq;
	LARGE_INTEGER nBeginTime;
	LARGE_INTEGER nEndTime;
	QueryPerformanceFrequency(&nFreq);
	QueryPerformanceCounter(&nBeginTime);
	for (size_t i = 0; i < TestCount; i++)
	{
		EdgeMatch::GetInstance().create_edge_model_path("D:\\Project\\Git\\EdgeMatch\\bin\\x64\\Release\\template.jpg",
			"0d4ed8a0-9a35-42cb-ac77-b06c76ed13c8", 15, 30, 3, 0.5, -45, 45, 1, 0.9);
	}
	QueryPerformanceCounter(&nEndTime);
	time = (1000.0 / TestCount) * (nEndTime.QuadPart - nBeginTime.QuadPart) / (double)nFreq.QuadPart;
	std::cout << "创建模型消耗时间(ms)：" << time << std::endl;


	QueryPerformanceCounter(&nBeginTime);
	for (size_t i = 0; i < TestCount; i++)
	{
		EdgeMatch::GetInstance().find_edge_model_path("D:\\Project\\Git\\EdgeMatch\\bin\\x64\\Release\\search1.jpg",
			"0d4ed8a0-9a35-42cb-ac77-b06c76ed13c8");
	}
	QueryPerformanceCounter(&nEndTime);
	time = (1000.0 / TestCount) * (nEndTime.QuadPart - nBeginTime.QuadPart) / (double)nFreq.QuadPart;
	std::cout << "搜索模型消耗时间(ms)：" << time << std::endl;

	system("pause");
}

int EdgeMatch::create_edge_model_path(IN const char* picPath, IN const char* modelID, IN int minGray, IN int maxGray, IN int pyrNum,
	IN float score, IN float startAngle, IN float endAngle, IN float stepAngle, IN float greediness)
{
	ModelInfo = new EdgeModelInfo;
	UuidFromStringA((RPC_CSTR)modelID, &ModelInfo->ModelID);
	ModelInfo->MinGray = minGray;
	ModelInfo->MaxGray = maxGray;
	ModelInfo->PyrNumber = pyrNum;

	cv::Mat src = cv::imread(picPath, cv::IMREAD_GRAYSCALE);
	cv::Mat sobelX, sobleY;
	cv::Sobel(src, sobelX, CV_32FC1, 1, 0, 3);
	cv::Sobel(src, sobleY, CV_32FC1, 0, 1, 3);
	std::vector<cv::Mat> pyrImage;
	std::vector<cv::Mat> pyrSobelX;
	std::vector<cv::Mat> pyrSobelY;
	getPyrImage(src, pyrNum, pyrImage);
	getPyrImage(sobelX, pyrNum, pyrSobelX);
	getPyrImage(sobleY, pyrNum, pyrSobelY);

	ModelInfo->EdgeModelBaseInfos = new EdgeModelBaseInfo * [pyrNum];
	ModelInfo->EdgeModelBaseInfoSize = new int[pyrNum];
	for (size_t i = 0; i < pyrNum; i++)
	{
		std::vector<EdgeModelBaseInfo> gradData;
		createGradInfo(pyrImage[i], pyrSobelX[i], pyrSobelY[i], gradData);
		deleteContourPoints(gradData);
		normalContourPoints(gradData);
		{
			ModelInfo->EdgeModelBaseInfoSize[i] = gradData.size();
			ModelInfo->EdgeModelBaseInfos[i] = new EdgeModelBaseInfo[gradData.size()];
			for (size_t j = 0; j < gradData.size(); j++)
			{
				ModelInfo->EdgeModelBaseInfos[i][j] = gradData[j];
			}
		}
		gradData.clear();
	}

	return 1;
}

void EdgeMatch::getPyrImage(IN cv::Mat srcImage, IN int pryNum, OUT std::vector<cv::Mat>& pyrImage)
{
	pyrImage.push_back(srcImage);

	for (size_t i = 1; i < pryNum; i++)
	{
		cv::pyrDown(srcImage, srcImage);
		pyrImage.push_back(srcImage);
	}
}

void EdgeMatch::createGradInfo(IN cv::Mat pyrImage, IN cv::Mat pyrSobelX, IN cv::Mat pyrSobelY, OUT std::vector<EdgeModelBaseInfo>& gradData)
{
	cv::Mat edge;
	cv::Canny(pyrImage, edge, ModelInfo->MinGray, ModelInfo->MaxGray);

	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(edge, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

	std::vector<cv::Point> contoursQuery;
	for (size_t j = 0; j < contours.size(); j++)
	{
		contoursQuery.insert(contoursQuery.end(), contours[j].begin(), contours[j].end());
	}

	float* pSobleX = nullptr;
	float* pSobleY = nullptr;
	if (pyrSobelX.isContinuous() && pyrSobelY.isContinuous())
	{
		pSobleX = (float*)pyrSobelX.ptr();
		pSobleY = (float*)pyrSobelY.ptr();
	}
	for (size_t j = 0; j < contoursQuery.size(); j++)
	{
		EdgeModelBaseInfo gradInfo;

		float gx = 0;
		float gy = 0;
		if (pSobleX != nullptr)
		{
			int index = contoursQuery[j].y * pyrSobelX.cols + contoursQuery[j].x;
			gx = pSobleX[index];
			gy = pSobleY[index];
		}
		else
		{
			gx = pyrSobelX.at<float>(contoursQuery[j]);
			gy = pyrSobelY.at<float>(contoursQuery[j]);
		}

		float grad = sqrt(gx * gx + gy * gy);
		gradInfo.grad_x = gx / grad;
		gradInfo.grad_y = gy / grad;
		gradInfo.magnitude = grad;
		if (cv::abs(grad) > 1e-7)
			gradInfo.magnitudeN = 1 / grad;
		gradInfo.contour = contoursQuery[j];

		gradData.push_back(gradInfo);
	}
}

void EdgeMatch::deleteContourPoints(IN OUT std::vector<EdgeModelBaseInfo>& gradData)
{
	if (gradData.size() < 50)return;

	std::vector<EdgeModelBaseInfo> data;

	int cullCount = 3;
	for (size_t i = 0; i < gradData.size(); i += 3)
	{
		std::map<float, int> culls;
		for (size_t j = 0; j < cullCount; j++)
		{
			if (i + j < gradData.size())
			{
				culls.insert(std::pair<float, int>(gradData[i + j].magnitude, j));
			}
		}

		data.push_back(gradData[i + culls.rbegin()->second]);
	}

	gradData.clear();
	for (auto& it : data)
	{
		gradData.push_back(it);
	}
	data.clear();
}

void EdgeMatch::normalContourPoints(IN OUT std::vector<EdgeModelBaseInfo>& gradData)
{
	if (gradData.size() < 0)return;
	float x = 0, y = 0;
	for (auto& it : gradData)
	{
		x += it.contour.x;
		y += it.contour.y;
	}
	x /= gradData.size();
	y /= gradData.size();
	for (auto& it : gradData)
	{
		it.contour.x -= x;
		it.contour.y -= y;
	}
}

int EdgeMatch::find_edge_model_path(IN const char* picPath, IN const char* modelID)
{
	GUID id = { 0 };
	UuidFromStringA((RPC_CSTR)modelID, &id);
	if (ModelInfo != nullptr && ModelInfo->ModelID == id)
	{
		cv::Mat src = cv::imread(picPath, cv::IMREAD_GRAYSCALE);
		cv::Mat sobelX, sobleY;
		cv::Sobel(src, sobelX, CV_32FC1, 1, 0, 3);
		cv::Sobel(src, sobleY, CV_32FC1, 0, 1, 3);
		std::vector<cv::Mat> pyrImage;
		std::vector<cv::Mat> pyrSobelX;
		std::vector<cv::Mat> pyrSobelY;
		getPyrImage(src, ModelInfo->PyrNumber, pyrImage);
		getPyrImage(sobelX, ModelInfo->PyrNumber, pyrSobelX);
		getPyrImage(sobleY, ModelInfo->PyrNumber, pyrSobelY);

		EdgeModelSearchInfo searchInfo;

		if (abs(ModelInfo->StepAngle) < 1e-7)ModelInfo->StepAngle = 1;

		uint center[] = { 0, 0, pyrSobelX[ModelInfo->PyrNumber - 1].rows, pyrSobelX[ModelInfo->PyrNumber - 1].cols };
		float* modelGradX = nullptr;
		float* modelGradY = nullptr;
		float* modelCenterX = nullptr;
		float* modelCenterY = nullptr;
		{
			//顶层搜索
			float step = ModelInfo->StepAngle * pow(2, ModelInfo->PyrNumber - 1);
			for (float angle = ModelInfo->StartAngle; angle < ModelInfo->EndAngle; angle += step)
			{
#ifdef Paraller_Rotate
				parallel_for_(cv::Range(0, ModelInfo->EdgeModelBaseInfoSize[ModelInfo->PyrNumber - 1]),
					Paraller_RotateGradInfo(ModelInfo->EdgeModelBaseInfos[ModelInfo->PyrNumber - 1],
						ModelInfo->EdgeModelBaseInfoSize[ModelInfo->PyrNumber - 1],
						angle, modelGradX, modelGradY, modelCenterX, modelCenterY));
#else
				rotateGradInfo(ModelInfo->EdgeModelBaseInfos[ModelInfo->PyrNumber - 1],
					ModelInfo->EdgeModelBaseInfoSize[ModelInfo->PyrNumber - 1],
					angle, modelGradX, modelGradY, modelCenterX, modelCenterY);
#endif

#ifdef Paraller_Search
				parallel_for_(cv::Range(center[0], center[2]),
					Paraller_SearchMatchModel(pyrSobelX[ModelInfo->PyrNumber - 1],
						pyrSobelY[ModelInfo->PyrNumber - 1],
						center,
						ModelInfo->Score,
						ModelInfo->Greediness,
						angle,
						ModelInfo->EdgeModelBaseInfoSize[ModelInfo->PyrNumber - 1],
						modelGradX, modelGradY, modelCenterX, modelCenterY,
						searchInfo));
#else
				searchMatchModel(pyrSobelX[ModelInfo->PyrNumber - 1],
					pyrSobelY[ModelInfo->PyrNumber - 1],
					center,
					ModelInfo->Score,
					ModelInfo->Greediness,
					angle,
					ModelInfo->EdgeModelBaseInfoSize[ModelInfo->PyrNumber - 1],
					modelGradX, modelGradY, modelCenterX, modelCenterY,
					searchInfo);
#endif
			}
#ifdef DrawContours
			rotateGradInfo(ModelInfo->EdgeModelBaseInfos[ModelInfo->PyrNumber - 1],
				ModelInfo->EdgeModelBaseInfoSize[ModelInfo->PyrNumber - 1],
				searchInfo.Angle, modelGradX, modelGradY, modelCenterX, modelCenterY);
			cv::Mat img = pyrImage[ModelInfo->PyrNumber - 1];
			drawContours(img,
				modelCenterX, modelCenterY,
				ModelInfo->EdgeModelBaseInfoSize[ModelInfo->PyrNumber - 1],
				searchInfo);
#endif
		}
		{
			for (int i = ModelInfo->PyrNumber - 2; i > -1; i--)
			{
				center[1] = 2 * searchInfo.CenterX - 10;
				center[3] = 2 * searchInfo.CenterX + 10;
				center[0] = 2 * searchInfo.CenterY - 10;
				center[2] = 2 * searchInfo.CenterY + 10;

				searchInfo.Score = 0;
				float step = ModelInfo->StepAngle * pow(2, i);
				float startAngle = searchInfo.Angle - step;
				float endAngle = searchInfo.Angle + step;
				for (float angle = startAngle; angle < endAngle; angle += step)
				{
#ifdef Paraller_Rotate
					parallel_for_(cv::Range(0, ModelInfo->EdgeModelBaseInfoSize[i]),
						Paraller_RotateGradInfo(ModelInfo->EdgeModelBaseInfos[i],
							ModelInfo->EdgeModelBaseInfoSize[i],
							angle, modelGradX, modelGradY, modelCenterX, modelCenterY));
#else
					rotateGradInfo(ModelInfo->EdgeModelBaseInfos[i],
						ModelInfo->EdgeModelBaseInfoSize[i],
						angle, modelGradX, modelGradY, modelCenterX, modelCenterY);
#endif

#ifdef Paraller_Search
					parallel_for_(cv::Range(center[0], center[2]),
						Paraller_SearchMatchModel(pyrSobelX[i],
							pyrSobelY[i],
							center,
							ModelInfo->Score,
							ModelInfo->Greediness,
							angle,
							ModelInfo->EdgeModelBaseInfoSize[i],
							modelGradX, modelGradY, modelCenterX, modelCenterY,
							searchInfo));
#else
					searchMatchModel(pyrSobelX[i],
						pyrSobelY[i],
						center,
						ModelInfo->Score,
						ModelInfo->Greediness,
						angle,
						ModelInfo->EdgeModelBaseInfoSize[i],
						modelGradX, modelGradY, modelCenterX, modelCenterY,
						searchInfo);
#endif
				}
			}
#ifdef DrawContours
			rotateGradInfo(ModelInfo->EdgeModelBaseInfos[0],
				ModelInfo->EdgeModelBaseInfoSize[0],
				searchInfo.Angle, modelGradX, modelGradY, modelCenterX, modelCenterY);
			cv::Mat img = pyrImage[0];
			drawContours(img,
				modelCenterX, modelCenterY,
				ModelInfo->EdgeModelBaseInfoSize[0],
				searchInfo);
			char buff[MAXBYTE];
			SYSTEMTIME sys;
			GetLocalTime(&sys);
			sprintf_s(buff, "%04d-%02d-%02d-%02d-%02d-%02d-%03d.png",
				sys.wYear, sys.wMonth, sys.wDay, sys.wHour, sys.wMinute, sys.wSecond, sys.wMilliseconds);
			cv::imwrite(buff, img);
#endif
		}
		if (modelGradX != nullptr)
		{
			delete[]modelGradX;
			modelGradX = nullptr;
		}
		if (modelGradY != nullptr)
		{
			delete[]modelGradY;
			modelGradY = nullptr;
		}
		if (modelCenterX != nullptr)
		{
			delete[]modelCenterX;
			modelCenterX = nullptr;
		}
		if (modelCenterY != nullptr)
		{
			delete[]modelCenterY;
			modelCenterY = nullptr;
		}
	}
	else
	{
		return 0;
	}
	return 1;
}

void EdgeMatch::rotateGradInfo(IN EdgeModelBaseInfo*& modelInfo, IN uint length, IN float angle,
	OUT float*& modelGradX, OUT float*& modelGradY, OUT float*& modelContourX, OUT float*& modelContourY)
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

	float rotRad = angle * M_PI / 180;
	float sinA = sin(rotRad);
	float cosA = cos(rotRad);

#ifdef SSE
	int count = length / 4;
	int count2 = length & 3;

	__m128 _sinA = _mm_set_ps1(sinA);
	__m128 _cosA = _mm_set_ps1(cosA);

	__m128 grad_x = _mm_set_ps1(0.0f);
	__m128 grad_y = _mm_set_ps1(0.0f);
	__m128 contour_x = _mm_set_ps1(0.0f);
	__m128 contour_y = _mm_set_ps1(0.0f);

	__m128 _grad_x = _mm_set_ps1(0.0f);
	__m128 _grad_y = _mm_set_ps1(0.0f);
	__m128 _contour_x = _mm_set_ps1(0.0f);
	__m128 _contour_y = _mm_set_ps1(0.0f);

	for (uint i = 0; i < length - count2; i += 4)
	{
		grad_x = _mm_set_ps(modelInfo[i + 3].grad_x,
			modelInfo[i + 2].grad_x, modelInfo[i + 1].grad_x, modelInfo[i].grad_x);

		grad_y = _mm_set_ps(modelInfo[i + 3].grad_y,
			modelInfo[i + 2].grad_y, modelInfo[i + 1].grad_y, modelInfo[i].grad_y);

		contour_x = _mm_set_ps(modelInfo[i + 3].contour.x,
			modelInfo[i + 2].contour.x, modelInfo[i + 1].contour.x, modelInfo[i].contour.x);

		contour_y = _mm_set_ps(modelInfo[i + 3].contour.y,
			modelInfo[i + 2].contour.y, modelInfo[i + 1].contour.y, modelInfo[i].contour.y);

		_grad_x = _mm_sub_ps(_mm_mul_ps(grad_x, _cosA), _mm_mul_ps(grad_y, _sinA));
		_grad_y = _mm_add_ps(_mm_mul_ps(grad_y, _cosA), _mm_mul_ps(grad_x, _sinA));
		_contour_x = _mm_sub_ps(_mm_mul_ps(contour_x, _cosA), _mm_mul_ps(contour_y, _sinA));
		_contour_y = _mm_add_ps(_mm_mul_ps(contour_y, _cosA), _mm_mul_ps(contour_x, _sinA));

		modelGradX[i] = _grad_x.m128_f32[0];
		modelGradX[i + 1] = _grad_x.m128_f32[1];
		modelGradX[i + 2] = _grad_x.m128_f32[2];
		modelGradX[i + 3] = _grad_x.m128_f32[3];

		modelGradY[i] = _grad_y.m128_f32[0];
		modelGradY[i + 1] = _grad_y.m128_f32[1];
		modelGradY[i + 2] = _grad_y.m128_f32[2];
		modelGradY[i + 3] = _grad_y.m128_f32[3];

		modelContourX[i] = _contour_x.m128_f32[0];
		modelContourX[i + 1] = _contour_x.m128_f32[1];
		modelContourX[i + 2] = _contour_x.m128_f32[2];
		modelContourX[i + 3] = _contour_x.m128_f32[3];

		modelContourY[i] = _contour_y.m128_f32[0];
		modelContourY[i + 1] = _contour_y.m128_f32[1];
		modelContourY[i + 2] = _contour_y.m128_f32[2];
		modelContourY[i + 3] = _contour_y.m128_f32[3];
	}
	for (uint i = length - count2; i < length; i++)
	{
		modelGradX[i] = modelInfo[i].grad_x * cosA - modelInfo[i].grad_y * sinA;
		modelGradY[i] = modelInfo[i].grad_y * cosA + modelInfo[i].grad_x * sinA;
		modelContourX[i] = modelInfo[i].contour.x * cosA - modelInfo[i].contour.y * sinA;
		modelContourY[i] = modelInfo[i].contour.y * cosA + modelInfo[i].contour.x * sinA;
	}
#else
	for (uint i = 0; i < length; i++)
	{
		modelGradX[i] = modelInfo[i].grad_x * cosA - modelInfo[i].grad_y * sinA;
		modelGradY[i] = modelInfo[i].grad_y * cosA + modelInfo[i].grad_x * sinA;
		modelContourX[i] = modelInfo[i].contour.x * cosA - modelInfo[i].contour.y * sinA;
		modelContourY[i] = modelInfo[i].contour.y * cosA + modelInfo[i].contour.x * sinA;
	}
#endif
}

void EdgeMatch::searchMatchModel(IN cv::Mat& dstSobleX, IN cv::Mat& dstSobleY, IN uint* center,
	IN float minScore, IN float greediness, IN float angle, IN uint length,
	IN float*& modelGradX, IN float*& modelGradY, IN float*& modelContourX, IN float*& modelContourY,
	OUT EdgeModelSearchInfo& searchInfo)
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

	for (uint y = center[0]; y < center[2]; y++)
	{
		for (uint x = center[1]; x < center[3]; x++)
		{
			float partialScore = 0;
			float score = 0;
			uint sum = 0;

#ifdef SSE
			int count = length / 4;
			int count2 = length & 3;

			__m128 _x = _mm_set_ps1(x);
			__m128 _y = _mm_set_ps1(y);
			__m128i _curX = _mm_setzero_si128();
			__m128i _curY = _mm_setzero_si128();
			__m128 _gx = _mm_set_ps1(0.0f);
			__m128 _gy = _mm_set_ps1(0.0f);
			for (uint index = 0; index < length - count2; index += 4)
			{
				sum += 4;

				_curX = _mm_cvttps_epi32(_mm_add_ps(_x, _mm_load_ps(modelContourX + index)));
				_curY = _mm_cvttps_epi32(_mm_add_ps(_y, _mm_load_ps(modelContourY + index)));

				if (pSobleX != nullptr)
				{
					for (uchar k = 0; k < 4; k++)
					{
						if (_curX.m128i_i32[k] < 0 || _curX.m128i_i32[k] > dstSobleX.cols - 1
							|| _curY.m128i_i32[k] < 0 || _curY.m128i_i32[k] > dstSobleX.rows - 1)
						{
							_gx.m128_f32[k] = 0;
							_gy.m128_f32[k] = 0;
						}
						else
						{
							uint l = _curY.m128i_i32[k] * dstSobleX.cols + _curX.m128i_i32[k];
							_gx.m128_f32[k] = pSobleX[l];
							_gy.m128_f32[k] = pSobleY[l];
						}
					}
				}
				else
				{
					for (uchar k = 0; k < 4; k++)
					{
						if (_curX.m128i_i32[k] < 0 || _curX.m128i_i32[k] > dstSobleX.cols - 1
							|| _curY.m128i_i32[k] < 0 || _curY.m128i_i32[k] > dstSobleX.rows - 1)
						{
							_gx.m128_f32[k] = 0;
							_gy.m128_f32[k] = 0;
						}
						else
						{
							_gx.m128_f32[k] = dstSobleX.at<float>(_curY.m128i_i32[k], _curX.m128i_i32[k]);
							_gy.m128_f32[k] = dstSobleY.at<float>(_curY.m128i_i32[k], _curX.m128i_i32[k]);
						}
					}
				}

				__m128 _graddot = _mm_add_ps(_mm_mul_ps(_gx, _mm_load_ps(modelGradX + index)), _mm_mul_ps(_gy, _mm_load_ps(modelGradY + index)));
				__m128 _grad = _mm_sqrt_ps(_mm_add_ps(_mm_mul_ps(_gx, _gx), _mm_mul_ps(_gy, _gy)));

				for (uchar k = 0; k < 4; k++)
				{
					if (abs(_gx.m128_f32[k]) > 1e-7 || abs(_gy.m128_f32[k]) > 1e-7)
					{
						partialScore += _graddot.m128_f32[k] / _grad.m128_f32[k];

						score = partialScore / (sum + k - 3);
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
			for (uint index = 0; index < length; index++)
			{
				sum++;
				int curX = x + modelContourX[index];
				int curY = y + modelContourY[index];

				if (curX < 0 || curX > dstSobleX.cols - 1 || curY < 0 || curY > dstSobleX.rows - 1)
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
#endif
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

void EdgeMatch::drawContours(IN cv::Mat& img, IN float*& modelContourX, IN float*& modelContourY, IN int length, IN EdgeModelSearchInfo& searchInfo)
{
	cv::cvtColor(img, img, cv::COLOR_GRAY2RGB);
	cv::Scalar color = cv::Scalar(0, 0, 255);
	for (size_t i = 0; i < length; i++)
	{
		cv::Point point = cv::Point(searchInfo.CenterX + modelContourX[i], searchInfo.CenterY + modelContourY[i]);
		cv::line(img, point, point, color, 3, cv::LINE_AA);
	}
}
