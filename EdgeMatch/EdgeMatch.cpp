// EdgeMatch.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#define _USE_MATH_DEFINES

#include "EdgeMatch.h"
#include <iostream>
#include <math.h>
#include <map>

#pragma comment(lib, "Rpcrt4.lib")

int main()
{
	double time = 0;
	LARGE_INTEGER nFreq;
	LARGE_INTEGER nBeginTime;
	LARGE_INTEGER nEndTime;
	QueryPerformanceFrequency(&nFreq);
	QueryPerformanceCounter(&nBeginTime);
	for (size_t i = 0; i < TestCount; i++)
	{
		EdgeMatch::GetInstance().create_edge_model_path("D:\\Download\\GeoMatch_demo\\Template.jpg",
			"0d4ed8a0-9a35-42cb-ac77-b06c76ed13c8", 15, 30, 3, 0.5, -45, 45, 1, 0.9);
	}
	QueryPerformanceCounter(&nEndTime);
	time = (1000.0 / TestCount) * (nEndTime.QuadPart - nBeginTime.QuadPart) / (double)nFreq.QuadPart;
	std::cout << "创建模型消耗时间(ms)：" << time << std::endl;


	QueryPerformanceCounter(&nBeginTime);
	for (size_t i = 0; i < TestCount; i++)
	{
		EdgeMatch::GetInstance().find_edge_model_path("D:\\Download\\GeoMatch_demo\\Search2.jpg",
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

	cv::Mat magnitudeImg, angleImg;
	cv::cartToPolar(pyrSobelX, pyrSobelY, magnitudeImg, angleImg);

	for (size_t j = 0; j < contoursQuery.size(); j++)
	{
		EdgeModelBaseInfo gradInfo;

		float grad = magnitudeImg.at<float>(contoursQuery[j]);
		float rad = angleImg.at<float>(contoursQuery[j]);

		gradInfo.grad_x = cos(rad);
		gradInfo.grad_y = sin(rad);
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

		int* center = new int[4]
		{
			0, 0,pyrSobelX[ModelInfo->PyrNumber - 1].rows,pyrSobelX[ModelInfo->PyrNumber - 1].cols
		};
		float* modelGradX = nullptr;
		float* modelGradY = nullptr;
		float* modelCenterX = nullptr;
		float* modelCenterY = nullptr;
		{
			//顶层搜索
			float step = ModelInfo->StepAngle * pow(2, ModelInfo->PyrNumber - 1);

			for (float angle = ModelInfo->StartAngle; angle < ModelInfo->EndAngle; angle += step)
			{
				rotateGradInfo(ModelInfo->EdgeModelBaseInfos[ModelInfo->PyrNumber - 1],
					ModelInfo->EdgeModelBaseInfoSize[ModelInfo->PyrNumber - 1],
					angle, modelGradX, modelGradY, modelCenterX, modelCenterY);

				searchMatchModel(pyrSobelX[ModelInfo->PyrNumber - 1],
					pyrSobelY[ModelInfo->PyrNumber - 1],
					center,
					ModelInfo->Score,
					ModelInfo->Greediness,
					angle,
					ModelInfo->EdgeModelBaseInfoSize[ModelInfo->PyrNumber - 1],
					modelGradX, modelGradY, modelCenterX, modelCenterY,
					searchInfo);
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
					rotateGradInfo(ModelInfo->EdgeModelBaseInfos[i],
						ModelInfo->EdgeModelBaseInfoSize[i],
						angle, modelGradX, modelGradY, modelCenterX, modelCenterY);

					searchMatchModel(pyrSobelX[i],
						pyrSobelY[i],
						center,
						ModelInfo->Score,
						ModelInfo->Greediness,
						angle,
						ModelInfo->EdgeModelBaseInfoSize[i],
						modelGradX, modelGradY, modelCenterX, modelCenterY,
						searchInfo);
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

void EdgeMatch::rotateGradInfo(IN EdgeModelBaseInfo*& modelInfo, IN int length, IN float angle,
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

	for (size_t i = 0; i < length; i++)
	{
		modelGradX[i] = modelInfo[i].grad_x * cosA - modelInfo[i].grad_y * sinA;
		modelGradY[i] = modelInfo[i].grad_y * cosA + modelInfo[i].grad_x * sinA;
		modelContourX[i] = modelInfo[i].contour.x * cosA - modelInfo[i].contour.y * sinA;
		modelContourY[i] = modelInfo[i].contour.y * cosA + modelInfo[i].contour.x * sinA;
	}
}

void EdgeMatch::searchMatchModel(IN cv::Mat& dstSobleX, IN cv::Mat& dstSobleY, IN int* center,
	IN float minScore, IN float greediness, IN float angle, IN int length,
	IN float*& modelGradX, IN float*& modelGradY, IN float*& modelContourX, IN float*& modelContourY,
	OUT EdgeModelSearchInfo& searchInfo)
{
	assert(dstSobleX.size() == dstSobleY.size());

	float NormGreediness = (1 - greediness * minScore) / (1 - greediness) / length;
	float NormMinScore = minScore / length;

	cv::Mat magnitudeImg, angleImg;
	cv::cartToPolar(dstSobleX, dstSobleY, magnitudeImg, angleImg);
	float* pAngleImg = nullptr;
	if (angleImg.isContinuous())
	{
		pAngleImg = (float*)angleImg.ptr();
	}

	for (size_t y = center[0]; y < center[2]; y++)
	{
		for (size_t x = center[1]; x < center[3]; x++)
		{
			float partialScore = 0;
			float score = 0;
			int sum = 0;

			for (size_t index = 0; index < length; index++)
			{
				int curX = x + modelContourX[index];
				int curY = y + modelContourY[index];

				if (curX < 0 || curY < 0 || curX > center[3] - 1 || curY > center[2] - 1)
					continue;

				float rad = 0;
				if (pAngleImg != nullptr)
				{
					rad = pAngleImg[curY * angleImg.cols + curX];
				}
				else
				{
					rad = angleImg.at<float>(curY, curX);
				}

				float gx = cos(rad);
				float gy = sin(rad);
				if (gx != 0 || gy != 0)
				{
					partialScore += (gx * modelGradX[index] + gy * modelGradY[index]);

					sum++;
					score = partialScore / sum;
					if (score < (min((minScore - 1) + NormGreediness * sum, NormMinScore * sum)))
						break;
				}
			}

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
	cv::Scalar color = cv::Scalar(255, 0, 0);
	for (size_t i = 0; i < length; i++)
	{
		cv::Point point = cv::Point(searchInfo.CenterX + modelContourX[i], searchInfo.CenterY + modelContourY[i]);
		cv::line(img, point, point, color);
	}
}
