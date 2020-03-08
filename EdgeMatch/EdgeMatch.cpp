// EdgeMatch.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include "EdgeMatch.h"
#include <iostream>
#include <math.h>
#include <map>

int main()
{
	EdgeMatch::GetInstance().create_edge_model_path("D:\\Download\\GeoMatch_demo\\Template.jpg",
		"0d4ed8a0-9a35-42cb-ac77-b06c76ed13c8", 15, 30, 3, 0.5, -45, 45, 1, 0.9);

	EdgeMatch::GetInstance().find_edge_model_path("D:\\Download\\GeoMatch_demo\\Search2.jpg",
		"0d4ed8a0-9a35-42cb-ac77-b06c76ed13c8");

	system("pause");
}

int EdgeMatch::create_edge_model_path(const char* picPath, const char* modelID, int minGray, int maxGray, int pryNum,
	float score, float startAngle, float endAngle, float stepAngle, float greediness)
{
	ModelInfo = new EdgeModelInfo;
	CONVERT_STR_2_GUID(modelID, ModelInfo->ModelID);
	ModelInfo->MinGray = minGray;
	ModelInfo->MaxGray = maxGray;
	ModelInfo->PryNumber = pryNum;

	cv::Mat src = cv::imread(picPath, cv::IMREAD_GRAYSCALE);
	std::vector<cv::Mat> pryCanny;
	std::vector<cv::Mat> prySobelX;
	std::vector<cv::Mat> prySobelY;
	getPryImage(src, pryCanny, prySobelX, prySobelY);

	ModelInfo->EdgeModelBaseInfos = new EdgeModelBaseInfo * [ModelInfo->PryNumber];
	ModelInfo->EdgeModelBaseInfoSize = new int[ModelInfo->PryNumber];
	for (size_t i = 0; i < ModelInfo->PryNumber; i++)
	{
		std::vector<EdgeModelBaseInfo> gradData;
		createGradInfo(pryCanny[i], prySobelX[i], prySobelY[i], gradData);
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

void EdgeMatch::getPryImage(cv::Mat srcImage, std::vector<cv::Mat>& cannyImage, std::vector<cv::Mat>& sobelXImage, std::vector<cv::Mat>& sobelYImage)
{
	cv::Mat cannyImg;
	cv::Canny(srcImage, cannyImg, ModelInfo->MinGray, ModelInfo->MaxGray);

	cv::Mat sobelX, sobleY;
	cv::Sobel(cannyImg, sobelX, CV_32FC1, 1, 0, 3);
	cv::Sobel(cannyImg, sobleY, CV_32FC1, 0, 1, 3);

	pryImage(cannyImg, cannyImage, ModelInfo->PryNumber);
	pryImage(sobelX, sobelXImage, ModelInfo->PryNumber);
	pryImage(sobleY, sobelYImage, ModelInfo->PryNumber);
}

void EdgeMatch::pryImage(cv::Mat srcImage, std::vector<cv::Mat>& pryImage, int pryNum)
{
	pryImage.push_back(srcImage);

	for (size_t i = 1; i < pryNum; i++)
	{
		cv::pyrDown(srcImage, srcImage);
		pryImage.push_back(srcImage);
	}
}

void EdgeMatch::createGradInfo(cv::Mat pryCanny, cv::Mat prySobelX, cv::Mat prySobelY, std::vector<EdgeModelBaseInfo>& gradData)
{
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(pryCanny, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

	std::vector<cv::Point> contoursQuery;
	for (size_t j = 0; j < contours.size(); j++)
	{
		contoursQuery.insert(contoursQuery.begin(), contours[j].begin(), contours[j].end());
	}

#ifdef DrawContours
	cv::Mat dst;
	cv::cvtColor(pryCanny, dst, cv::COLOR_GRAY2RGB);

	cv::Scalar color = cv::Scalar(255, 0, 0);
	for (size_t i = 0; i < contoursQuery.size(); i++)
	{
		cv::line(dst, contoursQuery[i], contoursQuery[i], color);
	}
#endif

	cv::Mat magnitudeImg, angleImg;
	cv::cartToPolar(prySobelX, prySobelY, magnitudeImg, angleImg);

	for (size_t j = 0; j < contoursQuery.size(); j++)
	{
		EdgeModelBaseInfo gradInfo;

		if (true)
		{
			float grad = magnitudeImg.at<float>(contoursQuery[j]);
			float angle = angleImg.at<float>(contoursQuery[j]);

			gradInfo.grad_x = cos(angle);
			gradInfo.grad_y = sin(angle);
			gradInfo.magnitude = grad;
			if (cv::abs(grad) > 1e-7)
				gradInfo.magnitudeN = 1 / grad;
			gradInfo.contour = contoursQuery[j];
		}
		else
		{
			float gx = prySobelX.at<float>(contoursQuery[j]);
			float gy = prySobelY.at<float>(contoursQuery[j]);
			float grad = sqrt(gx * gx + gy * gy);

			gradInfo.grad_x = gx / grad;
			gradInfo.grad_y = gy / grad;
			gradInfo.magnitude = grad;
			if (cv::abs(grad) > 1e-7)
				gradInfo.magnitudeN = 1 / grad;
			gradInfo.contour = contoursQuery[j];
		}

		gradData.push_back(gradInfo);
	}
}

void EdgeMatch::deleteContourPoints(std::vector<EdgeModelBaseInfo>& gradData)
{
	if (gradData.size() < 50)return;

	std::vector<EdgeModelBaseInfo> data;

	int cullCount = 3;
	for (size_t i = 0; i < gradData.size(); i += 3)
	{
		std::map<float, int> culls;
		for (size_t j = 0; j < cullCount, i + j < gradData.size(); j++)
		{
			culls.insert(std::pair<float, int>(gradData[i + j].magnitude, j));
		}

		auto it = culls.end();
		it--;
		data.push_back(gradData[i + it->second]);
	}

	gradData.clear();
	for (auto& it : data)
	{
		gradData.push_back(it);
	}
	data.clear();
}

void EdgeMatch::normalContourPoints(std::vector<EdgeModelBaseInfo>& gradData)
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

int EdgeMatch::find_edge_model_path(const char* picPath, const char* modelID)
{
	GUID id = { 0 };
	CONVERT_STR_2_GUID(modelID, id);
	if (ModelInfo != nullptr && ModelInfo->ModelID == id)
	{
		cv::Mat src = cv::imread(picPath, cv::IMREAD_GRAYSCALE);
		std::vector<cv::Mat> prySrc;
		std::vector<cv::Mat> pryCanny;
		std::vector<cv::Mat> prySobelX;
		std::vector<cv::Mat> prySobelY;
		pryImage(src, prySrc, ModelInfo->PryNumber);
		getPryImage(src, pryCanny, prySobelX, prySobelY);

		//std::vector < std::vector<EdgeModelBaseInfo>>gradDatas;
		//for (size_t i = 0; i < ModelInfo->PryNumber; i++)
		//{
		//	std::vector<EdgeModelBaseInfo> gradData;
		//	createGradInfo(pryCanny[i], prySobelX[i], prySobelY[i], gradData);
		//	deleteContourPoints(gradData);
		//	gradDatas.push_back(gradData);
		//	gradData.clear();
		//}

		EdgeModelSearchInfo searchInfo;

		float* modelGradX = nullptr;
		float* modelGradY = nullptr;
		float* modelCenterX = nullptr;
		float* modelCenterY = nullptr;
		{
			//顶层搜索
			float step = ModelInfo->StepAngle * pow(ModelInfo->PryNumber - 1, 2);

			for (float angle = ModelInfo->StartAngle; angle < ModelInfo->EndAngle; angle += step)
			{
				rotateGradInfo(angle,
					ModelInfo->EdgeModelBaseInfos[ModelInfo->PryNumber - 1],
					ModelInfo->EdgeModelBaseInfoSize[ModelInfo->PryNumber - 1],
					modelGradX, modelGradY, modelCenterX, modelCenterY);

				searchMatchModel(prySobelX[ModelInfo->PryNumber - 1],
					prySobelY[ModelInfo->PryNumber - 1],
					ModelInfo->Score,
					ModelInfo->Greediness,
					angle,
					ModelInfo->EdgeModelBaseInfoSize[ModelInfo->PryNumber - 1],
					modelGradX, modelGradY, modelCenterX, modelCenterY,
					searchInfo);
			}
#ifdef DrawContours
			rotateGradInfo(searchInfo.Angle,
				ModelInfo->EdgeModelBaseInfos[ModelInfo->PryNumber - 1],
				ModelInfo->EdgeModelBaseInfoSize[ModelInfo->PryNumber - 1],
				modelGradX, modelGradY, modelCenterX, modelCenterY);
			drawContours(prySrc[ModelInfo->PryNumber - 1],
				ModelInfo->EdgeModelBaseInfoSize[ModelInfo->PryNumber - 1],
				modelCenterX, modelCenterY, searchInfo);
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

void EdgeMatch::rotateGradInfo(float angle, EdgeModelBaseInfo*& modelInfo, int length,
	float*& modelGradX, float*& modelGradY, float*& modelContourX, float*& modelContourY)
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

	float sinA = sin(angle);
	float cosA = cos(angle);

	for (size_t i = 0; i < length; i++)
	{
		modelGradX[i] = modelInfo[i].grad_x * cosA - modelInfo[i].grad_y * sinA;
		modelGradY[i] = modelInfo[i].grad_y * cosA + modelInfo[i].grad_x * sinA;
		modelContourX[i] = modelInfo[i].contour.x * cosA - modelInfo[i].contour.y * sinA;
		modelContourY[i] = modelInfo[i].contour.y * cosA + modelInfo[i].contour.x * sinA;
	}
}

void EdgeMatch::searchMatchModel(cv::Mat& dstSobleX, cv::Mat& dstSobleY,
	float minScore, float greediness, float angle, int length,
	float*& modelGradX, float*& modelGradY, float*& modelContourX, float*& modelContourY,
	EdgeModelSearchInfo& searchInfo)
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

	int row = dstSobleX.rows;
	int col = dstSobleX.cols;
	for (size_t y = 0; y < row; y++)
	{
		for (size_t x = 0; x < col; x++)
		{
			/*以(x,y)为中心，计算相似度*/

			float partialScore = 0;
			float score = 0;
			int sum = 0;

			for (size_t index = 0; index < length; index++)
			{
				int curX = x + modelContourX[index];
				int curY = y + modelContourY[index];

				if (curX < 0 || curY < 0 || curX > col - 1 || curY > row - 1)
					continue;

				float angle = 0;
				if (pAngleImg != nullptr)
				{
					angle = pAngleImg[curY * col + curX];
				}
				else
				{
					angle = angleImg.at<float>(curY, curX);
				}

				float gx = cos(angle);
				float gy = sin(angle);
				if (gx != 0 || gy != 0)
				{
					partialScore += (gx * modelGradX[index] + gy * modelGradY[index]);

					sum++;
					score = partialScore / sum;
					if (score < (std::min((minScore - 1) + NormGreediness * sum, NormMinScore * sum)))
						break;
				}
			}

			//std::cout << "Angle = " << angle << ", OldScore = " << searchInfo.Score << ", NewScore = " << score << std::endl;
			if (score > searchInfo.Score)
			{
				std::cout << "Score = " << score << ", Angle = " << angle << ", CenterX = " << x << ", CenterY = " << y << std::endl;
				searchInfo.Score = score;
				searchInfo.Angle = angle;
				searchInfo.CenterX = x;
				searchInfo.CenterY = y;
			}
		}
	}
}

void EdgeMatch::drawContours(cv::Mat& dst, int length, float*& modelContourX, float*& modelContourY, EdgeModelSearchInfo& searchInfo)
{
	cv::cvtColor(dst, dst, cv::COLOR_GRAY2RGB);

	cv::Scalar color = cv::Scalar(255, 0, 0);
	for (size_t i = 0; i < length; i++)
	{
		cv::Point point = cv::Point(searchInfo.CenterX + modelContourX[i], searchInfo.CenterY + modelContourY[i]);
		cv::line(dst, point, point, color);
	}
}
