// EdgeMatch.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include "EdgeMatch.h"
#include <iostream>
#include <math.h>

int main()
{
    EdgeMatch::GetInstance().create_edge_model_path("D:\\Download\\GeoMatch_demo\\Template.jpg",
        "0d4ed8a0-9a35-42cb-ac77-b06c76ed13c8", 30, 45, 3);

    system("pause");
}

int EdgeMatch::create_edge_model_path(const char* picPath, const char* modelID, 
    int minGray, int maxGray, int pryNum)
{
    cv::Mat src = cv::imread(picPath, cv::IMREAD_GRAYSCALE);

    cv::Mat cannyImg;
    cv::Canny(src, cannyImg, minGray, maxGray);

    cv::Mat sobelX, sobleY;
    cv::Sobel(cannyImg, sobelX, CV_32FC1, 1, 0, 3);
    cv::Sobel(cannyImg, sobleY, CV_32FC1, 0, 1, 3);
    
    std::vector<cv::Mat> pryCanny;
    std::vector<cv::Mat> prySobelX;
    std::vector<cv::Mat> prySobelY;
    pryImage(cannyImg, pryCanny, pryNum);
    pryImage(sobelX, prySobelX, pryNum);
    pryImage(sobleY, prySobelY, pryNum);

    ModelInfo = new EdgeModelInfo;
    ModelInfo->MinGray = minGray;
    ModelInfo->MaxGray = maxGray;
    ModelInfo->PryNum = pryNum;

    for (size_t i = 0; i < pryNum; i++)
    {
        std::vector<EdgeModelBaseInfo> gradData;
        createGrad(pryCanny[i], prySobelX[i], prySobelY[i], gradData);
        ModelInfo->EdgeModelBaseInfos.push_back(gradData);
    }

    return 1;
}

void EdgeMatch::pryImage(cv::Mat srcImage,std::vector<cv::Mat>& pryImage, int pryNum)
{
    pryImage.push_back(srcImage);

    for (size_t i = 1; i < pryNum; i++)
    {
        cv::pyrDown(srcImage, srcImage);
        pryImage.push_back(srcImage);
    }
}

void EdgeMatch::createGrad(cv::Mat pryCanny, cv::Mat prySobelX, cv::Mat prySobelY, std::vector<EdgeModelBaseInfo>& gradData)
{
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(pryCanny, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Point> contoursQuery;
    for (size_t j = 0; j < contours.size(); j++)
    {
        contoursQuery.insert(contoursQuery.begin(), contours[j].begin(), contours[j].end());
    }

    cv::Mat magnitudeImg, angleImg;
    cv::cartToPolar(prySobelX, prySobelY, magnitudeImg, angleImg);

    for (size_t j = 0; j < contoursQuery.size(); j++)
    {
        EdgeModelBaseInfo gradInfo;
        float grad = magnitudeImg.at<float>(contoursQuery[j]);
        float angle = angleImg.at<float>(contoursQuery[j]);

        gradInfo.grad_x = cos(angle);
        gradInfo.grad_y = sin(angle);
        gradInfo.magnitude = grad;
        if (cv::abs(grad) < 1e-7)
            gradInfo.magnitudeN = 1 / grad;
        gradInfo.contour = contoursQuery[j];

        gradData.push_back(gradInfo);
    }
}