#include <iostream>
#include <opencv2/opencv.hpp>
#include <omp.h>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{    
    // Load image
    Mat image = imread("doge.jpg");
    if (image.empty()) {
        cout << "Failed to load image " << endl;
        return 1;
    }

    int rows = image.rows;
    int cols = image.cols;
    int channels = image.channels();

    omp_set_num_threads(12);
    
    double start_time = omp_get_wtime();


    //----------- 1. Mirrored image -----------

    Mat mirrored = image.clone();

    #pragma omp parallel for
        for (int i = 0; i < cols; i++) 
        {
            for (int j = 0; j < rows; j++)
            {
                Vec3b color2 = image.at<Vec3b>(Point(i, j));
                Vec3b color1 = mirrored.at<Vec3b>(Point((cols - 1) - i, j));
                color2.val[0] = color1.val[0];
                color2.val[1] = color1.val[1];
                color2.val[2] = color1.val[2];

                image.at<Vec3b>(Point(i, j)) = color1;
            }
        }

    //----------- 2. Negated image ----------- 

    Mat negated = image.clone();

    #pragma omp parallel for
        for (int i = 0; i < rows; ++i) 
        {
            for (int j = 0; j < cols; ++j) 
            {
                Vec3b pixel = image.at<Vec3b>(i, j);
                negated.at<Vec3b>(i, j) = Vec3b(255 - pixel[0], 255 - pixel[1], 255 - pixel[2]);
            }
        }


    //----------- 3. Gaussian Blur ----------- 

        //kernel_size = 3
        double sigma = 0.9; // standard deviation

        double kernel[3][3]; //creating kernel to be used
        double sum = 0.0;
        for (int i = 0; i < 3; i++) 
        {
            for (int j = 0; j < 3; j++) 
            {
                kernel[i][j] = exp(-(i * i + j * j) / (2 * sigma * sigma));
                sum += kernel[i][j];
            }
        }
        
        for (int i = 0; i < 3; i++) 
        {
            for (int j = 0; j < 3; j++) 
            {
                kernel[i][j] /= sum;
            }
        }

        Mat blurred = image.clone();

        #pragma omp parallel for
            for (int i = 3 / 2; i < rows - 3 / 2; i++) 
            {
                for (int j = 3 / 2; j < cols - 3 / 2; j++) 
                {
                    double sum[3] = { 0.0 };
                    for (int k = -3 / 2; k <= 3 / 2; k++) 
                    {
                        for (int l = -3 / 2; l <= 3 / 2; l++) 
                        {
                            for (int c = 0; c < channels; c++) 
                            {
                                sum[c] += blurred.at<Vec3b>(i + k, j + l)[c] * kernel[k + 3 / 2][l + 3 / 2];
                            }
                        }
                    }
                    for (int c = 0; c < channels; c++) 
                    {
                        blurred.at<Vec3b>(i, j)[c] = (uchar)sum[c];
                    }
                }
            }

        //----------- 4. Grayscale image ----------- 

        Mat gray(image.size(), CV_8UC1);
        #pragma omp parallel for
            for (int i = 0; i < rows; ++i) 
            {
                for (int j = 0; j < cols; ++j) 
                {
                    Vec3b pixel = image.at<Vec3b>(i, j);
                    gray.at<uchar>(i, j) = static_cast<uchar>(0.299 * pixel[2] + 0.587 * pixel[1] + 0.114 * pixel[0]);
                }
            }

        //----------- 5. Saturated image ----------- 

        Mat satur = image.clone();
        float alpha = 2.0;
        int beta = 75;

        int rows2 = rows;
        int cols2 = cols * channels;

        #pragma omp parallel for
            for (int i = 0; i < rows; i++)
            {
                uchar* ptr = satur.ptr<uchar>(i);

                for (int j = 0; j < cols; j++)
                {
                    for (int c = 0; c < channels; c++)
                    {
                        double val = ptr[j * channels + c];
                        val = alpha * val + beta;

                        if (val < 0)
                            val = 0;
                        if (val > 255)
                            val = 255;

                        ptr[j * channels + c] = static_cast<uchar>(val);
                    }
                }
            }


    double end_time = omp_get_wtime();
    double time_taken = end_time - start_time;

    // Display results
    namedWindow("Original", WINDOW_NORMAL);
    namedWindow("Mirrored", WINDOW_NORMAL);
    namedWindow("Negated", WINDOW_NORMAL);
    namedWindow("Gaussian Blur", WINDOW_NORMAL);
    namedWindow("Grayscaled", WINDOW_NORMAL);
    namedWindow("Saturated image", WINDOW_NORMAL);

    imshow("Original", image);
    imshow("Mirrored", mirrored);
    imshow("Negated", negated);
    imshow("Gaussian Blur", blurred);
    imshow("Grayscaled", gray);
    imshow("Saturated image", satur);
    
    cout << "\nTime taken: " << time_taken << " seconds" << endl;
    waitKey(0);
    
    //To get the time for serial implementation, simply remove the "#pragma omp parallel" lines and run again
    return 0;
}
