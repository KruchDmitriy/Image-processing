#include "img_proc.h"
#include <vector>

using namespace std;

void img_proc::hist(Mat src, Mat &dst, int kSizeSmooth, int norm_value) {
    CV_Assert(src.type() == CV_8U ||
              src.type() == CV_8UC3);
    
    int cn = src.channels();
    vector<Mat> bgr_planes;
    split(src, bgr_planes);
    
    int histSize = 256;
    if (cn == 3) {
        histSize *= 256 * 256;
    }
    float range[] = { 0, 256 };
    const float* histRange = { range };

    Mat hist(1, histSize, CV_8U);
    calcHist(&bgr_planes[0], cn, 0, Mat(), hist, 1, &histSize, &histRange, true, false);

    if (kSizeSmooth != -1) {
        boxFilter(hist, hist, -1, Size(1, kSizeSmooth));
    }

    normalize(hist, hist, 0, norm_value, NORM_MINMAX, -1, Mat());
    hist.copyTo(dst);
}

void img_proc::triangle_bin(Mat src, Mat hist, Mat &dst) {
    CV_Assert(src.depth() == CV_8U);

    int minI = 256, maxI = -1;
    float minH = 1e30f, maxH = -1.0f;
    for (int i = 127; i < 255; ++i) {
        float rightH = hist.at<float>(i);
        if (rightH > maxH) {
            maxH = rightH;
            maxI = 255 - i;
        }
    }
    for (int i = 0; i < maxI; ++i) {
        float leftH = hist.at<float>(i);
        if (leftH < minH && leftH > 1e-20) {
            minH = leftH;
            minI = i;
        }
    }

    float n1 = 1.0f, n2 = (minI - maxI) / (maxH - minH);
    n1 = n1 / sqrt(n1 * n1 + n2 * n2);
    n2 = n2 / sqrt(n1 * n1 + n2 * n2);

    int thresh;
    float maxLen = -1.0f;
    for (int i = minI; i < maxI; ++i) {
        float len = abs(n1 * i + n2 * hist.at<float>(i));
        if (len > maxLen) {
            maxLen = len;
            thresh = i;
        }
    }

    threshold(src, dst, thresh, 255, THRESH_BINARY);
}

int img_proc::num_peaks(Mat hist, float thresh) {
    float left_min = hist.at<uchar>(0);
    float right_min = 255;
    int left_i = 0, right_i = 0;

    int k = 0;
    for (int i = 1; i < hist.rows; i++) {
        float cur_h = hist.at<uchar>(i);
        if (cur_h < left_min) {
            left_min = cur_h;
            left_i = i;
        }
        else {
            uchar peak = 0;
            for (int j = i + 1, square = 0; j < hist.rows; j++) {
                square += cur_h;
                cur_h = hist.at<uchar>(j);
                if (peak < cur_h) {
                    peak = cur_h;
                }
                else if (right_min > cur_h) {
                    right_min = cur_h;
                    right_i = j;
                }
                else {
                    float measure = (1.0f - (float)(left_min + right_min) / (2.0f * peak)) *
                        (1.0f - square / ((float)(right_min - left_min) * peak));
                    if (measure > thresh) {
                        k++;
                    }
                    break;
                }
            }
        }
    }

    return k;
}