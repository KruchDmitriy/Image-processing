#include "img_proc.h"
#include <vector>

using namespace std;

void img_proc::hist(Mat src, Mat *&dst, int kSizeSmooth, int norm_value) {
    CV_Assert(src.type() == CV_8UC3);

    vector<Mat> bgr_planes;
    split(src, bgr_planes);
    
    int histSize = 256;

    float range[] = { 0, 256 };
    const float* histRange = { range };

    bool uniform = true, accumulate = false;

    Mat b_hist, g_hist, r_hist;

    calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

    if (kSizeSmooth != -1) {
        boxFilter(b_hist, b_hist, -1, Size(kSizeSmooth, kSizeSmooth));
        boxFilter(g_hist, g_hist, -1, Size(kSizeSmooth, kSizeSmooth));
        boxFilter(r_hist, r_hist, -1, Size(kSizeSmooth, kSizeSmooth));
    }

    normalize(b_hist, b_hist, 0, norm_value, NORM_MINMAX, -1, Mat());
    normalize(g_hist, g_hist, 0, norm_value, NORM_MINMAX, -1, Mat());
    normalize(r_hist, r_hist, 0, norm_value, NORM_MINMAX, -1, Mat());

    dst = new Mat[3];
    dst[0] = b_hist;
    dst[1] = g_hist;
    dst[2] = r_hist;
}

void img_proc::triangle_bin(Mat src, Mat hist, Mat &dst, bool grayscale) {
    CV_Assert(src.type() == CV_8UC3);

    int minI = 256, maxI = -1;
    float minH = 1e30f, maxH = -1.0f;
    for (int i = 0; i < 127; ++i) {
        float leftH = hist.at<float>(i);
        float rightH = hist.at<float>(255 - i);
        if (leftH < minH && leftH > 1e-20) {
            minH = leftH;
            minI = i;
        }
        if (rightH > maxH) {
            maxH = rightH;
            maxI = 255 - i;
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
    
    if (grayscale == true) {
        cvtColor(src, src, COLOR_BGR2GRAY);
    }
    threshold(src, dst, thresh, 255, THRESH_BINARY);
}