#include "img_proc.h"
#include <vector>

using namespace std;

void img_proc::hist(Mat src, Mat &dst, int kSizeSmooth, int norm_value) { // TODO: Work in progress
    CV_Assert(src.type() == CV_8U ||
              src.type() == CV_8UC3);

    vector<Mat> bgr_planes;
    split(src, bgr_planes);
    
    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };

    bool uniform = true, accumulate = false;

    Mat hist;
    int channels[3] = { 1, 1, 1 };
    calcHist(&bgr_planes[0], 1, channels, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

    if (kSizeSmooth != -1) {
        boxFilter(hist, hist, -1, Size(1, kSizeSmooth));
    }

    normalize(hist, hist, 0, norm_value, NORM_MINMAX, -1, Mat());
    hist.copyTo(dst);
}

void img_proc::triangle_bin(Mat src, Mat hist, Mat &dst, bool grayscale) {
    CV_Assert(src.type() == CV_8UC3);

    int minI = 256, maxI = -1;
    float minH = 1e30f, maxH = -1.0f;
    for (int i = 0; i < 127; ++i) {
        float rightH = hist.at<float>(255 - i);
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
    
    Mat triangle;
    src.copyTo(triangle);

    if (grayscale == true) {
        cvtColor(triangle, triangle, COLOR_BGR2GRAY);
    }
    threshold(triangle, dst, thresh, 255, THRESH_BINARY);
}

int img_proc::num_peaks(Mat hist, float thresh) {
    float left_min = hist.at<Vec3f>(0)[0];
    float right_min = left_min;
    int left_i = 0, right_i = 0;

    for (int i = 1, int k = 0; i < hist.rows; i++) {
        float cur_h = hist.at<Vec3f>(i)[0];
        if (cur_h < left_min) {
            left_min = cur_h;
            left_i = i;
        }
        else {
            float peak = -1.0f;
            for (int j = i + 1, float square = 0.0f; j < hist.rows; j++) {
                square += cur_h;
                cur_h = hist.at<Vec3f>(j)[0];
                if (peak < cur_h) {
                    peak = cur_h;
                }
                else if (right_min > cur_h) {
                    right_min = cur_h;
                    right_i = j;
                }
                else {
                    float peak = (1.0f - (float)(left_min + right_min) / (2.0f * peak)) *
                        (1.0f - square / ((float)(right_min - left_min) * peak));
                }
            }
        }
    }
}

struct cluster {
	Vec3f center;
	float num;
};

int img_proc::num_clusters(Mat src, int thresh) {
	vector<cluster> clusters;
	thresh *= thresh;

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			Vec3f cur = (Vec3f)src.at<Vec3b>(i, j);
			bool new_cluster = true;
			for (int k = 0; k < clusters.size(); k++) {
				Vec3f center = clusters[k].center;
				int len = (center[0] - cur[0]) * (center[0] - cur[0])
						  + (center[1] - cur[1]) * (center[1] - cur[1])
						  + (center[2] - cur[2]) * (center[2] - cur[2]);
				if (len < thresh) {
					int num = clusters[k].num;
					clusters[k].center = center * num / (float)num + cur / )(
					clusters[k].num += 1.0f;
				}
			}
		}
	}
}