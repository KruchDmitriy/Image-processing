#include "img_proc.h"
#include <math.h>

#define clamp(a, low, high) min(max(a, low), high)

#define M_PI       3.14159265358979323846
#define M_PI_2     1.57079632679489661923
#define M_1_PI     0.318309886183790671538
#define M_2_PI     0.636619772367581343076

static inline float L(float x) {
    if (abs(x) < 1e-20)
        return 1.0f;
    return (float)(sin(M_PI * x) * sin(M_PI_2 * x) * M_1_PI * M_2_PI / (x * x));
}

void img_proc::resize(Mat src, Mat dst, Size size, int interpolation) {
    CV_Assert(src.type() == CV_8UC3);
    CV_Assert(interpolation >= 0 && interpolation < 3);

    dst.create(size, src.type());
    dst = Scalar(0);

    /**************** Nearest neighbour *******************/
    if (interpolation == INT_NEAREST) {
        for (int i = 0; i < dst.rows; ++i) {
            int y = clamp((int)((float)i * src.rows / dst.rows + 0.5f), 0, src.rows - 1);
            for (int j = 0; j < dst.cols; ++j) {
                int x = clamp((int)((float)j * src.cols / dst.cols + 0.5f), 0, src.cols - 1);
                dst.at<Vec3b>(i, j) = src.at<Vec3b>(y, x);
            }
        }
    }

    /**************** Bilinear *******************/
    else if (interpolation == INT_LINEAR) {
        uchar *src_ptr, *dst_ptr;
        for (int i = 0; i < dst.rows; ++i) {
            float y = (float)i * src.rows / dst.rows;
            int start_y = (int)y;
            src_ptr = src.ptr(clamp(start_y - 1, 0, src.rows - 2));
            dst_ptr = dst.ptr(max(i - 1, 0));
            for (int j = 0; j < dst.cols; ++j) {
                float x = (float)j * src.cols / dst.cols;
                float dx_left = x - (int)x;
                float dy_down = y - (int)y;

                for (int k = 0; k < 3; ++k) {
                    int start_x = 3 * (int)x + k;
                    dst_ptr[3 * j + k] = (int)(dx_left * 
                                    (src_ptr[start_x + src.step] * dy_down
                                    + src_ptr[start_x] * (1 - dy_down))
                                    + (1 - dx_left) * 
                                    (src_ptr[start_x + src.step + 3] * dy_down
                                    + src_ptr[start_x + 3] * (1 - dy_down)) + 0.5f);
                }
            }
        }
    }

    /**************** Lanczos *******************/
    else if (interpolation == INT_NONLINEAR) {
        for (int i = 0; i < dst.rows; ++i) {
            float y = i * (float)src.rows / dst.rows;
            for (int j = 0; j < dst.cols; ++j) {
                float x = j * (float)src.cols / dst.cols;
                Vec3f sum;
                sum.zeros();
                for (int dy = max((int)y - 1, 0); dy <= min((int)y + 2, src.rows - 1); ++dy) {
                    for (int dx = max((int)x - 1, 0); dx <= min((int)x + 2, src.cols - 1); ++dx) {
                         sum += ((Vec3f)src.at<Vec3b>(dy, dx)) * L(x - dx) * L(y - dy);
                    }
                }
                dst.at<Vec3b>(i, j) = sum;
            }
        }
    }
}