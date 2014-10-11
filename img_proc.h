#include <opencv2\highgui.hpp>
#include <opencv2\imgcodecs.hpp>
#include <opencv2\imgproc.hpp>

#include <stdlib.h>
#include <math.h>

using namespace cv;

namespace img_proc {

    enum {
        INT_NEAREST,
        INT_LINEAR,
        INT_NONLINEAR
    };

    void resize(Mat src, Mat &dst, Size size, int interpolation = INT_LINEAR);
    void hist(Mat src, Mat &dst, int kSizeSmooth = 3, int norm_value = 255);
    void triangle_bin(Mat src, Mat hist, Mat &dst);
    int num_peaks(Mat hist, float thresh = 0.5f);
    int num_clusters(Mat src, float delta, float thresh);
    void kmeans(Mat src, Mat &dst, int num_cl, double accuracy, OutputArray center = noArray());
    void moment(Mat src, int num_cl, int *&result, int x, int y, int *x_0 = NULL, int *y_0 = NULL);
    void perimeter(Mat src, int num_cl, int *&result);
}