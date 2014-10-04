#include <opencv2\highgui.hpp>
#include <opencv2\imgcodecs.hpp>
#include <opencv2\imgproc.hpp>

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
}