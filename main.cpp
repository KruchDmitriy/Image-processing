#include "img_proc.h"

#define CLUST

using namespace img_proc;

const char *win_name_src = "src";
const char *win_name_dst = "dst";

Mat src, dst;

#ifdef RESIZE
    int new_w, new_h;
    int inter = INT_NONLINEAR;

    void trackbar_resize(int, void*); 
#endif
#ifdef HIST
    Mat h;
    int kSizeSmooth;
    const char *win_name_hist = "hist";
    bool grayscale = true;
    
    void trackbar_hist(int, void*);
#endif
#ifdef CLUST
    Mat clust;
    const char *win_name_clust = "k-means";
    float delta = 1000.0f;
    float thresh = 5e-2;
    float accuracy = 1e5;
#endif

void sample();

int main() {
    char *filename = "E:/Images/fish.jpg";
    src = imread(filename, IMREAD_COLOR);

    namedWindow(win_name_src);
    imshow(win_name_src, src);

    sample();

    waitKey();
    destroyAllWindows();
    return 0;
}

#ifdef RESIZE
    void sample() {
        new_w = src.cols / 2;
        new_h = src.rows / 2;

        createTrackbar("x", win_name_src, &new_w, src.cols * 4, trackbar_resize);
        createTrackbar("y", win_name_src, &new_h, src.rows * 4, trackbar_resize);
    }
    void trackbar_resize(int, void*) {
        resize(src, dst, Size(new_w, new_h), inter);
        imshow(win_name_dst, dst);
    }
#endif
#ifdef HIST
    void sample() {
        kSizeSmooth = 1;

        createTrackbar("kernel size", win_name_src, &kSizeSmooth, 5, trackbar_hist);
    }
    void trackbar_hist(int, void*) {
        hist(src, h, kSizeSmooth * 2 - 1, 512);
        
        Mat sumHist(Size(512, 512), CV_8UC3, Scalar(0, 0, 0));
        int hist_h = 512;
        for (int i = 1; i < 256; i++)
        {
            line(sumHist, Point(2 * (i - 1), hist_h - cvRound(h.at<float>(i - 1))),
                Point(2 * i, hist_h - cvRound(h.at<float>(i))),
                Scalar(255, 255, 255), 2, 8, 0);
        }
        imshow(win_name_hist, sumHist);

        Mat gray_hist;
        if (h.channels() == 3) {
            cvtColor(h, gray_hist, COLOR_BGR2GRAY);
        }
        else {
            h.copyTo(gray_hist);
        }

        triangle_bin(src, gray_hist, dst);
        imshow(win_name_dst, dst);
    }
#endif
#ifdef CLUST
    void sample() {
        int k = num_clusters(src, delta, thresh);
        Mat center;
        kmeans(src, dst, k, accuracy, center);
        
        clust.create(src.size(), src.type());
        Vec3b *c_ptr = (Vec3b *)center.data;
        for (int i = 0; i < dst.rows; i++) {
            for (int j = 0; j < dst.cols; j++) {
                clust.at<Vec3b>(i, j) = c_ptr[dst.at<uchar>(i, j)];
            }
        }

        imshow(win_name_clust, clust);
    }
#endif