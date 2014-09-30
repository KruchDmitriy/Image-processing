#include "img_proc.h"

#define HIST

using namespace img_proc;

const char *win_name_src = "src";
const char *win_name_dst = "dst";

Mat src, dst;

#ifdef RESIZE
    int new_w, new_h;
    int inter = INT_NONLINEAR;

    void sample_resize();
    void trackbar_resize(int, void*);
#endif
#ifdef HIST
    Mat *h;
    int kSizeSmooth;
    const char *win_name_hist = "hist";
    bool grayscale = true;
    
    void sample_hist();
    void trackbar_hist(int, void*);
#endif

int main() {
    char *filename = "E:/Images/fish.jpg";
    src = imread(filename);

    namedWindow(win_name_src);

#ifdef RESIZE
    sample_resize();
#endif
#ifdef HIST
    sample_hist();
#endif
    imshow(win_name_src, src);

    waitKey();
    destroyAllWindows();
    return 0;
}

#ifdef RESIZE
    void sample_resize() {
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
    void sample_hist() {
        kSizeSmooth = 1;

        createTrackbar("kernel size", win_name_src, &kSizeSmooth, 5, trackbar_hist);
    }
    void trackbar_hist(int, void*) {
        hist(src, h, kSizeSmooth * 2 - 1, 512);
        
        Mat sumHist(Size(512, 512), CV_8UC3, Scalar(0, 0, 0));
        int hist_h = 512;
        for (int i = 1; i < 256; i++)
        {
            line(sumHist, Point(2 * (i - 1), hist_h - cvRound(h[0].at<float>(i - 1))),
                Point(2 * i, hist_h - cvRound(h[0].at<float>(i))),
                Scalar(255, 0, 0), 2, 8, 0);
            line(sumHist, Point(2 * (i - 1), hist_h - cvRound(h[1].at<float>(i - 1))),
                Point(2 * i, hist_h - cvRound(h[1].at<float>(i))),
                Scalar(0, 255, 0), 2, 8, 0);
            line(sumHist, Point(2 * (i - 1), hist_h - cvRound(h[2].at<float>(i - 1))),
                Point(2 * i, hist_h - cvRound(h[2].at<float>(i))),
                Scalar(0, 0, 255), 2, 8, 0);
        }
        imshow(win_name_hist, sumHist);

        Mat gray_hist = 0.0722f * h[0] + 0.7152f * h[1] + 0.2126 * h[2];
        triangle_bin(src, gray_hist, dst, grayscale);
        imshow(win_name_dst, dst);
    }
#endif