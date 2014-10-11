#include "img_proc.h"

#define CLUST

using namespace img_proc;
using namespace std;

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

        /* 
        Now lets calc:
            1. Square.
            2. Perimeter.
            3. Compact.
            4. Center of graity
            5. Orientation main axe momentum
        */

        int *S, *P;
        moment(dst, k, S, 0, 0);
        perimeter(dst, k, P);

        int *m01, *m10, *m11, *m20, *m02;
        moment(dst, k, m01, 0, 1);
        moment(dst, k, m10, 1, 0);
        moment(dst, k, m11, 1, 1);
        moment(dst, k, m20, 2, 0);
        moment(dst, k, m02, 0, 2);

        float *C = new float[k];
        float *Xc = new float[k], 
            *Yc = new float[k];
        float *O = new float[k];

        for (int i = 0; i < k; i++) {
            C[i] = (float)P[i] * P[i] / S[i];
            Xc[i] = (float)m01[i] / S[i];
            Yc[i] = (float)m10[i] / S[i];
            O[i] = 0.5f * atan(2.0f * m11[i] / (m20[i] - m02[i]));
        }
        
        FILE *f;
        f = fopen("output.txt", "w");
        fprintf(f, "Square: \n");
        for (int i = 0; i < k; i++) {
            fprintf(f, "%d %d\n", i, S[i]);
        }
        fprintf(f, "\nPerimeter: \n");
        for (int i = 0; i < k; i++) {
            fprintf(f, "%d %d\n", i, P[i]);
        }
        fprintf(f, "\nCompact: \n");
        for (int i = 0; i < k; i++) {
            fprintf(f, "%d %f\n", i, C[i]);
        }
        fprintf(f, "\nCenter of gravity: \n");
        for (int i = 0; i < k; i++) {
            fprintf(f, "%d (%f, %f)\n", i, Xc[i], Yc[i]);
        }
        fprintf(f, "\nOrientation main axe momentum: \n");
        for (int i = 0; i < k; i++) {
            fprintf(f, "%d %f (rad)\n", i, O[i]);
        }
        fclose(f);
    }
#endif