#include "img_proc.h"
#include <stdlib.h>

using namespace std;

struct cluster {
    Vec3f center;
    int num;

    cluster(Vec3f _center, int _num) {
        center = _center;
        num = _num;
    }
};

int img_proc::num_clusters(Mat src, float delta, float thresh) {
    vector<cluster> clusters;

    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            Vec3f cur = (Vec3f)src.at<Vec3b>(i, j);
            float min_len = FLT_MAX;
            int min_i;
            for (int k = 0; k < clusters.size(); k++) {
                Vec3f center = clusters[k].center;
                float len = (center[0] - cur[0]) * (center[0] - cur[0])
                    + (center[1] - cur[1]) * (center[1] - cur[1])
                    + (center[2] - cur[2]) * (center[2] - cur[2]);
                if (len < min_len) {
                    min_len = len;
                    min_i = k;
                }
            }
            if (min_len < delta) {
                int num = clusters[min_i].num;
                clusters[min_i].center = clusters[min_i].center * num / (num + 1.0f) + cur / (num + 1.0f);
                clusters[min_i].num++;
            }
            else {
                clusters.push_back(cluster(cur, 1));
            }
        }
    }

    int size = src.rows * src.cols;
    int k = 0;
    for (int i = 0; i < clusters.size(); i++) {
        if (((float)clusters[i].num / size) > thresh) {
            k++;
        }
    }

    return k;
}

void img_proc::kmeans(Mat src, Mat &dst, int num_cl, double accuracy, OutputArray center) {
    CV_Assert(src.type() == CV_8U ||
              src.type() == CV_8UC3);

    int dim = src.channels();
    
    int *clust = new int[num_cl * dim];
    for (int i = 0; i < num_cl * dim; ++i) {
        clust[i] = rand() % 256;
    }

    Mat cluster(src.size(), CV_8U);
    int accrc;
    do {
        for (int i = 0; i < src.rows; i++) {
            for (int j = 0; j < src.cols; j++) {
                uchar *cur_vec = (uchar *)(src.ptr(i) + j * dim);
                int min_i, min_dist = INT_MAX;
                for (int k = 0; k < num_cl * dim; k += dim) {
                    int dist;
                    if (dim == 1) {
                        dist = abs(cur_vec[0] - clust[k]);    
                    }
                    else {
                        dist = (cur_vec[0] - clust[k]) * (cur_vec[0] - clust[k]) + 
                               (cur_vec[1] - clust[k + 1]) * (cur_vec[1] - clust[k + 1]) +
                               (cur_vec[2] - clust[k + 2]) * (cur_vec[2] - clust[k + 2]);
                    }
                    if (min_dist > dist) {
                            min_dist = dist;
                            min_i = k / dim;
                    }
                }
                cluster.at<uchar>(i, j) = min_i;
            }
        }
        
        float *mean = new float[num_cl * dim];
        int *quantity = new int[num_cl];
        memset(mean, 0, sizeof(float) * num_cl * dim);
        memset(quantity, 0, sizeof(int) * num_cl);

        for (int i = 0; i < src.rows; i++) {
            for (int j = 0; j < src.cols; j++) {
                int k = cluster.at<uchar>(i, j);
                uchar *cur_vec = (uchar *)(src.ptr(i) + j * dim);

                for (int z = 0; z < dim; z++) {
                    mean[k + z] += (float)(cur_vec[z]);
                }
                quantity[k] += 1;
            }
        }

        for (int k = 0; k < num_cl * dim; k++) {
            mean[k] /= (float)(quantity[k / dim]);
            clust[k] = cvRound(mean[k]);
        }

        accrc = 0;
        for (int i = 0; i < src.rows; ++i) {
            for (int j = 0; j < src.cols; ++j) {
                int k = cluster.at<uchar>(i, j);
                uchar *cur_vec = (uchar *)(src.ptr(i) + j * dim);

                for (int z = 0; z < dim; ++z) {
                    accrc += (clust[k + z] - cur_vec[z]) * (clust[k + z] - cur_vec[z]);
                }
            }
        }
    }
    while (accrc > accuracy);

    cluster.copyTo(dst);

    if (center.needed()) {
        center.create(Size(1, num_cl), src.type());
        Mat _center = center.getMat();
        uchar* c_ptr = _center.data;
        for (int i = 0; i < num_cl; i++) {
            c_ptr[i] = clust[i];
        }
    }
}