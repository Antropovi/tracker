#ifndef ECOFEATURES_H
#define ECOFEATURES_H

#include <string>
#include <math.h>
#include <vector>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <numeric>
#include <algorithm>

#include "params.h"
#include "ffttools.h"
#include "recttools.h"
#include "cn_data.cpp"

extern const float ColorNames[][10];

using namespace FFTTools;

class Features
{
public:
    Features();
    virtual    ~Features(){}

    std::vector<std::vector<cv::Mat> >  extractor(cv::Mat image, cv::Point2f pos, vector<float> scales, cv::Size output_sz, const eco_params& gparams);
    cv::Mat    sample_patch(const cv::Mat& im, const cv::Point2f& pos, cv::Size2f sample_sz, cv::Size2f output_sz, const eco_params& gparams);
    vector<cv::Mat>   get_hog(vector<cv::Mat> im);
    cv::Mat get_features_hog(const cv::Mat &im, const int bin_size);
    void computeHOG32D(const cv::Mat &imageM, cv::Mat &featM, const int sbin, const int pad_x, const int pad_y);
    vector<cv::Mat>   hog_feature_normalization(vector<cv::Mat>& feature);
    void cn_get_aver_reg(std::vector<cv::Mat>& feature);
    std::vector<cv::Mat> get_cn(const std::vector<cv::Mat> & patch_rgb, cv::Size &output_size);

    static std::vector<std::vector<cv::Mat> >   project_sample(const std::vector<std::vector<cv::Mat> >& x, const std::vector<cv::Mat>& projection_matrix);

    static std::vector<std::vector<cv::Mat> >   feats_pow2(const std::vector<std::vector<cv::Mat> >& feats);

    static std::vector<std::vector<cv::Mat> >   do_dft(const std::vector<std::vector<cv::Mat> >& xlw);
    static  std::vector<std::vector<cv::Mat> >  featDotMul(const std::vector<std::vector<cv::Mat> >& a, const std::vector<std::vector<cv::Mat> >& b);   // two features dot multiplication
    static  std::vector<std::vector<cv::Mat> >  FeatDotDivide(std::vector<std::vector<cv::Mat> > data1, std::vector<std::vector<cv::Mat> > data2);

    static std::vector<std::vector<cv::Mat> >  FeatAdd(std::vector<std::vector<cv::Mat> > data1, std::vector<std::vector<cv::Mat> > data2);
    static std::vector<std::vector<cv::Mat> > FeatMinus(std::vector<std::vector<cv::Mat> > data1, std::vector<std::vector<cv::Mat> > data2);
    static std::vector<cv::Mat> ProjAdd(std::vector<cv::Mat> data1, std::vector<cv::Mat> data2);
    static std::vector<cv::Mat> ProjMinus(std::vector<cv::Mat> data1, std::vector<cv::Mat> data2);
    static  std::vector<cv::Mat>   computeFeatSores(const std::vector<std::vector<cv::Mat> >& x, const std::vector<std::vector<cv::Mat> >& f); // compute socres  Sum(x * f)
    static  std::vector<std::vector<cv::Mat> >              computerFeatScores2(const std::vector<std::vector<cv::Mat> >& x, const std::vector<std::vector<cv::Mat> >& f);

    static  std::vector<std::vector<cv::Mat> >  FeatScale(std::vector<std::vector<cv::Mat> > data, float scale);

    static  void       symmetrize_filter(std::vector<std::vector<cv::Mat> >& hf);
    static  float      FeatEnergy(std::vector<std::vector<cv::Mat> >& feat);
    static  std::vector<cv::Mat>      FeatVec(const std::vector<std::vector<cv::Mat> >& x);   // vectorize features

    static  std::vector<std::vector<cv::Mat> >  FeatProjMultScale(const std::vector<std::vector<cv::Mat> >& x, const std::vector<cv::Mat>& projection_matrix);

    static  std::vector<cv::Mat>  ProjScale(std::vector<cv::Mat> data, float scale);


private:
    hog_feature	hog_features;
    cn_feature  cn_features;

    vector<cv::Mat> hog_feat_maps;
    vector<cv::Mat> cn_feat_maps;
};

template<typename T>
std::vector<T> operator+(const std::vector<T>& a, const std::vector<T>& b)
{
    assert(a.size() == b.size());
    std::vector<T> result;
    for (int i = 0; i < a.size(); ++i)
    {
        result.push_back(a[i] + b[i]);
    }
    return result;
}

template<typename T>
std::vector<T> operator-(const std::vector<T>& a, const std::vector<T>& b)
{
    assert(a.size() == b.size());
    std::vector<T> result;
    for (int i = 0; i < a.size(); ++i)
    {
        result.push_back(a[i] - b[i]);
    }
    return result;
}
#endif // FEATURES_H
