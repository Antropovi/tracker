#ifndef ECO_H
#define ECO_H

#include <opencv2/opencv.hpp>

#include "params.h"
#include "ecofeatures.h"
#include "sample.h"
#include "scoresoptimizer.h"
#include "trainer.h"

using namespace FFTTools;

class ECO
{
public:
    virtual ~ECO();
    ECO();
    ECO(int name);
    bool init(cv::Mat &im, const cv::Rect2d &rect);
    bool update(cv::Mat &frame, cv::Rect2d &rect);
    void init_features();

    void yf_gaussion();
    void cos_wind();

    std::vector<std::vector<cv::Mat> > do_windows_x(const std::vector<std::vector<cv::Mat> > &xl, vector<cv::Mat> &cos_win);
    std::vector<std::vector<cv::Mat> > interpolate_dft(const std::vector<std::vector<cv::Mat> > &xlf, vector<cv::Mat> &interp1_fs, vector<cv::Mat> &interp2_fs);

    std::vector<std::vector<cv::Mat> > compact_fourier_coeff(const std::vector<std::vector<cv::Mat> > &xf);

    vector<cv::Mat> init_projection_matrix(const std::vector<std::vector<cv::Mat> > &init_sample, const vector<int> &compressed_dim, const vector<int> &feature_dim);

    vector<cv::Mat> project_mat_energy(vector<cv::Mat> proj, vector<cv::Mat> yf);

    std::vector<std::vector<cv::Mat> > shift_sample(std::vector<std::vector<cv::Mat> > &xf, cv::Point2f shift, std::vector<cv::Mat> kx, std::vector<cv::Mat>  ky);

    std::vector<std::vector<cv::Mat> > full_fourier_coeff(std::vector<std::vector<cv::Mat> > &xf);

    static void  get_interp_fourier(cv::Size filter_sz, cv::Mat &interp1_fs, cv::Mat &interp2_fs, float a);
    static cv::Mat  cubic_spline_fourier(cv::Mat f, float a);
    static cv::Mat  get_reg_filter(cv::Size sz, cv::Size2f target_sz, const eco_params &params);

    cv::Mat precision(cv::Mat img);

private:

    long output_sz, k1, frameID, frames_since_last_train;     //*** the max size of feature and its index
    cv::Point2f pos;

    eco_params params;			 // *** ECO prameters ***

    //***  current target size,  initial target size,
    cv::Size target_sz, init_target_sz, img_sample_sz, img_support_sz;
    cv::Size2f base_target_sz;     // *** adaptive target size

    float currentScaleFactor; //*** current img scale ******

    hog_feature hog_features;       //*** corresponding to original matlab features{2}
    cn_feature  cn_features;

    vector<cv::Size> feature_sz, filter_sz;
    vector<int> feature_dim, compressed_dim;

    vector<cv::Mat> ky, kx, yf, cos_window;                 // *** Compute the Fourier series indices and their transposes
    vector<cv::Mat>	interp1_fs, interp2_fs;				 // *** interpl fourier series

    vector<cv::Mat>	reg_filter, projection_matrix;			 //**** spatial filter *****
    vector<float> reg_energy, scaleFactors;

    Features feat_extrator;

    Sample SampleUpdate;

    std::vector<std::vector<cv::Mat> > sample_energy;
    std::vector<std::vector<cv::Mat> > hf_full;
    Trainer eco_trainer;

    int _name;
};

#endif // ECO_H
