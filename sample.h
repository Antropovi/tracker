#ifndef SAMPLE_H
#define SAMPLE_H

#include <vector>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <numeric>

#include "ffttools.h"
#include "ecofeatures.h"

class Sample
{
public:

    Sample();

    virtual    ~Sample(){}

    void       init(const std::vector<cv::Size>& filter, const std::vector<int>& feature_dim);

    void       update_sample_sapce_model(std::vector<std::vector<cv::Mat> >& new_train_sample);

    cv::Mat    find_gram_vector( std::vector<std::vector<cv::Mat> >& new_train_sample) ;

    float      feat_dis_compute(std::vector<std::vector<cv::Mat> >& feat1, std::vector<std::vector<cv::Mat> >& feat2);

    void       update_distance_matrix(cv::Mat& gram_vector, float new_sample_norm, int id1, int id2, float w1, float w2);

    void       findMin(float& min_w, int& index)const;

    std::vector<std::vector<cv::Mat> >  merge_samples(std::vector<std::vector<cv::Mat> >& sample1, std::vector<std::vector<cv::Mat> >& sample2, float w1, float w2, std::string sample_merge_type = "merge");

    void       replace_sample(std::vector<std::vector<cv::Mat> >& new_sample, long idx);

    void       set_gram_matrix(int r, int c, float val);

    int        get_merge_id()const { return merged_sample_id; }

    int        get_new_id()const   { return new_sample_id; }

    std::vector<float>      get_samples_weight()const { return prior_weights; }

    std::vector<std::vector<std::vector<cv::Mat> >>  get_samples() const{ return samples_f; }

private:
     mutable cv::Mat                    distance_matrix, gram_matrix;  //**** distance matrix and its kernel

     const int                          nSamples = 50;

     const float                        learning_rate = 0.009;

     const float                        minmum_sample_weight = 0.0036;

     mutable std::vector<float>              sample_weight;

     mutable std::vector<std::vector<std::vector<cv::Mat> >>          samples_f;                     //**** all samples frontier ******

     mutable int                        num_training_samples = 0;      //**** the number of training samples ********

     std::vector<float>                      prior_weights;

     mutable std::vector<std::vector<cv::Mat> >                          new_sample, merged_sample;

     int                                merged_sample_id = -1, new_sample_id = -1;

};

#endif // SAMPLE_H
