#ifndef SCORESOPTIMIZER_H
#define SCORESOPTIMIZER_H

#include "ffttools.h"

using namespace FFTTools;
class ScoresOptimizer
{
public:
    virtual ~ScoresOptimizer(){}

    ScoresOptimizer(){};  // default constructor
    ScoresOptimizer(std::vector<cv::Mat>& pscores_fs, int pite)
        :scores_fs(pscores_fs), iterations(pite){}

    void  compute_scores();

    std::vector<cv::Mat> sample_fs(const std::vector<cv::Mat>& xf, cv::Size grid_sz = cv::Size(0, 0));

    inline float  get_disp_row()const  { return disp_row; }
    inline float  get_disp_col()const  { return disp_col; }
    inline int    get_scale_ind()const { return scale_ind;}

private:
    std::vector<cv::Mat> scores_fs;
    int iterations;

    int scale_ind;
    float disp_row;
    float disp_col;

};
#endif // SCORESOPTIMIZER_H
