#ifndef TRAINER_H
#define TRAINER_H

#include <iostream>
#include <string>
#include <math.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <algorithm>

#include "ffttools.h"
#include "recttools.h"
#include "params.h"
#include "ecofeatures.h"

class Trainer
{
public:

    Trainer();

    virtual ~Trainer();

    struct STATE
    {
        std::vector<std::vector<cv::Mat> > p, r_prev;
        float  rho;
    };

    typedef   struct rl_out                                 //*** the right and left side of the equation
    {
        rl_out(){}
        rl_out(std::vector<std::vector<cv::Mat> > pup_part, std::vector<cv::Mat> plow_part):
            up_part(pup_part), low_part(plow_part){}

        std::vector<std::vector<cv::Mat> >			       up_part;                 //*** this is f + delta(f)
        std::vector<cv::Mat>       low_part;                //**  this is delta(P)

        rl_out    operator*(float  scale);                  //*** the union structure scale transformation
        rl_out    operator+(rl_out data2);                  //*** the union structure scale transformation
        rl_out    operator-(rl_out data2);                  //*** the union structure scale transformation

    }joint_out, joint_fp;

    Trainer(std::vector<std::vector<cv::Mat> > phf, std::vector<std::vector<cv::Mat> > phf_inc, vector<cv::Mat> pproj_matrix, std::vector<std::vector<cv::Mat> > pxlf, vector<cv::Mat> pyf,
        vector<cv::Mat> preg_filter, std::vector<std::vector<cv::Mat> > psample_energy, vector<float> preg_energy, vector<cv::Mat> pproj_energy,
        eco_params& params)
    {
                       train_init( phf, phf_inc, pproj_matrix, pxlf, pyf, preg_filter, psample_energy, preg_energy, pproj_energy,params);
    }

    void train_init(std::vector<std::vector<cv::Mat> > phf, std::vector<std::vector<cv::Mat> > phf_inc, vector<cv::Mat> pproj_matrix, std::vector<std::vector<cv::Mat> > pxlf, vector<cv::Mat> pyf,
        vector<cv::Mat> preg_filter, std::vector<std::vector<cv::Mat> > psample_energy, vector<float> preg_energy, vector<cv::Mat> pproj_energy,
        eco_params& params);

    void                train_joint();

    std::vector<std::vector<cv::Mat> >           project_sample(const std::vector<std::vector<cv::Mat> >& x, const vector<cv::Mat>& projection_matrix);

    std::vector<std::vector<cv::Mat> >           mtimesx(std::vector<std::vector<cv::Mat> >& x, vector<cv::Mat> y, bool _conj = 0);   //*** feature * yf

    vector<cv::Mat>     compute_rhs2(const vector<cv::Mat>& proj_mat, const vector<cv::Mat>& init_samplef_H, const std::vector<std::vector<cv::Mat> >& fyf, const vector<int>& lf_ind);

    vector<cv::Mat>     feat_vec(const std::vector<std::vector<cv::Mat> >& x);                  //*** conver feature into a  vector-matrix

    joint_out           lhs_operation_joint(joint_fp& hf, const std::vector<std::vector<cv::Mat> >& samplesf, const vector<cv::Mat>& reg_filter, const std::vector<std::vector<cv::Mat> >& init_samplef, vector<cv::Mat>XH,
                        const std::vector<std::vector<cv::Mat> >&  init_hf, float proj_reg);          //*** the left side of equation in paper  A(x) to compute residual

    joint_fp            pcg_eco(const std::vector<std::vector<cv::Mat> >& init_samplef_proj, const vector<cv::Mat>& reg_filter, const std::vector<std::vector<cv::Mat> >& init_samplef,const vector<cv::Mat>& init_samplesf_H, const std::vector<std::vector<cv::Mat> >& init_hf, float proj_reg, // right side of equation A(x)
                                const joint_out& rhs_samplef,  // the left side of the equation
                                const joint_out& diag_M,       // preconditionor
                                joint_fp& hf);                   // the union of filter [f+delta(f) delta(p)]

    //**** access to private membership
    std::vector<std::vector<cv::Mat> >          get_hf()  const{ return hf; }
    vector<cv::Mat>    get_proj()const{ return projection_matrix; }

    //**** joint structure basic operation
    joint_out          joint_minus( const joint_out&a, const joint_out& b);    // minus
    joint_out          diag_precond(const joint_out&a, const joint_out& b);    // devide
    float              inner_product_joint(const joint_out&a, const joint_out& b);
    float              inner_product(const std::vector<std::vector<cv::Mat> >& a, const std::vector<std::vector<cv::Mat> >& b);

    //****  this part is for filter training ***
    //****  this part is for filter training ***
    void	           train_filter(const vector<std::vector<std::vector<cv::Mat> >>& samplesf, const vector<float>& sample_weights, const std::vector<std::vector<cv::Mat> >& sample_energy);
    std::vector<std::vector<cv::Mat> >          pcg_eco_filter(const vector<std::vector<std::vector<cv::Mat> >>& samplesf, const vector<cv::Mat>& reg_filter, const vector<float> &sample_weights,  // right side of equation A(x)
                        const std::vector<std::vector<cv::Mat> >& rhs_samplef,  // the left side of the equation
                        const std::vector<std::vector<cv::Mat> >& diag_M,       // preconditionor
                        std::vector<std::vector<cv::Mat> >& hf);                   // the union of filter [f+delta(f) delta(p)]

    std::vector<std::vector<cv::Mat> >          lhs_operation(std::vector<std::vector<cv::Mat> >& hf, const vector<std::vector<std::vector<cv::Mat> >>& samplesf, const vector<cv::Mat>& reg_filter, const vector<float> &sample_weights);

    std::vector<std::vector<cv::Mat> >          conv2std(const vector<std::vector<std::vector<cv::Mat> >>& samplesf)const;

private:

    std::vector<std::vector<cv::Mat> >			hf, hf_inc;      //*** filter parameters and its increament ***
    std::vector<std::vector<cv::Mat> >           xlf, sample_energy;
                                         //*** the features fronier transform and its energy ****

    vector<cv::Mat>     yf;              //*** the label of sample **********

    vector<cv::Mat>     reg_filter;
    vector<float>       reg_energy;

    vector<cv::Mat>     projection_matrix, proj_energy;              //**** projection matrix and its energy ***

    eco_params          params;

    float               resvec, resle;   //*** Prellocate vector for norm of residuals  norm(b - A(x))s
    STATE               state;
};
#endif // TRAINER_H
