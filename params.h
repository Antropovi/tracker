#ifndef PARAMS_H
#define PARAMS_H


#include <vector>
#include <string>
#include <opencv2/core.hpp>

using std::vector;
using std::string;

struct Hog_params
{
    Hog_params() :cell_size(4), compressed_dim(10), nOrients(9),nDim(32), penalty(0){}
    int           cell_size;
    int           compressed_dim;
    int           nOrients;
    long long     nDim;
    float         penalty;
};

struct Cn_params{
    Cn_params() :cell_size(4), compressed_dim(3), nDim(10), useForGray(false), tablename("CNorm"), penalty(0){}
    int cell_size;
    int compressed_dim;
    long long nDim;
    bool useForGray;
    string tablename;
    float penalty;
};

struct hog_feature
{
    Hog_params      fparams;

    cv::Size		img_input_sz;	            //*** input sample size ******
    cv::Size        img_sample_sz;			    //*** the size of sample *******
    cv::Size        data_sz_block1;			    //****  hog feature *****

};

struct cn_feature
{
    Cn_params      fparams;

    cv::Size		img_input_sz;	            //*** input sample size ******
    cv::Size        img_sample_sz;			    //*** the size of sample *******
    cv::Size        data_sz_block1;			    //****  hog feature *****

};

struct eco_params
{
    eco_params() :
        search_area_scale(4.5),  min_image_sample_size(40000), max_image_sample_size(62500),
        refinement_iterations(1), newton_iterations(5),      clamp_position(false),
        output_sigma_factor(0.0833333f), learning_rate(0.009), nSamples(50), sample_replace_strategy("lowest_prior"), lt_size(0), train_gap(5), skip_after_frame(1), use_detection_sample(1),
        use_reg_window(true), reg_window_min(1e-4), reg_window_edge(10e-3), reg_window_power(2), reg_sparsity_threshold(0.05f),
        interpolation_method("bicubic"), interpolation_bicubic_a(-0.75f), interpolation_centering(true), interpolation_windowing(false),
        number_of_scales(5), scale_step(1.02f)
    {}

    Hog_params hog_params;
    Cn_params cn_params;

    hog_feature hog_feat;
    cn_feature cn_feat;


    //***** img sample parameters *****
    float  search_area_scale;
    int    min_image_sample_size;
    int    max_image_sample_size;


    //***** Detection parameters *****
    int    refinement_iterations;               // Number of iterations used to refine the resulting position in a frame
    int	   newton_iterations ;                  // The number of Newton iterations used for optimizing the detection score
    bool   clamp_position;                      // Clamp the target position to be inside the image


    //***** Learning parameters
    float	output_sigma_factor;			    // Label function sigma
    float	learning_rate ;	 				    // Learning rate
    long	nSamples;                           // Maximum number of stored training samples
    string	sample_replace_strategy;            // Which sample to replace when the memory is full
    bool	lt_size;			                // The size of the long - term memory(where all samples have equal weight)
    int 	train_gap;					        // The number of intermediate frames with no training(0 corresponds to training every frame)
    int 	skip_after_frame;                   // After which frame number the sparse update scheme should start(1 is directly)
    bool	use_detection_sample;               // Use the sample that was extracted at the detection stage also for learning

    // Regularization window parameters
    bool	use_reg_window; 					// Use spatial regularization or not
    double	reg_window_min;						// The minimum value of the regularization window
    double  reg_window_edge;					// The impact of the spatial regularization
    long  reg_window_power;					// The degree of the polynomial to use(e.g. 2 is a quadratic window)
    float	reg_sparsity_threshold;				// A relative threshold of which DFT coefficients that should be set to zero


    // Interpolation parameters
    string  interpolation_method;				// The kind of interpolation kernel
    float   interpolation_bicubic_a;			// The parameter for the bicubic interpolation kernel
    bool    interpolation_centering;			// Center the kernel at the feature sample
    bool    interpolation_windowing;			// Do additional windowing on the Fourier coefficients of the kernel

    // Scale parameters for the translation model
    long  number_of_scales ;					// Number of scales to run the detector
    float   scale_step;                         // The scale factor


    //***  Conjugate Gradient parameters
    int     CG_iter      = 5;                  // The number of Conjugate Gradient iterations in each update after the first frame
    int     init_CG_iter = 10 * 15;            // The total number of Conjugate Gradient iterations used in the first frame
    int     init_GN_iter = 10;                 // The number of Gauss - Newton iterations used in the first frame(only if the projection matrix is updated)
    bool    CG_use_FR = false;                 // Use the Fletcher - Reeves(true) or Polak - Ribiere(false) formula in the Conjugate Gradient
    bool    pCG_standard_alpha = true;         // Use the standard formula for computing the step length in Conjugate Gradient
    int     CG_forgetting_rate = 75;	 	   // Forgetting rate of the last conjugate direction
    float   precond_data_param = 0.3;	 	   // Weight of the data term in the preconditioner
    float   precond_reg_param  = 0.015;	 	   // Weight of the regularization term in the preconditioner
    int     precond_proj_param = 35;	 	   // Weight of the projection matrix part in the preconditioner

    double  projection_reg = 5e-8; 	 	       // Regularization paremeter of the projection matrix

};





#endif // PARAMS_H
