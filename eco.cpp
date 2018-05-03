#include "eco.h"

ECO::ECO(int name)
{
    _name = name;
    k1 = 0;
}

bool ECO::init(cv::Mat &im, const cv::Rect2d &rect)
{

    std::cout << "My name is : " << _name << std::endl;

    pos.x = rect.x + float(rect.width - 1) / 2;
    pos.y = rect.y + float(rect.height - 1) / 2;

    target_sz = rect.size();

    // *** Calculate search area and initial scale factor ****
    int search_area = rect.area() *  pow(params.search_area_scale, 2);
    if (search_area > params.max_image_sample_size)
        currentScaleFactor = sqrt((float)search_area / params.max_image_sample_size);
    else if (search_area < params.min_image_sample_size)
        currentScaleFactor = sqrt((float)search_area / params.min_image_sample_size);
    else
        currentScaleFactor = 1.0;

    base_target_sz = cv::Size2f(target_sz.width / currentScaleFactor, target_sz.height / currentScaleFactor);
    if (currentScaleFactor > 1)
        img_sample_sz = cv::Size(250, 250);
    else
        img_sample_sz = cv::Size(200, 200);

    init_features();  //TODO init_features() Kappa

    //*** the input-img-size of hog is the same as support size
    img_support_sz = hog_features.img_input_sz;
    std::cout << "img_support_sz is : " << img_support_sz << std::endl;

    feature_sz.push_back(cn_features.data_sz_block1);
    feature_dim.push_back(cn_features.fparams.nDim);
    compressed_dim.push_back(cn_features.fparams.compressed_dim);

    feature_sz.push_back(hog_features.data_sz_block1);
    feature_dim.push_back(hog_features.fparams.nDim);
    compressed_dim.push_back(hog_features.fparams.compressed_dim);
    //consider about ic_feature

    //***Number of Fourier coefficients to save for each filter layer.This will be an odd number.

    for (long i = 0; i != feature_sz.size(); ++i) {
        long size = feature_sz[i].width + (feature_sz[i].width + 1) % 2 ;
        filter_sz.push_back(cv::Size(size, size));
        k1 = size > output_sz ? i : k1;
        output_sz = std::max(size, output_sz);
    }

    //***Compute the Fourier series indices and their transposes***
    std::cout << "filter_sz.size() is : " << filter_sz.size() << std::endl;
    for (long i = 0; i < filter_sz.size(); ++i) {
        cv::Mat_<float> tempy(filter_sz[i].height, 1, CV_32FC1);
        cv::Mat_<float> tempx(1, filter_sz[i].height / 2 + 1, CV_32FC1);

        for (int j = 0; j < tempy.rows; j++) {
            tempy.at<float>(j, 0) = j - (tempy.rows / 2);
        }
        ky.push_back(tempy);

        float *tempxData = tempx.ptr<float>(0);
        for (int j = 0; j < tempx.cols; j++) {

            tempxData[j] = j - (filter_sz[i].height / 2);

        }
        kx.push_back(tempx);
    }

    //construct the Gaussian label function using Poisson formula
    yf_gaussion();
    // construct cosine window
    cos_wind();

    // compute Fourier series of interpolation function
    for (long i = 0; i < filter_sz.size(); ++i) {
        cv::Mat interp1_fs1, interp2_fs1;
        get_interp_fourier(filter_sz[i], interp1_fs1, interp2_fs1, params.interpolation_bicubic_a);

        interp1_fs.push_back(interp1_fs1);
        interp2_fs.push_back(interp2_fs1);
    }

    // construct spatial regularization filter
    // compute the energy of the filter (used for preconditioner)
    for (long i = 0; i < filter_sz.size(); i++) {
        cv::Mat temp = get_reg_filter(img_support_sz, base_target_sz, params);
        reg_filter.push_back(temp);
        cv::Mat_<float> t = temp.mul(temp);
        float energy = FFTTools::mat_sum(t);
        reg_energy.push_back(energy);
    }


    //*** scale facator **
    for (int i = -2; i < 3; i++) {
        scaleFactors.push_back(pow(params.scale_step, i));
    }


    std::vector<std::vector<cv::Mat> > xl, xlf, xlf_proj;

    std::cout << yf[0].size().width << "  |  " << yf[0].size().height;
    std::cout << yf[1].size().width << "  |  " << yf[1].size().height;

    xl = feat_extrator.extractor(im, pos, vector<float>(1, currentScaleFactor), feature_sz[0], params);

    //*** Do windowing of features ***
    xl = do_windows_x(xl, cos_window);

    //*** Compute the fourier series ***
    xlf = Features::do_dft(xl);

    //*** Interpolate features to the continuous domain **
    xlf = interpolate_dft(xlf, interp1_fs, interp2_fs);

    //*** New sample to be added
    xlf = compact_fourier_coeff(xlf);

    //*** Compress feature dementional projection matrix
    projection_matrix = init_projection_matrix(xl, compressed_dim, feature_dim);  //*** EXACT EQUAL TO MATLAB

    //*** project sample *****
    xlf_proj = Features::project_sample(xlf, projection_matrix);

    //*** Update the samplesf to include the new sample.The distance matrix, kernel matrix and prior weight are also updated
    SampleUpdate.init(filter_sz, compressed_dim);

    SampleUpdate.update_sample_sapce_model(xlf_proj);

    //**** used for precondition ******
    std::vector<std::vector<cv::Mat> > new_sample_energy = Features::feats_pow2(xlf_proj);
    sample_energy = new_sample_energy;

    vector<cv::Mat> proj_energy = project_mat_energy(projection_matrix, yf);

    std::vector<std::vector<cv::Mat> > hf, hf_inc;
    for (long i = 0; i < xlf.size(); i++) {
        hf.push_back(vector<cv::Mat>(xlf_proj[i].size(), cv::Mat::zeros(xlf_proj[i][0].size(), CV_32FC2)));
        hf_inc.push_back(vector<cv::Mat>(xlf_proj[i].size(), cv::Mat::zeros(xlf_proj[i][0].size(), CV_32FC2)));
    }

    eco_trainer.train_init(hf, hf_inc, projection_matrix, xlf, yf, reg_filter,
                           new_sample_energy, reg_energy, proj_energy, params);

    eco_trainer.train_joint();
    //   repeoject sample and updata sample space
    projection_matrix = eco_trainer.get_proj();

    SampleUpdate.replace_sample(xlf_proj, 0);

    //  Find the norm of the reprojected sample
    float new_sample_norm = Features::FeatEnergy(xlf_proj);  // equal to matlab
    SampleUpdate.set_gram_matrix(0, 0, 2 * new_sample_norm);

    frames_since_last_train = 0;

    std::vector<std::vector<cv::Mat> > temp = eco_trainer.get_hf();
    hf_full = full_fourier_coeff(temp);

    return true;
}

bool ECO::update(cv::Mat &frame, cv::Rect2d &rect)
{

    std::cout << "My name is : " << _name << std::endl;

    cv::Point sample_pos = cv::Point(pos);
    vector<float> det_samples_pos;

    for (long i = 0; i < scaleFactors.size(); ++i) {
        det_samples_pos.push_back(currentScaleFactor * scaleFactors[i]);
    }

    // 1: Extract features at multiple resolutions
    std::vector<std::vector<cv::Mat> > xt = feat_extrator.extractor(frame, sample_pos, det_samples_pos, feature_sz[0], params);

    //2:  project sample *****
    std::vector<std::vector<cv::Mat> > xt_proj = Features::FeatProjMultScale(xt, projection_matrix);

    // Do windowing of features ***
    xt_proj = do_windows_x(xt_proj, cos_window);

    // 3: Compute the fourier series ***
    xt_proj = Features::do_dft(xt_proj);

    // 4: Interpolate features to the continuous domain
    xt_proj = interpolate_dft(xt_proj, interp1_fs, interp2_fs);

    // 5: compute the scores of different scale of target
    vector<cv::Mat> scores_fs_sum;
    for (long i = 0; i < scaleFactors.size(); i++)
        scores_fs_sum.push_back(cv::Mat::zeros(filter_sz[k1], CV_32FC2));

    for (long i = 0; i < xt_proj.size(); i++) {
        int pad = (filter_sz[k1].height - xt_proj[i][0].rows) / 2;
        cv::Rect roi = cv::Rect(pad, pad, xt_proj[i][0].cols, xt_proj[i][0].rows);
        for (long j = 0; j < xt_proj[i].size(); j++) {
            cv::Mat score = complexMultiplication(xt_proj[i][j], hf_full[i][j % hf_full[i].size()]);
            score += scores_fs_sum[j / hf_full[i].size()](roi);
            score.copyTo(scores_fs_sum[j / hf_full[i].size()](roi));
        }
    }

    // 6: Locate the positon of target
    ScoresOptimizer scores(scores_fs_sum, params.newton_iterations);
    scores.compute_scores();
    float dx, dy;
    int scale_change_factor;
    scale_change_factor = scores.get_scale_ind();

    //scale_change_factor = 2;
    dx = scores.get_disp_col() * (img_support_sz.width / output_sz) * currentScaleFactor * scaleFactors[scale_change_factor];
    dy = scores.get_disp_row() * (img_support_sz.height / output_sz) * currentScaleFactor * scaleFactors[scale_change_factor];
    cv::Point old_pos;
    pos = cv::Point2f(sample_pos) + cv::Point2f(dx, dy);

    currentScaleFactor = currentScaleFactor *  scaleFactors[scale_change_factor];
    vector<float> sample_scale;
    for (long i = 0; i < scaleFactors.size(); ++i) {
        sample_scale.push_back(scaleFactors[i] * currentScaleFactor);
    }

    //*****************************************************************************
    //*****                     Model update step
    //******************************************************************************

    // 1: Use the sample that was used for detection
    std::vector<std::vector<cv::Mat> > xtlf_proj;
    for (long i = 0; i < xt_proj.size(); ++i) {
        std::vector<cv::Mat> tmp;
        int start_ind = scale_change_factor      *  projection_matrix[i].cols;
        int end_ind = (scale_change_factor + 1)  *  projection_matrix[i].cols;
        for (long j = start_ind; j < end_ind; ++j) {
            tmp.push_back(xt_proj[i][j].colRange(0, xt_proj[i][j].rows / 2 + 1));
        }
        xtlf_proj.push_back(tmp);
    }

    // 2: cv::Point shift_samp = pos - sample_pos : should ba added later !!!
    cv::Point2f shift_samp = cv::Point2f(pos - cv::Point2f(sample_pos));
    shift_samp = shift_samp * 2 * CV_PI * (1 / (currentScaleFactor * img_support_sz.width));
    xtlf_proj = shift_sample(xtlf_proj, shift_samp, kx, ky);

    // 3: Update the samplesf new sample, distance matrix, kernel matrix and prior weight
    SampleUpdate.update_sample_sapce_model(xtlf_proj);

    // 4: insert new sample
    if (SampleUpdate.get_merge_id() > 0) {
        SampleUpdate.replace_sample(xtlf_proj, SampleUpdate.get_merge_id());
    }
    if (SampleUpdate.get_new_id() > 0) {
        SampleUpdate.replace_sample(xtlf_proj, SampleUpdate.get_new_id());
    }

    // 5: update filter parameters
    bool train_tracker = frames_since_last_train >= params.train_gap;
    if (train_tracker) {
        std::vector<std::vector<cv::Mat> > new_sample_energy = Features::feats_pow2(xtlf_proj);
        sample_energy = Features::FeatScale(sample_energy, 1 - params.learning_rate) + Features::FeatScale(new_sample_energy, params.learning_rate);
        eco_trainer.train_filter(SampleUpdate.get_samples(), SampleUpdate.get_samples_weight(), sample_energy);
        frames_since_last_train = 0;
    } else {
        ++frames_since_last_train;
    }
    projection_matrix = eco_trainer.get_proj();//*** exect to matlab tracker

    std::vector<std::vector<cv::Mat> > temp = eco_trainer.get_hf();
    hf_full = full_fourier_coeff(temp);

    //*****************************************************************************
    //*****                    just for test
    //******************************************************************************
    cv::Rect2d resbox;
    resbox.width = base_target_sz.width * currentScaleFactor;
    resbox.height = base_target_sz.height * currentScaleFactor;
    resbox.x = pos.x - resbox.width / 2;
    resbox.y = pos.y - resbox.height / 2;

    rect = resbox;
}

void ECO::init_features()
{
    int max_cell_size = hog_features.fparams.cell_size;
    int new_sample_sz = (1 + 2 * img_sample_sz.width / (2 * max_cell_size)) * max_cell_size;
    vector<int> feature_sz_choices, num_odd_dimensions;
    int max_odd = -100, max_idx = -1;
    for (long i = 0; i < max_cell_size; i++) {
        int sz = (new_sample_sz + i) / max_cell_size;
        feature_sz_choices.push_back(sz);
        if (sz % 2 == 1) {
            max_idx = max_odd >= sz ? max_idx : i;
            max_odd = max_odd >= sz ? max_odd : sz;
        }
    }
    new_sample_sz += max_idx;
    img_support_sz = cv::Size(new_sample_sz, new_sample_sz);
    hog_features.img_sample_sz = img_support_sz;
    hog_features.img_input_sz = img_support_sz;
    cn_features.img_sample_sz = img_support_sz;
    cn_features.img_input_sz = img_support_sz;

    cn_features.data_sz_block1 = cv::Size(img_support_sz.width / cn_features.fparams.cell_size, img_support_sz.height / cn_features.fparams.cell_size);

    hog_features.data_sz_block1 = cv::Size(img_support_sz.width / hog_features.fparams.cell_size, img_support_sz.height / hog_features.fparams.cell_size);
    ECO::img_support_sz = img_support_sz;

    params.hog_feat = hog_features;
    params.cn_feat = cn_features;
}

void ECO::yf_gaussion()
{
    float sig_y = sqrt(int(base_target_sz.width) * int(base_target_sz.height)) *
                  (params.output_sigma_factor) * (float(output_sz) / img_support_sz.width);

    for (int i = 0; i < ky.size(); i++) {
        // ***** opencv matrix operation ******
        cv::Mat tempy(ky[i].size(), CV_32FC1);
        tempy = CV_PI * sig_y * ky[i] / output_sz;
        cv::exp(-2 * tempy.mul(tempy), tempy);
        tempy = sqrt(2 * CV_PI) * sig_y / output_sz * tempy;

        cv::Mat tempx(kx[i].size(), CV_32FC1);
        tempx = CV_PI * sig_y * kx[i] / output_sz;
        cv::exp(-2 * tempx.mul(tempx), tempx);
        tempx = sqrt(2 * CV_PI) * sig_y / output_sz * tempx;

        yf.push_back(cv::Mat(tempy * tempx));
    }
}

void ECO::cos_wind()
{
    for (long i = 0; i < feature_sz.size(); i++) {
        cv::Mat hann1t = cv::Mat(cv::Size(feature_sz[i].width + 2, 1), CV_32F, cv::Scalar(0));
        cv::Mat hann2t = cv::Mat(cv::Size(1, feature_sz[i].height + 2), CV_32F, cv::Scalar(0));
        for (int i = 0; i < hann1t.cols; i++)
            hann1t.at<float >(0, i) = 0.5 * (1 - std::cos(2 * CV_PI * i / (hann1t.cols - 1)));
        for (int i = 0; i < hann2t.rows; i++)
            hann2t.at<float >(i, 0) = 0.5 * (1 - std::cos(2 * CV_PI * i / (hann2t.rows - 1)));
        cv::Mat hann2d = hann2t * hann1t;
        cos_window.push_back(hann2d(cv::Range(1, hann2d.rows - 1), cv::Range(1, hann2d.cols - 1)));
    }//end for
}

std::vector<std::vector<cv::Mat> >  ECO::do_windows_x(const std::vector<std::vector<cv::Mat> > &xl, vector<cv::Mat> &cos_win)
{
    std::vector<std::vector<cv::Mat> > xlw;
    for (long i = 0; i < xl.size(); i++) {
        vector<cv::Mat> temp;
        for (long j = 0; j < xl[i].size(); j++)
            temp.push_back(cos_win[i].mul(xl[i][j]));
        xlw.push_back(temp);
    }
    return xlw;
}

std::vector<std::vector<cv::Mat> >  ECO::interpolate_dft(const std::vector<std::vector<cv::Mat> > &xlf, vector<cv::Mat> &interp1_fs, vector<cv::Mat> &interp2_fs)
{
    std::vector<std::vector<cv::Mat> > result;

    for (long i = 0; i < xlf.size(); i++) {
        cv::Mat interp1_fs_mat = RectTools::subwindow(interp1_fs[i], cv::Rect(cv::Point(0, 0), cv::Size(interp1_fs[i].rows, interp1_fs[i].rows)), cv::BORDER_REPLICATE);
        cv::Mat interp2_fs_mat = RectTools::subwindow(interp2_fs[i], cv::Rect(cv::Point(0, 0), cv::Size(interp2_fs[i].cols, interp2_fs[i].cols)), cv::BORDER_REPLICATE);
        vector<cv::Mat> temp;
        for (long j = 0; j < xlf[i].size(); j++) {
            temp.push_back(precision(complexMultiplication(complexMultiplication(interp1_fs_mat, xlf[i][j]), interp2_fs_mat)));
        }
        result.push_back(temp);
    }
    return result;
}

std::vector<std::vector<cv::Mat> >  ECO::compact_fourier_coeff(const std::vector<std::vector<cv::Mat> > &xf)
{
    std::vector<std::vector<cv::Mat> > result;
    for (long i = 0; i < xf.size(); i++) {
        vector<cv::Mat> temp;
        for (long j = 0; j < xf[i].size(); j++)
            temp.push_back(xf[i][j].colRange(0, (xf[i][j].cols + 1) / 2));
        result.push_back(temp);
    }
    return result;
}

vector<cv::Mat> ECO::init_projection_matrix(const std::vector<std::vector<cv::Mat> > &init_sample, const vector<int> &compressed_dim, const vector<int> &feature_dim)
{
    vector<cv::Mat> result;
    for (long i = 0; i < init_sample.size(); i++) {
        cv::Mat feat_vec(init_sample[i][0].size().area(), feature_dim[i], CV_32FC1);
        cv::Mat mean(init_sample[i][0].size().area(), feature_dim[i], CV_32FC1);
        for (int j = 0; j < init_sample[i].size(); j++) {
            float mean = cv::mean(init_sample[i][j])[0];
            for (long r = 0; r < init_sample[i][j].rows; r++)
                for (long c = 0; c < init_sample[i][j].cols; c++)
                    feat_vec.at<float>(c * init_sample[i][j].rows + r, j) = init_sample[i][j].at<float>(r, c) - mean;
        }
        result.push_back(feat_vec);
    }

    vector<cv::Mat> proj_mat;
    //****** svd operation ******
    for (long i = 0; i < result.size(); i++) {
        cv::Mat S, V, D;
        cv::SVD::compute(result[i].t()*result[i], S, V, D);
        vector<cv::Mat> V_;
        V_.push_back(V);
        V_.push_back(cv::Mat::zeros(V.size(), CV_32FC1));
        cv::merge(V_, V);
        proj_mat.push_back(V.colRange(0, compressed_dim[i]));  //** two channels : complex
    }

    return proj_mat;
}

vector<cv::Mat> ECO::project_mat_energy(vector<cv::Mat> proj, vector<cv::Mat> yf)
{
    vector<cv::Mat> result;

    for (long i = 0; i < yf.size(); i++) {
        cv::Mat temp(proj[i].size(), CV_32FC1), temp_compelx;
        float sum_dim = std::accumulate(feature_dim.begin(), feature_dim.end(), 0);
        cv::Mat x = yf[i].mul(yf[i]);
        temp = 2 * FFTTools::mat_sum(x) / sum_dim * cv::Mat::ones(proj[i].size(), CV_32FC1);
        result.push_back(temp);
    }
    return result;
}

std::vector<std::vector<cv::Mat> > ECO::shift_sample(std::vector<std::vector<cv::Mat> > &xf, cv::Point2f shift, std::vector<cv::Mat> kx, std::vector<cv::Mat>  ky)
{
    std::vector<std::vector<cv::Mat> > res;

    for (long i = 0; i < xf.size(); ++i) {
        cv::Mat shift_exp_y(ky[i].size(), CV_32FC2), shift_exp_x(kx[i].size(), CV_32FC2);
        for (long j = 0; j < ky[i].rows; j++) {
            shift_exp_y.at<cv::Vec<float, 2>>(j, 0) = cv::Vec<float, 2>(cos(shift.y * ky[i].at<float>(j, 0)), sin(shift.y * ky[i].at<float>(j, 0)));
        }

        for (long j = 0; j < kx[i].cols; j++) {
            shift_exp_x.at<cv::Vec<float, 2>>(0, j) = cv::Vec<float, 2>(cos(shift.x * kx[i].at<float>(0, j)), sin(shift.x * kx[i].at<float>(0, j)));
        }

        cv::Mat shift_exp_y_mat = RectTools::subwindow(shift_exp_y, cv::Rect(cv::Point(0, 0), xf[i][0].size()), cv::BORDER_REPLICATE);
        cv::Mat shift_exp_x_mat = RectTools::subwindow(shift_exp_x, cv::Rect(cv::Point(0, 0), xf[i][0].size()), cv::BORDER_REPLICATE);

        vector<cv::Mat> tmp;
        for (long j = 0; j < xf[i].size(); j++) {
            tmp.push_back(complexMultiplication(complexMultiplication(shift_exp_y_mat, xf[i][j]), shift_exp_x_mat));
        }
        res.push_back(tmp);
    }
    return res;
}

std::vector<std::vector<cv::Mat> >  ECO::full_fourier_coeff(std::vector<std::vector<cv::Mat> > &xf)
{
    std::vector<std::vector<cv::Mat> > res;
    for (long i = 0; i < xf.size(); i++) {
        vector<cv::Mat> tmp;
        for (long j = 0; j < xf[i].size(); j++) {
            cv::Mat temp = xf[i][j].colRange(0, xf[i][j].cols - 1).clone();
            rot90(temp, 3);
            cv::hconcat(xf[i][j], mat_conj(temp), temp);
            tmp.push_back(temp);
        }
        res.push_back(tmp);
    }

    return res;
}


void  ECO::get_interp_fourier(cv::Size filter_sz, cv::Mat &interp1_fs, cv::Mat &interp2_fs, float a)
{
    cv::Mat temp1(filter_sz.height, 1, CV_32FC1);
    cv::Mat temp2(1, filter_sz.width, CV_32FC1);
    for (int j = 0; j < temp1.rows; j++) {
        temp1.at<float>(j, 0) = j - temp1.rows / 2;
        temp2.at<float>(0, j) = j - temp1.rows / 2;
    }

    interp1_fs = cubic_spline_fourier(temp1 / filter_sz.height, a) / filter_sz.height;
    interp2_fs = cubic_spline_fourier(temp2 / filter_sz.width, a) / filter_sz.width;

    // ***Center the feature grids by shifting the interpolated features
    //*** Multiply Fourier coeff with e ^ (-i*pi*k / N)

    cv::Mat result1(temp1.size(), CV_32FC1), result2(temp1.size(), CV_32FC1);
    temp1 = temp1 / filter_sz.height;
    temp2 = temp2 / filter_sz.width;
    std::transform(temp1.begin<float>(), temp1.end<float>(), result1.begin<float>(), [](float x) -> float {return cos(x * CV_PI);});
    std::transform(temp1.begin<float>(), temp1.end<float>(), result2.begin<float>(), [](float x) -> float {return sin(x * CV_PI);});
    cv::Mat planes1[] = { interp1_fs.mul(result1), interp1_fs.mul(result2) };
    cv::merge(planes1, 2, interp1_fs);

    interp2_fs = interp1_fs.t();

}

cv::Mat ECO::cubic_spline_fourier(cv::Mat f, float a)
{
    if (f.empty())
        return cv::Mat();

    cv::Mat bf(f.size(), CV_32FC1), temp1(f.size(), CV_32FC1), temp2(f.size(), CV_32FC1),
    temp3(f.size(), CV_32FC1), temp4(f.size(), CV_32FC1);
    std::transform(f.begin<float>(), f.end<float>(), temp1.begin<float>(), [](float x) -> float {return cos(2 * x * CV_PI);});
    std::transform(f.begin<float>(), f.end<float>(), temp2.begin<float>(), [](float x) -> float {return cos(4 * x * CV_PI);});

    std::transform(f.begin<float>(), f.end<float>(), temp3.begin<float>(), [](float x) -> float {return sin(2 * x * CV_PI);});
    std::transform(f.begin<float>(), f.end<float>(), temp4.begin<float>(), [](float x) -> float {return sin(4 * x * CV_PI);});

    bf = -1 * (-12 * a * cv::Mat::ones(f.size(), CV_32FC1) + 24 * temp1 +
               12 * a * temp2 + CV_PI * 24 * f.mul(temp3) +
               CV_PI * a * 32 * f.mul(temp3) + CV_PI * 8 * a * f.mul(temp4) -
               24 * cv::Mat::ones(f.size(), CV_32FC1));

    cv::Mat L(f.size(), CV_32FC1);
    cv::pow(f, 4, L);
    cv::divide(bf, 16 * L * cv::pow(CV_PI, 4), bf);
    bf.at<float>(bf.rows / 2, bf.cols / 2) = 1;

    return bf;
}

cv::Mat ECO::get_reg_filter(cv::Size sz, cv::Size2f target_sz, const eco_params &params)
{
    float reg_window_edge = params.reg_window_edge;
    cv::Size2f reg_scale = cv::Size2f(target_sz.width * 0.5, target_sz.height * 0.5);

    // *** construct the regukarization window ***
    cv::Mat reg_window(sz, CV_32FC1);
    for (float x = -0.5 * (sz.height - 1), counter1 = 0; counter1 < sz.height; x += 1, ++counter1)
        for (float y = -0.5 * (sz.width - 1), counter2 = 0; counter2 < sz.width; y += 1, ++counter2)
            reg_window.at<float>(counter1, counter2) = (reg_window_edge - params.reg_window_min) *
                    (pow((abs(x / reg_scale.height)), 2) + pow(abs(y / reg_scale.width), 2))
                    + params.reg_window_min;

    //**** find the max value and norm ****
    cv::Mat reg_window_dft = fftd(reg_window) / sz.area();
    cv::Mat reg_win_abs(sz, CV_32FC1);
    reg_win_abs = magnitude(reg_window_dft);
    double minv = 0.0, maxv = 0.0;;
    cv::minMaxLoc(reg_win_abs, &minv, &maxv);

    //*** set to zero while the element smaller than threshold ***
    for (long i = 0; i < reg_window_dft.rows; i++)
        for (long j = 0; j < reg_window_dft.cols; j++) {
            if (reg_win_abs.at<float>(i, j) < (params.reg_sparsity_threshold * maxv))
                reg_window_dft.at<cv::Vec<float, 2>>(i, j) = cv::Vec<float, 2>(0, 0);
        }
    cv::Mat reg_window_sparse = FFTTools::real(FFTTools::fftd(reg_window_dft, true));
    cv::minMaxLoc(magnitude(reg_window_sparse), &minv, &maxv);
    reg_window_dft.at<float>(0, 0) -= sz.area() * minv + params.reg_window_min;
    reg_window_dft = FFTTools::fftshift(reg_window_dft);

    cv::Mat tmp, result;
    for (long i = 0; i < reg_window_dft.rows; i++) {
        for (long j = 0; j < reg_window_dft.cols; j++) {
            if (((reg_window_dft.at<cv::Vec<float, 2>>(i, j) != cv::Vec<float, 2>(0, 0)) &&
                    (reg_window_dft.at<cv::Vec<float, 2>>(i, j) != cv::Vec<float, 2>(2, 0)))) {
                tmp.push_back(reg_window_dft.row(i));
                break;
            }
        } //end for
    }//end for

    tmp = tmp.t();
    for (long i = 0; i < tmp.rows; i++) {
        for (long j = 0; j < tmp.cols; j++) {
            if (((tmp.at<cv::Vec<float, 2>>(i, j) != cv::Vec<float, 2>(0, 0)) &&
                    (tmp.at<cv::Vec<float, 2>>(i, j) != cv::Vec<float, 2>(1, 0)))) {
                result.push_back(FFTTools::real(tmp.row(i)));
                break;
            }
        } //end for
    }//end for
    result = result.t();

    return result;
}


cv::Mat ECO::precision(cv::Mat img)
{
    if (img.empty()) {
        return img;
    }
    std::vector<cv::Mat> img_v;
    cv::split(img, img_v);

    for (long i = 0; i < img_v.size(); i++) {
        img_v[i].convertTo(img_v[i], CV_32FC1);
        for (long r = 0; r < img_v[i].rows; r++) {
            for (long c = 0; c < img_v[i].cols; c++) {
                if (abs(img_v[i].at<float>(r, c)) < 0.0000499999) {
                    img_v[i].at<float>(r, c) = 0;
                    continue;
                }
                if ((abs(img_v[i].at<float>(r, c)) > 0.0000499999) && (abs(img_v[i].at<float>(r, c)) < 0.0001)) {
                    img_v[i].at<float>(r, c) = 0.0001;
                    continue;
                }
            }
        }//end for
        cv::Mat result;
        cv::merge(img_v, result);
        return result;
    }
}
