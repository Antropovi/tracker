#include "ecofeatures.h"

Features::Features()
{
}

std::vector<std::vector<cv::Mat> > Features::extractor(cv::Mat image, cv::Point2f pos, vector<float> scales, cv::Size output_sz, const eco_params& params)
{
    int num_features = 2, num_scales = scales.size();
    hog_features = params.hog_feat;
    cn_features = params.cn_feat;

    // extract image pathes for different kinds of feautures
    vector<vector<cv::Mat>> img_samples;
    for (int i = 0; i < num_features; ++i)
    {
        vector<cv::Mat> img_samples_temp(num_scales);
        for (int j = 0; j < scales.size(); ++j)
        {
            cv::Size2f img_sample_sz = (i == 0) ? cn_features.img_sample_sz : hog_features.img_sample_sz;
            cv::Size2f img_input_sz =  (i == 0) ? cn_features.img_input_sz : hog_features.img_input_sz;

            img_sample_sz.width *= scales[j];
            img_sample_sz.height *= scales[j];
            img_samples_temp[j] = sample_patch(image, pos, img_sample_sz, img_input_sz, params);
        }
        img_samples.push_back(img_samples_temp);

    }

    // Extract image patches features(all kinds of features)
    std::vector<std::vector<cv::Mat> > sum_features;
    cv::Mat xa;
    if (scales.size() != 1)
        xa = img_samples[0][2];
    else
        xa = img_samples[0][0];

    std::vector<cv::Mat> temp = get_cn(img_samples[0], output_sz);
    sum_features.push_back(temp);

    hog_feat_maps = get_hog(img_samples[img_samples.size() - 1]);
    vector<cv::Mat> hog_maps_vec = hog_feature_normalization(hog_feat_maps);
    sum_features.push_back(hog_maps_vec);

    return sum_features;
}

void Features::cn_get_aver_reg(std::vector<cv::Mat> &feature) {

    std::vector<cv::Mat> integralVecImg(feature.size());

    for (int i = 0; i < feature.size(); ++i){
        cv::Mat temp = cv::Mat::zeros(feature[i].rows + 1, feature[i].cols + 1, CV_32FC1);
        temp.copyTo(integralVecImg[i]);
        for (int c = 2; c < feature[i].cols + 1; ++c){
            for (int r = 2; r < feature[i].rows + 1; ++r)
                integralVecImg[i].at<cv::Vec3d>(c, r) = integralVecImg[i].at<cv::Vec3d>(c, r - 1) + feature[i].at<cv::Vec3d>(c, r - 1);
        }
        for (int c = 2; c < feature[i].cols + 1; ++c){
            for (int r = 2; r < feature[i].rows + 1; ++r)
                integralVecImg[i].at<cv::Vec3d>(c, r) = integralVecImg[i].at<cv::Vec3d>(c - 1, r) + feature[i].at<cv::Vec3d>(c - 1, r);
        }
        temp.release();
    }


}

cv::Mat Features::sample_patch(const cv::Mat& im, const cv::Point2f& poss, cv::Size2f sample_sz, cv::Size2f output_sz, const eco_params& gparams)
{
    cv::Point pos(poss.operator cv::Point());

    // Downsample factor
    float resize_factor = std::min(sample_sz.width / output_sz.width, sample_sz.height / output_sz.height);
    int df = std::max((float)floor(resize_factor - 0.1), float(1));
    cv::Mat new_im;
    im.copyTo(new_im);
    if (df > 1)
    {
        cv::Point os((pos.x - 0) % df, ((pos.y - 0) % df));
        pos.x = (pos.x - os.x - 1) / df + 1;
        pos.y = (pos.y - os.y - 1) / df + 1;

        sample_sz.width = sample_sz.width / df;
        sample_sz.height = sample_sz.height / df;

        int r = (im.rows - os.y) / df + 1, c = (im.cols - os.x) / df;
        cv::Mat new_im2(r, c, im.type());

        new_im = new_im2;
        int m, n;
        for (long i = 0 + os.y, m = 0; i < im.rows && m < new_im.rows; i += df, ++m)
            for (long j = 0 + os.x, n = 0; j < im.cols && n < new_im.cols; j += df, ++n)
                if (im.channels() == 1)
                    new_im.at<uchar>(m, n) = im.at<uchar>(i, j);
                else
                    new_im.at<cv::Vec3b>(m, n) = im.at<cv::Vec3b>(i, j);
    }

    // *** extract image ***
    sample_sz.width = round(sample_sz.width);
    sample_sz.height = round(sample_sz.height);
    cv::Point pos2(pos.x - floor((sample_sz.width + 1) / 2) + 1, pos.y - floor((sample_sz.height + 1) / 2) + 1);
    cv::Mat im_patch = RectTools::subwindow(new_im, cv::Rect(pos2, sample_sz), cv::BORDER_REPLICATE);
    cv::Mat resized_patch;
    cv::resize(im_patch, resized_patch, output_sz);
    return resized_patch;
}

vector<cv::Mat> Features::get_hog(vector<cv::Mat> ims)
{
    if (ims.empty())
        return vector<cv::Mat>();

    vector<cv::Mat> hog_feats;
    for (int i = 0; i < ims.size(); i++)
    {
        cv::Mat temp;
        ims[i].convertTo(temp, CV_32FC3);
        cv::Mat t = get_features_hog(temp, hog_features.fparams.cell_size);
        hog_feats.push_back(t);
    }
    return hog_feats;
}


void Features::computeHOG32D(const cv::Mat &imageM, cv::Mat &featM, const int sbin, const int pad_x, const int pad_y)
{
    const int dimHOG = 32;
    CV_Assert(pad_x >= 0);
    CV_Assert(pad_y >= 0);
    CV_Assert(imageM.channels() == 3);
    CV_Assert(imageM.depth() == CV_64F);

    // epsilon to avoid division by zero
    const double eps = 0.0001;
    // number of orientations
    const int numOrient = 18;
    // unit vectors to compute gradient orientation
    const double uu[9] = {1.000, 0.9397, 0.7660, 0.5000, 0.1736, -0.1736, -0.5000, -0.7660, -0.9397};
    const double vv[9] = {0.000, 0.3420, 0.6428, 0.8660, 0.9848,  0.9848,  0.8660,  0.6428,  0.3420};

    // image size
    const cv::Size imageSize = imageM.size();
    // block size
    // int bW = cvRound((double)imageSize.width/(double)sbin);
    // int bH = cvRound((double)imageSize.height/(double)sbin);
    int bW = cvFloor((double)imageSize.width/(double)sbin);
    int bH = cvFloor((double)imageSize.height/(double)sbin);
    const cv::Size blockSize(bW, bH);
    // size of HOG features
    int oW = std::max(blockSize.width-2, 0) + 2*pad_x;
    int oH = std::max(blockSize.height-2, 0) + 2*pad_y;
    cv::Size outSize = cv::Size(oW, oH);
    // size of visible
    const cv::Size visible = blockSize*sbin;

    // initialize historgram, norm, output feature matrices
    cv::Mat histM = cv::Mat::zeros(cv::Size(blockSize.width*numOrient, blockSize.height), CV_64F);
    cv::Mat normM = cv::Mat::zeros(cv::Size(blockSize.width, blockSize.height), CV_64F);
    featM = cv::Mat::zeros(cv::Size(outSize.width*dimHOG, outSize.height), CV_64F);

    // get the stride of each matrix
    const size_t imStride = imageM.step1();
    const size_t histStride = histM.step1();
    const size_t normStride = normM.step1();
    const size_t featStride = featM.step1();

    // calculate the zero offset
    const double* im = imageM.ptr<double>(0);
    double* const hist = histM.ptr<double>(0);
    double* const norm = normM.ptr<double>(0);
    double* const feat = featM.ptr<double>(0);

    for (int y = 1; y < visible.height - 1; y++)
    {
        for (int x = 1; x < visible.width - 1; x++)
        {
            // OpenCV uses an interleaved format: BGR-BGR-BGR
            const double* s = im + 3*std::min(x, imageM.cols-2) + std::min(y, imageM.rows-2)*imStride;

            // blue image channel
            double dyb = *(s+imStride) - *(s-imStride);
            double dxb = *(s+3) - *(s-3);
            double vb = dxb*dxb + dyb*dyb;

            // green image channel
            s += 1;
            double dyg = *(s+imStride) - *(s-imStride);
            double dxg = *(s+3) - *(s-3);
            double vg = dxg*dxg + dyg*dyg;

            // red image channel
            s += 1;
            double dy = *(s+imStride) - *(s-imStride);
            double dx = *(s+3) - *(s-3);
            double v = dx*dx + dy*dy;

            // pick the channel with the strongest gradient
            if (vg > v) { v = vg; dx = dxg; dy = dyg; }
            if (vb > v) { v = vb; dx = dxb; dy = dyb; }

            // snap to one of the 18 orientations
            double best_dot = 0;
            int best_o = 0;
            for (int o = 0; o < (int)numOrient/2; o++)
            {
                double dot =  uu[o]*dx + vv[o]*dy;
                if (dot > best_dot)
                {
                    best_dot = dot;
                    best_o = o;
                }
                else if (-dot > best_dot)
                {
                    best_dot = -dot;
                    best_o = o + (int)(numOrient/2);
                }
            }

            // add to 4 historgrams around pixel using bilinear interpolation
            double yp =  ((double)y+0.5)/(double)sbin - 0.5;
            double xp =  ((double)x+0.5)/(double)sbin - 0.5;
            int iyp = (int)cvFloor(yp);
            int ixp = (int)cvFloor(xp);
            double vy0 = yp - iyp;
            double vx0 = xp - ixp;
            double vy1 = 1.0 - vy0;
            double vx1 = 1.0 - vx0;
            v = sqrt(v);

            // fill the value into the 4 neighborhood cells
            if (iyp >= 0 && ixp >= 0)
                *(hist + iyp*histStride + ixp*numOrient + best_o) += vy1*vx1*v;

            if (iyp >= 0 && ixp+1 < blockSize.width)
                *(hist + iyp*histStride + (ixp+1)*numOrient + best_o) += vx0*vy1*v;

            if (iyp+1 < blockSize.height && ixp >= 0)
                *(hist + (iyp+1)*histStride + ixp*numOrient + best_o) += vy0*vx1*v;

            if (iyp+1 < blockSize.height && ixp+1 < blockSize.width)
                *(hist + (iyp+1)*histStride + (ixp+1)*numOrient + best_o) += vy0*vx0*v;

        } // for y
    } // for x

    // compute the energy in each block by summing over orientation
    for (int y = 0; y < blockSize.height; y++)
    {
        const double* src = hist + y*histStride;
        double* dst = norm + y*normStride;
        double const* const dst_end = dst + blockSize.width;
        // for each cell
        while (dst < dst_end)
        {
            *dst = 0;
            for (int o = 0; o < (int)(numOrient/2); o++)
            {
                *dst += (*src + *(src + numOrient/2))*
                    (*src + *(src + numOrient/2));
                src++;
            }
            dst++;
            src += numOrient/2;
        }
    }

    // compute the features
    for (int y = pad_y; y < outSize.height - pad_y; y++)
    {
        for (int x = pad_x; x < outSize.width - pad_x; x++)
        {
            double* dst = feat + y*featStride + x*dimHOG;
            double* p, n1, n2, n3, n4;
            const double* src;

            p = norm + (y - pad_y + 1)*normStride + (x - pad_x + 1);
            n1 = 1.0f / sqrt(*p + *(p + 1) + *(p + normStride) + *(p + normStride + 1) + eps);
            p = norm + (y - pad_y)*normStride + (x - pad_x + 1);
            n2 = 1.0f / sqrt(*p + *(p + 1) + *(p + normStride) + *(p + normStride + 1) + eps);
            p = norm + (y- pad_y + 1)*normStride + x - pad_x;
            n3 = 1.0f / sqrt(*p + *(p + 1) + *(p + normStride) + *(p + normStride + 1) + eps);
            p = norm + (y - pad_y)*normStride + x - pad_x;
            n4 = 1.0f / sqrt(*p + *(p + 1) + *(p + normStride) + *(p + normStride + 1) + eps);

            double t1 = 0.0, t2 = 0.0, t3 = 0.0, t4 = 0.0;

            // contrast-sesitive features
            src = hist + (y - pad_y + 1)*histStride + (x - pad_x + 1)*numOrient;
            for (int o = 0; o < numOrient; o++)
            {
                double val = *src;
                double h1 = std::min(val*n1, 0.2);
                double h2 = std::min(val*n2, 0.2);
                double h3 = std::min(val*n3, 0.2);
                double h4 = std::min(val*n4, 0.2);
                *(dst++) = 0.5 * (h1 + h2 + h3 + h4);

                src++;
                t1 += h1;
                t2 += h2;
                t3 += h3;
                t4 += h4;
            }

            // contrast-insensitive features
            src =  hist + (y - pad_y + 1)*histStride + (x - pad_x + 1)*numOrient;
            for (int o = 0; o < numOrient/2; o++)
            {
                double sum = *src + *(src + numOrient/2);
                double h1 = std::min(sum * n1, 0.2);
                double h2 = std::min(sum * n2, 0.2);
                double h3 = std::min(sum * n3, 0.2);
                double h4 = std::min(sum * n4, 0.2);
                *(dst++) = 0.5 * (h1 + h2 + h3 + h4);
                src++;
            }

            // texture features
            *(dst++) = 0.2357 * t1;
            *(dst++) = 0.2357 * t2;
            *(dst++) = 0.2357 * t3;
            *(dst++) = 0.2357 * t4;
            // truncation feature
            *dst = 0;
        }// for x
    }// for y
    // Truncation features
    for (int m = 0; m < featM.rows; m++)
    {
        for (int n = 0; n < featM.cols; n += dimHOG)
        {
            if (m > pad_y - 1 && m < featM.rows - pad_y && n > pad_x*dimHOG - 1 && n < featM.cols - pad_x*dimHOG)
                continue;

            featM.at<double>(m, n + dimHOG - 1) = 1;
        } // for x
    }// for y
}

cv::Mat Features::get_features_hog(const cv::Mat &im, const int bin_size)
{
    cv::Mat hogmatrix;
    cv::Mat im_;
    im.convertTo(im_, CV_64FC3, 1.0/255.0);
    computeHOG32D(im_,hogmatrix,bin_size,1,1);
    hogmatrix.convertTo(hogmatrix, CV_32F);
    cv::Size hog_size = im.size();
    hog_size.width /= bin_size;
    hog_size.height /= bin_size;
    cv::Mat hogc(hog_size, CV_32FC(32), hogmatrix.data);
    std::vector<cv::Mat> features;
    cv::split(hogc, features);
    return hogc;
}

vector<cv::Mat> Features::hog_feature_normalization(vector<cv::Mat>& hog_feat_maps)
{
    vector<cv::Mat> hog_maps_vec;
    for (long i = 0; i < hog_feat_maps.size(); i++)
    {
        vector<cv::Mat> temp_vec, result_vec;
        cv::split(hog_feat_maps[i], result_vec);
        hog_maps_vec.insert(hog_maps_vec.end(), result_vec.begin(), result_vec.end());
    }

    return hog_maps_vec;
}

std::vector<cv::Mat> Features::get_cn(const std::vector<cv::Mat> & patch_rgb, cv::Size &output_size)
{
    std::vector<cv::Mat> result;
    for (int t = 0; t < patch_rgb.size(); ++t){
        cv::Mat patch_data = patch_rgb[t].clone();
        cv::Vec3b & pixel = patch_data.at<cv::Vec3b>(0,0);
        unsigned index;

        cv::Mat cnFeatures = cv::Mat::zeros(patch_data.rows,patch_data.cols,CV_32FC(10));

        for(int i=0;i<patch_data.rows;i++){
            for(int j=0;j<patch_data.cols;j++){
                pixel=patch_data.at<cv::Vec3b>(i,j);
                index=(unsigned)(cvFloor((float)pixel[2]/8)+32*cvFloor((float)pixel[1]/8)+32*32*cvFloor((float)pixel[0]/8));

                //copy the values
                for(int k=0;k<10;k++){
                    cnFeatures.at<cv::Vec<float,10> >(i,j)[k]=(float)ColorNames[index][k];
                }
            }
        }
        std::vector<cv::Mat> res;
        cv::split(cnFeatures, res);
        for (size_t i = 0; i < res.size(); i++) {
            if (output_size.width > 0 && output_size.height > 0) {
                resize(res.at(i), res.at(i), output_size, cv::INTER_CUBIC);
            }
        }

        result.insert(result.end(), res.begin(), res.end());
    }
    return result;
}


std::vector<std::vector<cv::Mat> > Features::featDotMul(const std::vector<std::vector<cv::Mat> >& a, const std::vector<std::vector<cv::Mat> >& b)
{
    std::vector<std::vector<cv::Mat> > res;
    if (a.size() != b.size())
        assert("Unamtched feature size");

    for (long i = 0; i < a.size(); i++)
    {
        std::vector<cv::Mat> temp;
        for (long j = 0; j < a[i].size(); j++)
        {
            temp.push_back(FFTTools::complexMultiplication(a[i][j], b[i][j]));
        }
        res.push_back(temp);
    }
    return res;
}

std::vector<std::vector<cv::Mat> > Features::do_dft(const std::vector<std::vector<cv::Mat> >& xlw)
{
    std::vector<std::vector<cv::Mat> > xlf;
    for (long i = 0; i < xlw.size(); i++)
    {
        std::vector<cv::Mat> temp;
        for (long j = 0; j < xlw[i].size(); j++)
        {
            int size = xlw[i][j].rows;
            if (size % 2 == 1)
                temp.push_back(FFTTools::fftshift(fftd(xlw[i][j])));
            else
            {
                cv::Mat xf = FFTTools::fftshift(fftd(xlw[i][j]));
                cv::Mat xf_pad = RectTools::subwindow(xf, cv::Rect(cv::Point(0, 0), cv::Size(size + 1, size + 1)));
                for (long k = 0; k < xf_pad.rows; k++)
                {
                    xf_pad.at<cv::Vec<float, 2>>(size, k) = xf_pad.at<cv::Vec<float, 2>>(size - 1, k).conj();
                    xf_pad.at<cv::Vec<float, 2>>(k, size) = xf_pad.at<cv::Vec<float, 2>>(k, size - 1).conj();

                }
                temp.push_back(xf_pad);
            }
        }

        xlf.push_back(temp);
    }
    return xlf;

}

std::vector<std::vector<cv::Mat> > Features::project_sample(const std::vector<std::vector<cv::Mat> >& x, const std::vector<cv::Mat>& projection_matrix)
{
    std::vector<std::vector<cv::Mat> > result;

    for (long i = 0; i < x.size(); i++)
    {
        //**** smaple projection ******
        cv::Mat x_mat;
        for (long j = 0; j < x[i].size(); j++)
        {
            cv::Mat t = x[i][j].t();
            //wangsen ��ȷ��t�ǲ���iscontinuous
            x_mat.push_back(cv::Mat(1, x[i][j].size().area(), CV_32FC2, t.data));
        }
        x_mat = x_mat.t();

        cv::Mat res_temp = x_mat * projection_matrix[i];

        //**** reconver to standard formation ****
        std::vector<cv::Mat> temp;
        for (long j = 0; j < res_temp.cols; j++)
        {
            cv::Mat temp2 = res_temp.col(j);
            cv::Mat tt;
            temp2.copyTo(tt);                                 // the memory should be continous!!!!!!!!!!
            cv::Mat temp3(x[i][0].cols, x[i][0].rows, CV_32FC2, tt.data); //(x[i][0].cols, x[i][0].rows, CV_32FC2, temp2.data) int size[2] = { x[i][0].cols, x[i][0].rows };cv::Mat temp3 = temp2.reshape(2, 2, size)
            temp.push_back(temp3.t());
        }
        result.push_back(temp);
    }
    return result;

}

float Features::FeatEnergy(std::vector<std::vector<cv::Mat> >& feat)
{
    float res = 0;
    if (feat.empty())
        return res;

    cv::Mat temp;

    for (long i = 0; i < feat.size(); i++)
    {
        for (long j = 0; j < feat[i].size(); j++)
        {
            temp = FFTTools::real(FFTTools::complexMultiplication(FFTTools::mat_conj(feat[i][j]), feat[i][j]));
            //wangsen mat_sum "Can be improved, the overall function can also be improved"
            res += FFTTools::mat_sum(temp);
        }
    }
    return res;
}

std::vector<std::vector<cv::Mat> > Features::feats_pow2(const std::vector<std::vector<cv::Mat> >& feats)
{
    std::vector<std::vector<cv::Mat> > result;

    if (feats.empty())
    {
        return feats;
    }

    for (long i = 0; i < feats.size(); i++)
    {
        std::vector<cv::Mat> feat_vec; //*** locate memory ****
        for (long j = 0; j < feats[i].size(); j++)
        {
            cv::Mat temp(feats[i][0].size(), CV_32FC2);
            feats[i][j].copyTo(temp);
            for (long r = 0; r < feats[i][j].rows; r++)
            {
                for (long c = 0; c < feats[i][j].cols; c++)
                {
                    temp.at<cv::Vec<float, 2>>(r, c)[0] = pow(temp.at<cv::Vec<float, 2>>(r, c)[0], 2) + pow(temp.at<cv::Vec<float, 2>>(r, c)[1], 2);
                    temp.at<cv::Vec<float, 2>>(r, c)[1] = 0;
                }
            }
            feat_vec.push_back(temp);
        }
        result.push_back(feat_vec);
    }

    return result;

}

std::vector<std::vector<cv::Mat> >  Features::FeatDotDivide(std::vector<std::vector<cv::Mat> > a, std::vector<std::vector<cv::Mat> > b)
{
    std::vector<std::vector<cv::Mat> > res;

    if (a.size() != b.size())
        assert("Unamtched feature size");

    for (long i = 0; i < a.size(); i++)
    {
        std::vector<cv::Mat> temp;
        for (long j = 0; j < a[i].size(); j++)
        {
            temp.push_back(FFTTools::complexDivision(a[i][j], b[i][j]));
        }
        res.push_back(temp);
    }
    return res;

}

std::vector<cv::Mat> Features::computeFeatSores(const std::vector<std::vector<cv::Mat> >& x, const std::vector<std::vector<cv::Mat> >& f)
{
    std::vector<cv::Mat> res;

    std::vector<std::vector<cv::Mat> > res_temp = featDotMul(x, f);
    for (long i = 0; i < res_temp.size(); i++)
    {
        cv::Mat temp(cv::Mat::zeros(res_temp[i][0].size(), res_temp[i][0].type()));
        for (long j = 0; j < res_temp[i].size(); j++)
        {
            temp = temp + res_temp[i][j];
        }
        res.push_back(temp);
    }

    return res;
}

std::vector<std::vector<cv::Mat> >  Features::FeatScale(std::vector<std::vector<cv::Mat> > data, float scale)
{
    std::vector<std::vector<cv::Mat> > res;

    for (long i = 0; i < data.size(); i++)
    {
        std::vector<cv::Mat> tmp;
        for (long j = 0; j < data[i].size(); j++)
        {
            tmp.push_back(data[i][j] * scale);
        }
        res.push_back(tmp);
    }
    return res;
}


std::vector<std::vector<cv::Mat> >  Features::FeatAdd(std::vector<std::vector<cv::Mat> > data1, std::vector<std::vector<cv::Mat> > data2)
{
    std::vector<std::vector<cv::Mat> > res;

    for (long i = 0; i < data1.size(); i++)
    {
        std::vector<cv::Mat> tmp;
        for (long j = 0; j < data1[i].size(); j++)
        {
            tmp.push_back(data1[i][j] + data2[i][j]);
        }
        res.push_back(tmp);
    }

    return res;

}

std::vector<std::vector<cv::Mat> > Features::FeatMinus(std::vector<std::vector<cv::Mat> > data1, std::vector<std::vector<cv::Mat> > data2)
{
    std::vector<std::vector<cv::Mat> > res;

    for (long i = 0; i < data1.size(); i++)
    {
        std::vector<cv::Mat> tmp;
        for (long j = 0; j < data1[i].size(); j++)
        {
            tmp.push_back(data1[i][j] - data2[i][j]);
        }
        res.push_back(tmp);
    }

    return res;

}

void Features::symmetrize_filter(std::vector<std::vector<cv::Mat> >& hf)
{

    for (long i = 0; i < hf.size(); i++)
    {
        int dc_ind = (hf[i][0].rows + 1) / 2;
        for (long j = 0; j < hf[i].size(); j++)
        {
            int c = hf[i][j].cols - 1;
            for (long r = dc_ind; r < hf[i][j].rows; r++)
            {
                //cout << hf[i][j].at<cv::Vec<float, 2>>(r, c);
                hf[i][j].at<cv::Vec<float, 2>>(r, c) = hf[i][j].at<cv::Vec<float, 2>>(2 * dc_ind - r - 2, c).conj();
            }
        }

    }
}

std::vector<std::vector<cv::Mat> > Features::FeatProjMultScale(const std::vector<std::vector<cv::Mat> >& x, const std::vector<cv::Mat>& projection_matrix)
{
    std::vector<std::vector<cv::Mat> > result;
    //vector<cv::Mat> featsVec = FeatVec(x);
    for (long i = 0; i < x.size(); i++)
    {
        int org_dim = projection_matrix[i].rows;
        int numScales = x[i].size() / org_dim;
        std::vector<cv::Mat> temp;

        for (long s = 0; s < numScales; s++)   // for every scale
        {
            cv::Mat x_mat;
            for (long j = s * org_dim; j < (s + 1) * org_dim; j++)
            {
                cv::Mat t = x[i][j].t();
                x_mat.push_back(cv::Mat(1, x[i][j].size().area(), CV_32FC1, t.data));
            }

            x_mat = x_mat.t();

            cv::Mat res_temp = x_mat * FFTTools::real(projection_matrix[i]);

            //**** reconver to standard formation ****
            for (long j = 0; j < res_temp.cols; j++)
            {
                cv::Mat temp2 = res_temp.col(j);
                cv::Mat tt; temp2.copyTo(tt);                                 // the memory should be continous!!!!!!!!!!
                cv::Mat temp3(x[i][0].cols, x[i][0].rows, CV_32FC1, tt.data); //(x[i][0].cols, x[i][0].rows, CV_32FC2, temp2.data) int size[2] = { x[i][0].cols, x[i][0].rows };cv::Mat temp3 = temp2.reshape(2, 2, size)
                temp.push_back(temp3.t());
            }
        }
        result.push_back(temp);
    }
    return result;
}

std::vector<cv::Mat> Features::FeatVec(const std::vector<std::vector<cv::Mat> >& x)
{
    if (x.empty())
        return std::vector<cv::Mat>();

    std::vector<cv::Mat> res;
    for (long i = 0; i < x.size(); i++)
    {
        cv::Mat temp;
        for (long j = 0; j < x[i].size(); j++)
        {
            cv::Mat temp2 = x[i][j].t();
            temp.push_back(cv::Mat(1, x[i][j].size().area(), CV_32FC2, temp2.data));
        }
        res.push_back(temp);
    }
    return res;
}

std::vector<cv::Mat> Features::ProjScale(std::vector<cv::Mat> data, float scale)
{
    std::vector<cv::Mat> res;
    for (long i = 0; i < data.size(); i++)
    {
        res.push_back(data[i] * scale);
    }
    return res;
}

std::vector<cv::Mat> Features::ProjAdd(std::vector<cv::Mat> data1, std::vector<cv::Mat> data2)
{
    std::vector<cv::Mat> res;
    for (long i = 0; i < data1.size(); i++)
    {
        res.push_back(data1[i] + data2[i]);
    }
    return res;
}

std::vector<cv::Mat> Features::ProjMinus(std::vector<cv::Mat> data1, std::vector<cv::Mat> data2)
{
    std::vector<cv::Mat> res;
    for (long i = 0; i < data1.size(); i++)
    {
        res.push_back(data1[i] - data2[i]);
    }
    return res;
}
