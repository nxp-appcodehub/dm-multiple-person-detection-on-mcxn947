/*
 * Copyright (c) 2015-2016, Freescale Semiconductor, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * o Redistributions of source code must retain the above copyright notice, this list
 *   of conditions and the following disclaimer.
 *
 * o Redistributions in binary form must reproduce the above copyright notice, this
 *   list of conditions and the following disclaimer in the documentation and/or
 *   other materials provided with the distribution.
 *
 * o Neither the name of Freescale Semiconductor, Inc. nor the names of its
 *   contributors may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 * ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef MODEL_PERSONDET_POST_PROCESSING_H_
#define MODEL_PERSONDET_POST_PROCESSING_H_

#include <vector>
#include <forward_list>
#include "tensorflow/lite/c/common.h"

namespace fast_detection{

struct Box {
        float x;
        float y;
        float w;
        float h;
    };

    struct Detection {
        Box bbox;
        std::vector<float> prob;
        float objectness;
        int clasesidx;
    };


    struct Network {
    	int height;
    	int width;
    	union {
    	                float* modelOutput_f32;
    	                int8_t* modelOutput_int8;
			};
		float scale;
		int zeroPoint;
		TfLiteType type;
		int inputWidth;
		int inputHeight;
		int numClasses;
		int topN;
    };

    struct PostProcessParams {
        int inputImgRows;
        int inputImgCols;

        int originalImageWidth;
        int originalImageHeight;

        float threshold;
        float nms;
        int topN;

        float outs_scale;
        int outs_zeroPoint;
    };

    class DetectionResult {
    public:
        /**
         * @brief       Constructor
         * @param[in]   normalisedVal   Result normalized value
         * @param[in]   x0              Top corner x starting point
         * @param[in]   y0              Top corner y starting point
         * @param[in]   w               Detection result width
         * @param[in]   h               Detection result height
         **/
        DetectionResult(double normalisedVal, float x0, float y0, float w, float h,
                int m_class, float score) :
                m_normalisedVal(normalisedVal),
                m_x0(x0),
                m_y0(y0),
                m_w(w),
                m_h(h),
				m_class(m_class),
				m_score(score)
            {
            }

        DetectionResult() = default;
        ~DetectionResult() = default;

        double  m_normalisedVal{0.0};
		float m_x0{0};
		float m_y0{0};
		float m_w{0};
		float m_h{0};
		int m_class{0};
		float   m_score{0.0f};
    };

}

class BasePostProcess {

    public:
        virtual ~BasePostProcess() = default;

        /**
         * @brief       Should perform post-processing of the result of inference then populate
         *              populate result data for any later use.
         * @return      true if successful, false otherwise.
         **/
        virtual bool DoPostProcess() = 0;
    };


    class DetectorPostProcess : public BasePostProcess {
    public:
        /**
         * @brief        Constructor.
         * @param[in]    outputTensor0       Pointer to the TFLite Micro output Tensor at index 0.
         * @param[in]    outputTensor1       Pointer to the TFLite Micro output Tensor at index 1.
         * @param[out]   results             Vector of detected results.
         * @param[in]    postProcessParams   Struct of various parameters used in post-processing.
         **/
    	explicit DetectorPostProcess(const TfLiteTensor* outputTensor,
    									std::vector<fast_detection::DetectionResult>& results,
										const fast_detection::PostProcessParams& postProcessParams);

        /**
         * @brief    Should perform YOLO post-processing of the result of inference then
         *           populate Detection result data for any later use.
         * @return   true if successful, false otherwise.
         **/
        bool DoPostProcess() override;

    private:
        const TfLiteTensor* m_outputTensor;
        /* Output tensor index 1 */
        std::vector<fast_detection::DetectionResult>& m_results;       /* Single inference results. */
        const fast_detection::PostProcessParams& m_postProcessParams;  /* Post processing param struct. */
        fast_detection::Network m_net;

        /**
         * @brief       Insert the given Detection in the list.
         * @param[in]   detections   List of detections.
         * @param[in]   det          Detection to be inserted.
         **/
        void InsertTopNDetections(std::forward_list<fast_detection::Detection>& detections, fast_detection::Detection& det);

        /**
         * @brief        Given a Network calculate the detection boxes.
         * @param[in]    net           Network.
         * @param[in]    imageWidth    Original image width.
         * @param[in]    imageHeight   Original image height.
         * @param[in]    threshold     Detections threshold.
         * @param[out]   detections    Detection boxes.
         **/
        void GetNetworkBoxes(
            fast_detection::Network& net,
                             int imageWidth,
                             int imageHeight,
                             float threshold,
                             std::forward_list<fast_detection::Detection>& detections);
};


#endif /* MODEL_PERSONDET_POST_PROCESSING_H_ */
