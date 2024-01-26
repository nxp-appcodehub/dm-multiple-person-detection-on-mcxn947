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
#include <persondet_post_processing.h>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <forward_list>
#include <vector>

#include "flatbuffers/flexbuffers.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_utils.h"

#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#ifndef WIN32
#include "fsl_debug_console.h"
#endif
namespace fast_detection{

float Calculate1DOverlap(float x1Center, float width1, float x2Center, float width2)
    {
        float left_1 = x1Center - width1/2;
        float left_2 = x2Center - width2/2;
        float leftest = left_1 > left_2 ? left_1 : left_2;

        float right_1 = x1Center + width1/2;
        float right_2 = x2Center + width2/2;
        float rightest = right_1 < right_2 ? right_1 : right_2;

        return rightest - leftest;
    }

    float CalculateBoxIntersect(Box& box1, Box& box2)
    {
        float width = Calculate1DOverlap(box1.x, box1.w, box2.x, box2.w);
        if (width < 0) {
            return 0;
        }
        float height = Calculate1DOverlap(box1.y, box1.h, box2.y, box2.h);
        if (height < 0) {
            return 0;
        }

        float total_area = width*height;
        return total_area;
    }

    float CalculateBoxUnion(Box& box1, Box& box2)
    {
        float boxes_intersection = CalculateBoxIntersect(box1, box2);
        float boxes_union = box1.w * box1.h + box2.w * box2.h - boxes_intersection;
        return boxes_union;
    }

    float CalculateBoxIOU(Box& box1, Box& box2)
    {
        float boxes_intersection = CalculateBoxIntersect(box1, box2);
        if (boxes_intersection == 0) {
            return 0;
        }

        float boxes_union = CalculateBoxUnion(box1, box2);
        if (boxes_union == 0) {
            return 0;
        }

        return boxes_intersection / boxes_union;
    }

    void CalculateNMS(std::forward_list<Detection>& detections, int classes, float iouThreshold)
    {
        int idxClass{0};
        auto CompareProbs = [idxClass](Detection& prob1, Detection& prob2) {
            return prob1.prob[idxClass] > prob2.prob[idxClass];
        };

        for (idxClass = 0; idxClass < classes; ++idxClass) {
            detections.sort(CompareProbs);

            for (auto it=detections.begin(); it != detections.end(); ++it) {
                if (it->prob[idxClass] == 0) continue;
                for (auto itc=std::next(it, 1); itc != detections.end(); ++itc) {
                    if (itc->prob[idxClass] == 0) {
                        continue;
                    }
                    if (CalculateBoxIOU(it->bbox, itc->bbox) > iouThreshold) {
                        itc->prob[idxClass] = 0;
                    }
                }
            }
        }
    }

}

DetectorPostProcess::DetectorPostProcess(
		const TfLiteTensor* modelOutput,
		std::vector<fast_detection::DetectionResult>& results,
		const fast_detection::PostProcessParams& postProcessParams)
        :   m_outputTensor{modelOutput},
            m_results{results},
            m_postProcessParams{postProcessParams}
{

    /* Init PostProcessing */

    fast_detection::Network net;

    net.height = this->m_outputTensor->dims->data[1];
    net.width = this->m_outputTensor->dims->data[2];

    net.type = this->m_outputTensor->type;
    if (net.type == kTfLiteFloat32) {

    	net.modelOutput_f32 = this->m_outputTensor->data.f;

            } else if (net.type == kTfLiteInt8) {

            	net.modelOutput_int8 = this->m_outputTensor->data.int8;
            	net.scale = (static_cast<TfLiteAffineQuantization*>(this->m_outputTensor->quantization.params))->scale->data[0];
            	net.zeroPoint =(static_cast<TfLiteAffineQuantization*>(this->m_outputTensor->quantization.params))->zero_point->data[0];
            }

    net.inputHeight = postProcessParams.inputImgRows;
	net.inputWidth = postProcessParams.inputImgCols;
    net.numClasses = this->m_outputTensor->dims->data[3] - 5;
    
	net.topN = postProcessParams.topN;
	this->m_net = net;

    /* End init */
}

bool DetectorPostProcess::DoPostProcess()
{
    /* Start postprocessing */
    int originalImageWidth  = m_postProcessParams.originalImageWidth;
    int originalImageHeight = m_postProcessParams.originalImageHeight;

    std::forward_list<fast_detection::Detection> detections;
    GetNetworkBoxes(this->m_net, originalImageWidth, originalImageHeight, m_postProcessParams.threshold, detections);

    /* Do nms */
    CalculateNMS(detections, this->m_net.numClasses, this->m_postProcessParams.nms);

    for (auto& it: detections) {
        float xMin = it.bbox.x - it.bbox.w / 2.0f;
        float xMax = it.bbox.x + it.bbox.w / 2.0f;
        float yMin = it.bbox.y - it.bbox.h / 2.0f;
        float yMax = it.bbox.y + it.bbox.h / 2.0f;

        if (xMin < 0) {
            xMin = 0;
        }
        if (yMin < 0) {
            yMin = 0;
        }
        if (xMax > originalImageWidth) {
            xMax = originalImageWidth;
        }
        if (yMax > originalImageHeight) {
            yMax = originalImageHeight;
        }

        float boxX = xMin;
        float boxY = yMin;
        float boxWidth = xMax - xMin;
        float boxHeight = yMax - yMin;

        for (int j = 0; j < this->m_net.numClasses; ++j) {
        	float score = it.prob[j];
            if ( score > 0 && score < 1.0f) {

                fast_detection::DetectionResult tmpResult = {};
                tmpResult.m_normalisedVal = score;
                tmpResult.m_x0 = boxX;
                tmpResult.m_y0 = boxY;
                tmpResult.m_w = boxWidth;
                tmpResult.m_h = boxHeight;
                tmpResult.m_class = j;
                tmpResult.m_score = score;

                this->m_results.push_back(tmpResult);
            }
        }
    }
    return true;
}

void DetectorPostProcess::InsertTopNDetections(
    std::forward_list<fast_detection::Detection>& detections,
    fast_detection::Detection& det) {
    std::forward_list<fast_detection::Detection>::iterator it;
    std::forward_list<fast_detection::Detection>::iterator last_it;
    for ( it = detections.begin(); it != detections.end(); ++it ) {
        if(it->objectness > det.objectness)
            break;
        last_it = it;
    }
    if(it != detections.begin()) {
        detections.emplace_after(last_it, det);
        detections.pop_front();
    }
}

float _sigmoid(float x)
{
  return (1/(1 + std::exp(-x)));
}

float _tanh(float x)
{
	float x1 = std::exp(x);
	float x2 = std::exp(-x);
    
	return (x1 - x2) / (x1 + x2);
}

float expSum(const std::vector<float>& x) {
    float sum = 0.0;
    for (int i = 0; i < x.size(); i++) {
        sum += std::exp(x[i]);
    }
    return sum;
}

std::vector<float> _softmax(const std::vector<float>& x) {
    float expSumVal = expSum(x);
    std::vector<float> result(x.size());

    for (int i = 0; i < x.size(); i++) {
        result[i] = exp(x[i]) / expSumVal;
    }

    return result;
}

int _argmax(std::vector<float> list){

	std::vector<float>::iterator biggest = std::max_element(list.begin(), list.end());

	int idx = std::distance(list.begin(), biggest);

	return idx;
}

void DetectorPostProcess::GetNetworkBoxes(
    fast_detection::Network& net,
        int imageWidth,
        int imageHeight,
        float threshold,
        std::forward_list<fast_detection::Detection>& detections)
{
    int numClasses = net.numClasses;
    int num = 0;
    static int bbox_obj_offset;
     static float objectness;
    auto det_objectness_comparator = [](fast_detection::Detection& pa, fast_detection::Detection& pb) {
        return pa.objectness < pb.objectness;
    };
    int channel = numClasses + 5;
    for (int h = 0; h < net.height;h ++){

    	for (int w = 0; w < net.width; w++) {

    		bbox_obj_offset = h * net.width * channel + w * channel;
    		if (net.type == kTfLiteFloat32){
    			objectness = _sigmoid(
    			                        static_cast<float>(net.modelOutput_f32[bbox_obj_offset])
    			                      );
    		}
    		else if (net.type == kTfLiteInt8){
    			objectness =  _sigmoid((static_cast<float>(
    			                                        net.modelOutput_int8[bbox_obj_offset] -
    			                                        net.zeroPoint) *
    			                                    net.scale));
    		}

    		if(objectness > threshold) {
     			fast_detection::Detection det;
				det.objectness = objectness;

				float x,y,bw,bh;

				if (net.type == kTfLiteFloat32){
					x = _tanh(static_cast<float>(net.modelOutput_f32[bbox_obj_offset + 1]));
					y = _tanh(static_cast<float>(net.modelOutput_f32[bbox_obj_offset + 2]));
					bw = _sigmoid(static_cast<float>(net.modelOutput_f32[bbox_obj_offset + 3]));
					bh = _sigmoid(static_cast<float>(net.modelOutput_f32[bbox_obj_offset + 4]));
				}
				else if (net.type == kTfLiteInt8){
					x = _tanh(static_cast<float>((net.modelOutput_int8[bbox_obj_offset + 1] - net.zeroPoint) * net.scale) );
					y = _tanh(static_cast<float>((net.modelOutput_int8[bbox_obj_offset + 2]- net.zeroPoint)* net.scale));
					bw = _sigmoid(static_cast<float>((net.modelOutput_int8[bbox_obj_offset + 3]- net.zeroPoint)* net.scale));
					bh = _sigmoid(static_cast<float>((net.modelOutput_int8[bbox_obj_offset + 4]- net.zeroPoint)* net.scale));

				}

				if (numClasses > 1){
					std::vector<float> clses;

					for (int c = 0; c < numClasses; c++)
					{
						if (net.type == kTfLiteFloat32)
						{
							clses.push_back(static_cast<float>(net.modelOutput_f32[bbox_obj_offset + 5 + c]));

						}else if (net.type == kTfLiteInt8){
							clses.push_back(static_cast<float>((net.modelOutput_int8[bbox_obj_offset + 5 + c] - net.zeroPoint)* net.scale));
						}

					}
                    std::vector<float> cls_res = _softmax(clses);
                    for (int i = 0; i < cls_res.size();i++) {
                        float sig = cls_res[i];
                        det.prob.push_back((sig > threshold) ? sig: 0);
                    }
				}
				else {

					det.clasesidx = 0;
					float sig;

					sig = objectness;
					det.prob.push_back((sig > threshold) ? sig : 0);

				}

				float bcx = (x + w) / net.width;
				float bcy = (y + h) / net.height;

				det.bbox.x = bcx;
				det.bbox.y= bcy;

				det.bbox.w = bw;
				det.bbox.h = bh;

				/* Correct boxes */
				det.bbox.x *= imageWidth;
				det.bbox.w *= imageWidth;
				det.bbox.y *= imageHeight;
				det.bbox.h *= imageHeight;

				if (num < net.topN || net.topN <=0) {
					detections.emplace_front(det);
					num += 1;
				} else if (num == net.topN) {
					detections.sort(det_objectness_comparator);
					InsertTopNDetections(detections, det);
					num += 1;
				} else {
					InsertTopNDetections(detections, det);
				}

    		}
    	}
    }

    if(num > net.topN)
        num -=1;
}

