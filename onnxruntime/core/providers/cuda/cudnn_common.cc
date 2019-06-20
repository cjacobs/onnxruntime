// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cudnn_common.h"
#include "gsl/gsl_util"
#include "shared_inc/cuda_call.h"
#include "core/providers/cpu/tensor/utils.h"

namespace onnxruntime {
namespace cuda {

CudnnTensor::CudnnTensor()
    : tensor_(nullptr) {
}

CudnnTensor::~CudnnTensor() {
  if (tensor_ != nullptr) {
    cudnnDestroyTensorDescriptor(tensor_);
    tensor_ = nullptr;
  }
}

Status CudnnTensor::CreateTensorIfNeeded() {
  if (!tensor_)
    CUDNN_RETURN_IF_ERROR(cudnnCreateTensorDescriptor(&tensor_));
  return Status::OK();
}

Status CudnnTensor::Set(const std::vector<int64_t>& input_dims, cudnnDataType_t dataType) {
  ORT_RETURN_IF_ERROR(CreateTensorIfNeeded());

  int rank = gsl::narrow_cast<int>(input_dims.size());
  TensorPitches pitches(input_dims);
  std::vector<int> dims(rank);
  std::vector<int> strides(rank);
  for (int i = 0; i < rank; i++) {
    dims[i] = gsl::narrow_cast<int>(input_dims[i]);
    strides[i] = gsl::narrow_cast<int>(pitches[i]);
  }
  CUDNN_RETURN_IF_ERROR(cudnnSetTensorNdDescriptor(tensor_, dataType, static_cast<int>(rank), dims.data(), strides.data()));
  return Status::OK();
}

Status CudnnTensor::Set(const CudnnTensor& x_desc, cudnnBatchNormMode_t mode) {
  ORT_RETURN_IF_ERROR(CreateTensorIfNeeded());
  CUDNN_RETURN_IF_ERROR(cudnnDeriveBNTensorDescriptor(tensor_, x_desc, mode));
  return Status::OK();
}

template <typename ElemType>
cudnnDataType_t CudnnTensor::GetDataType() {
  if (typeid(ElemType) == typeid(float))
    return CUDNN_DATA_FLOAT;
  else if (typeid(ElemType) == typeid(double))
    return CUDNN_DATA_DOUBLE;
  else if (typeid(ElemType) == typeid(half))
    return CUDNN_DATA_HALF;
  else
    ORT_THROW("cuDNN engine currently supports only single/double/half precision data types.");
}

CudnnDataTensor::CudnnDataTensor()
    : tensor_(nullptr) {
}

CudnnDataTensor::~CudnnDataTensor() {
  if (tensor_ != nullptr) {
    cudnnDestroyRNNDataDescriptor(tensor_);
    tensor_ = nullptr;
  }
}

Status CudnnDataTensor::CreateTensorIfNeeded() {
  if (!tensor_)
    CUDNN_RETURN_IF_ERROR(cudnnCreateRNNDataDescriptor(&tensor_));
  return Status::OK();
}

Status CudnnDataTensor::Set(cudnnDataType_t dataType,
                            int64_t max_seq_length,
                            int64_t batch_size,
                            int64_t data_size,
                            const int32_t* seq_lengths) {
  ORT_RETURN_IF_ERROR(CreateTensorIfNeeded());

  cudnnRNNDataLayout_t layout = CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED;
  float padding_fill = 0.0f;
  CUDNN_RETURN_IF_ERROR(cudnnSetRNNDataDescriptor(tensor_, dataType, layout,
                                                  static_cast<int>(max_seq_length),
                                                  static_cast<int>(batch_size),
                                                  static_cast<int>(data_size),
                                                  seq_lengths,
                                                  static_cast<void*>(&padding_fill)));
  return Status::OK();
}

CudnnFilterDescriptor::CudnnFilterDescriptor() : desc_(nullptr) {
  cudnnCreateFilterDescriptor(&desc_);
}

CudnnFilterDescriptor::~CudnnFilterDescriptor() {
  if (desc_ != nullptr) {
    cudnnDestroyFilterDescriptor(desc_);
    desc_ = nullptr;
  }
}

Status CudnnFilterDescriptor::Set(const std::vector<int64_t>& filter_dims, cudnnDataType_t data_type) {
  if (!desc_)
    CUDNN_RETURN_IF_ERROR(cudnnCreateFilterDescriptor(&desc_));

  int rank = gsl::narrow_cast<int>(filter_dims.size());
  std::vector<int> w_dims(rank);
  for (int i = 0; i < rank; i++) {
    w_dims[i] = gsl::narrow_cast<int>(filter_dims[i]);
  }

  CUDNN_RETURN_IF_ERROR(cudnnSetFilterNdDescriptor(desc_,
                                                   data_type,
                                                   CUDNN_TENSOR_NCHW,
                                                   rank,
                                                   w_dims.data()));
  return Status::OK();
}

template cudnnDataType_t CudnnTensor::GetDataType<float>();
template cudnnDataType_t CudnnTensor::GetDataType<double>();
template cudnnDataType_t CudnnTensor::GetDataType<half>();

template <>
const float Consts<float>::One = 1;

template <>
const double Consts<double>::One = 1;

template <>
const float Consts<float>::Zero = 0;

template <>
const double Consts<double>::Zero = 0;

const float Consts<half>::Zero = 0;

const float Consts<half>::One = 1;

}  // namespace cuda
}  // namespace onnxruntime
