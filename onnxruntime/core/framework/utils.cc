// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/utils.h"

#include <iomanip>

#include "core/graph/graph_viewer.h"
#include "core/framework/data_transfer_manager.h"
#include "core/framework/execution_frame.h"
#include "core/framework/execution_providers.h"
#include "core/framework/feeds_fetches_manager.h"
#include "core/framework/kernel_def_builder.h"
#include "core/framework/kernel_registry_manager.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/framework/parallel_executor.h"
#include "core/framework/session_state.h"
#include "core/framework/sequential_executor.h"

namespace onnxruntime {
namespace utils {
AllocatorPtr GetAllocator(const SessionState& session_state, const OrtAllocatorInfo& allocator_info) {
  return session_state.GetExecutionProviders().GetAllocator(allocator_info);
}

common::Status AllocateHelper(const IExecutionProvider& execution_provider, int device_id, const Tensor& fetched_tensor,
                              OrtValue& output_mlvalue) {
  auto allocator = execution_provider.GetAllocator(device_id, OrtMemTypeDefault);
  if (!allocator) {
    return Status(common::ONNXRUNTIME, common::FAIL, "invalid allocator");
  }

  std::unique_ptr<Tensor> p_tensor = std::make_unique<Tensor>(fetched_tensor.DataType(),
                                                              fetched_tensor.Shape(),
                                                              allocator);
  output_mlvalue.Init(p_tensor.release(),
                      DataTypeImpl::GetType<Tensor>(),
                      DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());

  return Status::OK();
}

const std::string& GetNodeInputProviderType(const SessionState::NodeInfo& info) {
  // the input index will be std::numeric_limits<size_t>::max() if it's an implicit input to a control flow node.
  // the input will be processed fully when executing the subgraph that consumes the implicit input.
  bool implicit_input = info.index == std::numeric_limits<size_t>::max();

  // node may declare input_mem_type to be on CPU explicitly
  // skip implicit inputs as they don't have a valid 'index' value
  bool node_input_on_cpu = !implicit_input && info.kci && info.kci->kernel_def->IsInputOnCpu(info.index);

  // need a std::string that doesn't go away for kCpuExecutionProvider so we can return a reference.
  static const std::string cpu_execution_provider{onnxruntime::kCpuExecutionProvider};

  auto& required_provider_type = node_input_on_cpu ? cpu_execution_provider
                                                   : info.p_node->GetExecutionProviderType();

  return required_provider_type;
}

static Status CopyMLValue(const DataTransferManager& data_transfer_mgr,
                          const FeedsFetchesManager::MLValueCopyInfo& copy_info,
                          const OrtValue& source_mlvalue,
                          OrtValue& target_mlvalue) {
  if (copy_info.copy_provider == nullptr) {
    target_mlvalue = source_mlvalue;
  } else {
    auto& source_tensor = source_mlvalue.Get<Tensor>();

    if (!target_mlvalue.IsAllocated()) {
      ORT_RETURN_IF_ERROR(utils::AllocateHelper(*copy_info.allocation_provider, copy_info.allocation_device_id,
                                                source_tensor, target_mlvalue));
    }

    Tensor* p_output_tensor = target_mlvalue.GetMutable<Tensor>();

    ORT_RETURN_IF_ERROR(data_transfer_mgr.CopyTensor(source_tensor, *p_output_tensor));
  }

  return Status::OK();
}

// TODO should we handle the case of one input name feeding 2 nodes placed on different devices?
common::Status CopyOneInputAcrossDevices(const SessionState& session_state, const std::string& input_name,
                                         const OrtValue& orig_mlvalue, OrtValue& new_mlvalue, bool& needed_copy,
                                         FeedsFetchesManager::MLValueCopyInfo& copy_info) {
  needed_copy = false;

  //TODO: make it configurable
  const int target_device_id = 0;
  std::vector<SessionState::NodeInfo> node_info_vec;
  ORT_RETURN_IF_ERROR(session_state.GetInputNodeInfo(input_name, node_info_vec));

  auto& exec_providers = session_state.GetExecutionProviders();

  do {
    // currently we only support one device per input. see SessionState::AddInputNameToNodeInfoMapping for more
    // info on the logic to create the node_info_vec.
    // for (auto& node_info : node_info_vec) {
    auto& node_info = node_info_vec.front();
    if (node_info.p_node == nullptr) {
      // dummy entry for an input that we didn't find a use of in the graph.
      // use the input as is given we don't believe it's actually needed.
      new_mlvalue = orig_mlvalue;
      break;
    }

    if (!orig_mlvalue.IsTensor()) {
      // copying not supported for non-tensor types
      new_mlvalue = orig_mlvalue;
      break;
    }

    auto& required_provider_type = GetNodeInputProviderType(node_info);
    auto& input_tensor = orig_mlvalue.Get<Tensor>();
    auto& input_tensor_loc = input_tensor.Location();

    auto* p_input_provider = exec_providers.Get(input_tensor_loc);
    if (!p_input_provider) {
      p_input_provider = exec_providers.Get(onnxruntime::kCpuExecutionProvider);
      ORT_ENFORCE(p_input_provider);
    }

    //no copy for nGraph
    if (required_provider_type == onnxruntime::kNGraphExecutionProvider) {
      new_mlvalue = orig_mlvalue;
      break;
    }

    auto input_provider_type = p_input_provider->Type();
    if (input_provider_type == required_provider_type && input_tensor_loc.mem_type == OrtMemTypeDefault) {
      new_mlvalue = orig_mlvalue;
      break;
    }

    // If a node requires input on cpu and input tensor is allocated with pinned memory allocator, don't do copy
    if (required_provider_type == onnxruntime::kCpuExecutionProvider &&
        input_tensor_loc.mem_type == OrtMemTypeCPU) {
      new_mlvalue = orig_mlvalue;
      break;
    }

    auto* required_provider = exec_providers.Get(required_provider_type);
    ORT_ENFORCE(required_provider);

    auto* p_copy_provider = (required_provider_type != onnxruntime::kCpuExecutionProvider)
                                ? required_provider
                                : p_input_provider;

    copy_info.allocation_device_id = target_device_id;
    copy_info.allocation_provider = required_provider;
    copy_info.copy_provider = p_copy_provider;

    ORT_RETURN_IF_ERROR(CopyMLValue(session_state.GetDataTransferMgr(), copy_info, orig_mlvalue, new_mlvalue));

    needed_copy = true;

    // } loop of node_info_vec
  } while (false);

  return Status::OK();
}

common::Status CopyOneInputAcrossDevices(const SessionState& session_state, const std::string& input_name,
                                         const OrtValue& orig_mlvalue, OrtValue& new_mlvalue) {
  bool needed_copy;
  FeedsFetchesManager::MLValueCopyInfo ignored;
  return CopyOneInputAcrossDevices(session_state, input_name, orig_mlvalue, new_mlvalue, needed_copy, ignored);
}

// copies inputs across devices only if required and save copy_info
static common::Status CopyInputsAcrossDevices(const SessionState& session_state,
                                              const std::vector<std::string>& feed_names,
                                              const std::vector<OrtValue>& orig_feeds, std::vector<OrtValue>& new_feeds,
                                              bool& needed_copy,
                                              std::vector<FeedsFetchesManager::MLValueCopyInfo>* copy_info) {
  bool copied = false;
  size_t num_feeds = orig_feeds.size();
  ORT_ENFORCE(feed_names.size() == num_feeds);

  new_feeds.resize(num_feeds);
  if (copy_info) {
    copy_info->resize(num_feeds);
  }

  for (size_t idx = 0; idx < num_feeds; ++idx) {
    bool copied_this_input = false;
    FeedsFetchesManager::MLValueCopyInfo current_copy_info = {};  // init for each call
    ORT_RETURN_IF_ERROR(CopyOneInputAcrossDevices(session_state, feed_names[idx], orig_feeds[idx], new_feeds[idx],
                                                  copied_this_input, current_copy_info));

    if (copied_this_input) {
      copied = true;

      if (copy_info) {
        (*copy_info)[idx] = current_copy_info;
      }
    }
  }

  needed_copy = copied;

  return Status::OK();
}

// copies inputs across devices only if required using cached copy_info
static common::Status CachedCopyInputsAcrossDevices(
    const std::vector<OrtValue>& orig_feeds, std::vector<OrtValue>& new_feeds,
    const std::vector<FeedsFetchesManager::MLValueCopyInfo>& copy_info,
    const DataTransferManager& data_transfer_mgr) {
  size_t num_feeds = orig_feeds.size();
  ORT_ENFORCE(copy_info.size() == num_feeds);

  new_feeds.resize(num_feeds);

  for (size_t idx = 0; idx < num_feeds; ++idx) {
    ORT_RETURN_IF_ERROR(CopyMLValue(data_transfer_mgr, copy_info[idx], orig_feeds[idx], new_feeds[idx]));
  }

  return Status::OK();
}

// Setup fetches for execution. Use any provided fetches directly if the provider matches.
// If the provider doesn't match, we don't know what device the execution output may be on, so can't assume the output
// can be returned to the user directly.
// TODO: We should be able to use the allocation plan to know which device an output will be on.
static common::Status SetupFetchesForExecute(const SessionState& session_state,
                                             const std::vector<std::string>& output_names,
                                             std::vector<OrtValue>& fetches, std::vector<OrtValue>& new_fetches,
                                             std::vector<bool>* copy_to_new_fetches_cached_values) {
  ORT_ENFORCE(new_fetches.empty());

  const auto& execution_providers = session_state.GetExecutionProviders();
  auto num_outputs = output_names.size();

  new_fetches.resize(num_outputs);

  // track which fetches can be copied to new_fetches and used directly in the execution.
  std::vector<bool> local_can_copy_flags(num_outputs, false);

  std::set<std::string> seen_outputs;
  auto p_graph = session_state.GetGraphViewer();
  ORT_ENFORCE(p_graph);

  auto contains = [](const std::vector<std::string>& output_names,
                     const std::string& name) {
    auto it = std::find(std::begin(output_names), std::end(output_names), name);
    if (it == output_names.end()) {
      return std::make_pair(false, size_t(0));
    }

    return std::pair<bool, size_t>(true, it - output_names.begin());
  };

  std::pair<bool, size_t> found;
  for (auto& node : p_graph->Nodes()) {
    if (seen_outputs.size() == num_outputs) {
      break;
    }

    for (auto* arg : node.OutputDefs()) {
      if (!arg->Exists() ||
          !(found = contains(output_names, arg->Name())).first) {
        continue;
      }

      seen_outputs.insert(arg->Name());
      size_t idx = found.second;
      const OrtValue& provided_mlvalue = fetches[idx];

      if (provided_mlvalue.IsAllocated()) {
        if (!provided_mlvalue.IsTensor()) {
          new_fetches[idx] = fetches[idx];
          local_can_copy_flags[idx] = true;
          continue;
        }

        const auto& node_provider_type = node.GetExecutionProviderType();
        const auto& provided_tensor = provided_mlvalue.Get<Tensor>();
        const auto& provided_tensor_loc = provided_tensor.Location();
        const auto* tensor_provider = execution_providers.Get(provided_tensor_loc);
        if (!tensor_provider) {
          tensor_provider = execution_providers.Get(onnxruntime::kCpuExecutionProvider);
        }

        auto tensor_provider_type = tensor_provider->Type();
        if (node_provider_type == tensor_provider_type) {
          new_fetches[idx] = fetches[idx];
          local_can_copy_flags[idx] = true;
          continue;
        }

        continue;
      }
    }
  }

  if (copy_to_new_fetches_cached_values) {
    *copy_to_new_fetches_cached_values = local_can_copy_flags;
  }

  return Status::OK();
}

static common::Status CachedSetupFetchesForExecute(std::vector<OrtValue>& fetches, std::vector<OrtValue>& new_fetches,
                                                   const std::vector<bool>& copy_to_new_fetches_cached_values) {
  auto num_outputs = fetches.size();
  ORT_ENFORCE(new_fetches.empty());
  ORT_ENFORCE(copy_to_new_fetches_cached_values.size() == num_outputs);

  new_fetches.resize(num_outputs);

  // use the cached values
  for (size_t i = 0; i < num_outputs; ++i) {
    if (copy_to_new_fetches_cached_values[i]) {
      new_fetches[i] = fetches[i];
    }
  }

  return Status::OK();
}

// copies outputs across devices only if required
static common::Status CopyOutputsAcrossDevices(const SessionState& session_state, const std::vector<OrtValue>& fetches,
                                               std::vector<OrtValue>& user_fetches, bool& needed_copy,
                                               std::vector<FeedsFetchesManager::MLValueCopyInfo>* copiers) {
  needed_copy = false;
  auto num_outputs = fetches.size();

  if (copiers) {
    // resize so we have default values and only need to update an entry if there's a device copy required.
    copiers->resize(num_outputs);
  }

  auto& execution_providers = session_state.GetExecutionProviders();

  // CPU execution provider is always registered so this is not null
  const auto* cpu_execution_provider = execution_providers.Get(onnxruntime::kCpuExecutionProvider);

  for (size_t idx = 0; idx < num_outputs; ++idx) {
    auto& fetched_mlvalue = fetches[idx];
    if (!fetched_mlvalue.IsTensor()) {
      user_fetches[idx] = fetched_mlvalue;
      continue;
    }

    auto& fetched_tensor = fetched_mlvalue.Get<Tensor>();
    auto& fetched_tensor_location = fetched_tensor.Location();
    auto* p_fetched_provider = execution_providers.Get(fetched_tensor_location);
    if (!p_fetched_provider) {
      p_fetched_provider = cpu_execution_provider;
    }

    auto fetched_provider_type = p_fetched_provider->Type();
    auto& output_mlvalue = user_fetches[idx];

    const IExecutionProvider* p_output_provider = nullptr;

    if (output_mlvalue.IsAllocated()) {
      Tensor* p_output_tensor = output_mlvalue.GetMutable<Tensor>();
      p_output_provider = execution_providers.Get(p_output_tensor->Location());
    }

    if (!p_output_provider) {
      p_output_provider = cpu_execution_provider;
    }

    auto output_provider_type = p_output_provider->Type();

    if (fetched_provider_type == output_provider_type ||
        (p_output_provider == cpu_execution_provider && fetched_tensor_location.mem_type == OrtMemTypeCPUOutput)) {
      user_fetches[idx] = fetched_mlvalue;
      continue;
    }

    needed_copy = true;

    auto* p_copy_provider = (fetched_provider_type != onnxruntime::kCpuExecutionProvider)
                                ? p_fetched_provider
                                : p_output_provider;

    const int device_id = 0;  // TODO: As per comment in the copy input code, make this configurable.
    FeedsFetchesManager::MLValueCopyInfo copy_info{device_id, p_output_provider, p_copy_provider};
    ORT_RETURN_IF_ERROR(CopyMLValue(session_state.GetDataTransferMgr(), copy_info, fetched_mlvalue, output_mlvalue));

    if (copiers) {
      (*copiers)[idx] = copy_info;
    }
  }

  return Status::OK();
}

static common::Status CachedCopyOutputsAcrossDevices(
    const std::vector<OrtValue>& fetches, std::vector<OrtValue>& user_fetches,
    const std::vector<FeedsFetchesManager::MLValueCopyInfo>& copy_info,
    const DataTransferManager& data_transfer_mgr) {
  auto num_outputs = fetches.size();

  // internal logic error if these are mismatched
  ORT_ENFORCE(num_outputs == copy_info.size());

  // used the cached copy logic if available
  for (size_t idx = 0; idx < num_outputs; ++idx) {
    ORT_RETURN_IF_ERROR(CopyMLValue(data_transfer_mgr, copy_info[idx], fetches[idx], user_fetches[idx]));
  }

  return Status::OK();
}

static DeviceCopyCheck CheckExecutionProviders(const ExecutionProviders& execution_providers) {
  for (const auto& execution_provider : execution_providers) {
    if (execution_provider->Type() != onnxruntime::kCpuExecutionProvider &&
        execution_provider->Type() != onnxruntime::kMklDnnExecutionProvider &&
        execution_provider->Type() != onnxruntime::kNGraphExecutionProvider &&
        execution_provider->Type() != onnxruntime::kNupharExecutionProvider &&
        execution_provider->Type() != onnxruntime::kOpenVINOExecutionProvider) {
      return DeviceCopyCheck::Unknown;
    }
  }

  return DeviceCopyCheck::NoCopy;
}

// execute graph with cached info from FeedsFetchesManager.
common::Status ExecuteGraphWithCachedInfo(
    const SessionState& session_state, const FeedsFetchesManager& feeds_fetches_manager,
    const std::vector<OrtValue>& feeds, std::vector<OrtValue>& fetches,
    const std::unordered_map<size_t, IExecutor::CustomAllocator>& fetch_allocators, bool sequential_execution,
    const bool& terminate_flag, const logging::Logger& logger) {
  const auto& feeds_fetches_info = feeds_fetches_manager.GetFeedsFetchesInfo();
  auto device_copy_checks = feeds_fetches_manager.GetDeviceCopyChecks();

  std::unique_ptr<IExecutor> p_exec;
  if (sequential_execution) {
    p_exec = std::unique_ptr<IExecutor>(new SequentialExecutor(terminate_flag));
  } else {
    p_exec = std::unique_ptr<IExecutor>(new ParallelExecutor(session_state, terminate_flag));
  }

  if (device_copy_checks.status == DeviceCopyCheck::NoCopy) {
    // no device copies are needed so simple execute
    ORT_RETURN_IF_ERROR(p_exec->Execute(session_state,
                                        feeds_fetches_info.feeds_mlvalue_idxs, feeds,
                                        feeds_fetches_info.fetches_mlvalue_idxs, fetches, fetch_allocators, logger));
  } else {
    const std::vector<OrtValue>* p_feeds = &feeds;
    std::vector<OrtValue>* p_fetches = &fetches;
    std::vector<OrtValue> device_feeds;
    std::vector<OrtValue> device_fetches;

    // Copy inputs
    if (device_copy_checks.input_copy_needed == DeviceCopyCheck::Copy) {
      ORT_RETURN_IF_ERROR(CachedCopyInputsAcrossDevices(feeds, device_feeds,
                                                        feeds_fetches_manager.GetFeedsDeviceCopiers(),
                                                        session_state.GetDataTransferMgr()));
      p_feeds = &device_feeds;
    }

    // setup fetches.
    if (fetches.empty()) {
      fetches.resize(feeds_fetches_info.output_names.size());
    }

    // if no output copy is needed, we can just use the fetches directly. otherwise we need to use a temporary set
    // and run CopyOutputsAcrossDevices.
    if (device_copy_checks.output_copy_needed == DeviceCopyCheck::Copy) {
      ORT_RETURN_IF_ERROR(CachedSetupFetchesForExecute(fetches, device_fetches,
                                                       feeds_fetches_manager.GetCanUseFetchDuringExecutionFlags()));
      p_fetches = &device_fetches;
    }

    ORT_RETURN_IF_ERROR(p_exec->Execute(session_state,
                                        feeds_fetches_info.feeds_mlvalue_idxs, *p_feeds,
                                        feeds_fetches_info.fetches_mlvalue_idxs, *p_fetches, fetch_allocators,
                                        logger));

    if (device_copy_checks.output_copy_needed == DeviceCopyCheck::Copy) {
      ORT_RETURN_IF_ERROR(CachedCopyOutputsAcrossDevices(*p_fetches, fetches,
                                                         feeds_fetches_manager.GetFetchesDeviceCopiers(),
                                                         session_state.GetDataTransferMgr()));
    }
  }

  return Status::OK();
}

// execute graph and update feeds_fetches_manager with cached copy info if cache_copy_info is true
common::Status ExecuteGraph(const SessionState& session_state, FeedsFetchesManager& feeds_fetches_manager,
                            const std::vector<OrtValue>& feeds, std::vector<OrtValue>& fetches,
                            const std::unordered_map<size_t, IExecutor::CustomAllocator>& fetch_allocators,
                            bool sequential_execution, const bool& terminate_flag, const logging::Logger& logger,
                            bool cache_copy_info) {
  const auto& feeds_fetches_info = feeds_fetches_manager.GetFeedsFetchesInfo();
  auto device_copy_checks = feeds_fetches_manager.GetDeviceCopyChecks();

  ORT_ENFORCE(device_copy_checks.status == DeviceCopyCheck::Unknown);

  std::unique_ptr<IExecutor> p_exec;
  if (sequential_execution) {
    p_exec = std::unique_ptr<IExecutor>(new SequentialExecutor(terminate_flag));
  } else {
    p_exec = std::unique_ptr<IExecutor>(new ParallelExecutor(session_state, terminate_flag));
  }

  // see if we can skip copies due to the types of execution providers available
  if (CheckExecutionProviders(session_state.GetExecutionProviders()) == DeviceCopyCheck::NoCopy) {
    device_copy_checks.input_copy_needed = DeviceCopyCheck::NoCopy;
    device_copy_checks.output_copy_needed = DeviceCopyCheck::NoCopy;

    // no device copies are needed so simple execute
    ORT_RETURN_IF_ERROR(p_exec->Execute(session_state,
                                        feeds_fetches_info.feeds_mlvalue_idxs, feeds,
                                        feeds_fetches_info.fetches_mlvalue_idxs, fetches, fetch_allocators, logger));
  } else {
    bool copy_needed = false;

    const std::vector<OrtValue>* p_feeds = &feeds;
    std::vector<OrtValue>* p_fetches = &fetches;
    std::vector<OrtValue> device_feeds;
    std::vector<OrtValue> device_fetches;

    // Copy inputs
    auto* copiers = cache_copy_info ? &feeds_fetches_manager.GetMutableFeedsDeviceCopiers() : nullptr;
    ORT_RETURN_IF_ERROR(CopyInputsAcrossDevices(session_state,
                                                feeds_fetches_info.feed_names, feeds, device_feeds,
                                                copy_needed, copiers));

    if (copy_needed) {
      p_feeds = &device_feeds;
    }

    device_copy_checks.input_copy_needed = copy_needed ? DeviceCopyCheck::Copy
                                                       : DeviceCopyCheck::NoCopy;

    // setup fetches.
    if (fetches.empty()) {
      fetches.resize(feeds_fetches_info.output_names.size());
    }

    auto* use_provided_fetch_flags =
        cache_copy_info ? &feeds_fetches_manager.GetMutableCanUseFetchDuringExecutionFlags()
                        : nullptr;

    ORT_RETURN_IF_ERROR(SetupFetchesForExecute(session_state, feeds_fetches_info.output_names,
                                               fetches, device_fetches,
                                               use_provided_fetch_flags));
    p_fetches = &device_fetches;

    ORT_RETURN_IF_ERROR(p_exec->Execute(session_state,
                                        feeds_fetches_info.feeds_mlvalue_idxs, *p_feeds,
                                        feeds_fetches_info.fetches_mlvalue_idxs, *p_fetches, fetch_allocators,
                                        logger));

    copiers = cache_copy_info ? &feeds_fetches_manager.GetMutableFetchesDeviceCopiers() : nullptr;
    ORT_RETURN_IF_ERROR(CopyOutputsAcrossDevices(session_state, *p_fetches, fetches, copy_needed, copiers));

    device_copy_checks.output_copy_needed = copy_needed ? DeviceCopyCheck::Copy : DeviceCopyCheck::NoCopy;
  }

  // save the result of all the checks and use cached info next time
  if (cache_copy_info) {
    feeds_fetches_manager.SetDeviceCopyChecks(device_copy_checks);
  }

  return Status::OK();
}

#if defined(DEBUG_NODE_INPUTS_OUTPUTS)
std::ostream& operator<<(std::ostream& out, const BFloat16& value) {
  return out << value.ToFloat();
}

std::ostream& operator<<(std::ostream& out, const MLFloat16& value) {
  return out << value.val;
}

template <typename T>
static void DumpTensor(const Tensor& tensor, const TensorShape& shape) {
  auto num_items = shape.Size();

  if (num_items == 0) {
    std::cout << "no data";
    return;
  }

  size_t num_dims = shape.NumDimensions();
  size_t num_rows = 1;
  if (num_dims > 1) {
    num_rows = static_cast<size_t>(shape[0]);
  }

  size_t row_size = num_items / num_rows;

  auto data = tensor.DataAsSpan<T>();

  auto print_val = [](const T& value) {
    if (std::is_floating_point_v<T>)
      std::cout << std::setprecision(8) << value;
    else
      std::cout << value;
  };

  for (int row = 0; row < num_rows; ++row) {
    print_val(data[row * row_size]);
    for (int i = 1; i < row_size; ++i) {
      std::cout << ", ";
      print_val(data[row * row_size + i]);
    }
    std::cout << "\n";
  }

  std::cout << std::endl;
}

void DumpNodeInputs(const OpKernelContext& context, const Node& node) {
  std::cout << "-----------\n";
  std::cout << node.OpType() << " node: " << node.Name() << "\n";

  const auto& input_defs = node.InputDefs();

  for (auto i = 0, end = context.InputCount(); i < end; ++i) {
    if (input_defs[i]->Exists()) {
      std::cout << "Input " << i << " Name: " << input_defs[i]->Name();

      const auto* type = context.InputType(i);

      if (type) {
        if (type->IsTensorType()) {
          const auto& tensor = *context.Input<Tensor>(i);
          const auto& shape = tensor.Shape();

          std::cout << " Shape: " << shape << "\n";
        } else {
          std::cout << " is non-tensor type.\n";
        }
      } else {
        // should never happen...
        std::cout << " was missing data type\n";
      }
    } else {
      std::cout << "Input " << i << " is optional and was not provided.\n";
    }
  }
}

void DumpNodeOutputs(OpKernelContext& context, const Node& node, const SessionState& session_state) {
  std::cout << "-----------\n";
  const auto& output_defs = node.OutputDefs();

  const auto& execution_providers = session_state.GetExecutionProviders();
  const auto* cpu_execution_provider = execution_providers.Get(onnxruntime::kCpuExecutionProvider);

  for (auto i = 0, end = context.OutputCount(); i < end; ++i) {
    if (output_defs[i]->Exists()) {
      std::cout << "Output " << i << " Name: " << output_defs[i]->Name();

      const auto* type = context.OutputType(i);

      if (type) {
        if (type->IsTensorType()) {
          const auto& tensor = *context.Output<Tensor>(i);
          const auto data_type = tensor.DataType();
          const auto& shape = tensor.Shape();

          std::cout << " Shape: " << shape << "\n";

          // check tensor is on CPU before dumping it
          auto& tensor_location = tensor.Location();
          auto* provider = execution_providers.Get(tensor_location);
          if (!provider) {
            provider = cpu_execution_provider;
          }

          if (provider == cpu_execution_provider || tensor_location.mem_type == OrtMemTypeCPUOutput) {
            DispatchOnTensorType(data_type, DumpTensor, tensor, shape);
          } else {
            std::cout << " is not on CPU. Provider=" << provider->Type() << "\n";
          }
        } else {
          std::cout << " is non-tensor type.\n";
        }
      } else {
        // should never happen...
        std::cout << "missing data type\n";
      }
    } else {
      std::cout << "Output " << i << " is optional and was not produced.\n";
    }

    std::cout << std::endl;
  }
}
#endif

}  // namespace utils
}  // namespace onnxruntime
