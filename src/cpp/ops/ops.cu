#include "cuda.cuh"
#include "ops.h"

// Ops

void scatter_to_axes(const TensorView &src, const TensorView &out,
                     const std::vector<py::ssize_t> &fancy_dims_in_src,
                     const std::vector<TensorView> &fancy_dims_data,
                     const std::vector<bool> &out_axis_is_fancy,
                     const std::vector<py::ssize_t> &out_axis_target) {
  _require_cpu(src, "_C.scatter_to_axes");
  _require_cpu(out, "_C.scatter_to_axes");

  // TODO(#15) (optim): optimize with chunk memcopy
  py::ssize_t out_ndim = out.shape.size();
  py::ssize_t ind_ndim = fancy_dims_data.empty()
                             ? 0
                             : (py::ssize_t)fancy_dims_data[0].shape.size();
  py::ssize_t src_ndim = src.shape.size();
  py::ssize_t numel = numel_from_shape(src.shape);

  auto indexers_data = std::vector<const int64_t *>(fancy_dims_data.size());
  for (size_t q = 0; q < indexers_data.size(); ++q)
    indexers_data[q] =
        static_cast<const int64_t *>(fancy_dims_data[q].storage->data());

  auto fancy_axes = std::vector<py::ssize_t>(out_ndim, -1);
  for (size_t p = 0; p < fancy_dims_in_src.size(); ++p)
    fancy_axes[fancy_dims_in_src[p]] = p;

  auto loc_src = std::vector<py::ssize_t>(src_ndim);
  auto loc_out_basic = std::vector<py::ssize_t>(out_ndim);
  auto loc_out_ind = std::vector<py::ssize_t>(ind_ndim);
  DispatchAll::run(out.storage->dtype(), [&]<class T>() {
    auto out_data = static_cast<T *>(out.storage->data());
    auto src_data = static_cast<const T *>(src.storage->data());
    for (py::ssize_t idx = 0; idx < numel; ++idx) {

      // Set output loc wrt basic/fancy index combination
      for (py::ssize_t i = 0; i < src_ndim; ++i) {
        py::ssize_t src_dim = out_axis_target[i];
        if (out_axis_is_fancy[i])
          loc_out_ind[src_dim] = loc_src[i];
        else
          loc_out_basic[src_dim] = loc_src[i];
      }

      // Compute index in source storage
      py::ssize_t idx_src = src.offset;
      for (py::ssize_t i = 0; i < src_ndim; ++i)
        idx_src += src.strides[i] * loc_src[i];

      // Compute index in destination storage
      py::ssize_t idx_out = out.offset;
      for (py::ssize_t i = 0; i < out_ndim; ++i) {
        py::ssize_t p = fancy_axes[i];
        py::ssize_t src_coord_i;
        if (p == -1)
          src_coord_i = loc_out_basic[i];
        else {
          const TensorView &indexer = fancy_dims_data[p];
          py::ssize_t flat_idx = indexer.offset;
          for (size_t d = 0; d < indexer.shape.size(); ++d)
            flat_idx += indexer.strides[d] * loc_out_ind[d];
          src_coord_i = indexers_data[p][flat_idx];
        }
        idx_out += out.strides[i] * src_coord_i;
      }

      out_data[idx_out] = src_data[idx_src];

      // Carry over
      py::ssize_t i = src_ndim - 1;
      while (i >= 0) {
        loc_src[i] += 1;
        if (loc_src[i] < src.shape[i])
          break;
        loc_src[i] = 0;
        i -= 1;
      }
    }
  });
}

std::shared_ptr<Storage>
gather_from_axes(const TensorView &x, const std::vector<py::ssize_t> &new_shape,
                 const std::vector<py::ssize_t> &fancy_dims_in_src,
                 const std::vector<TensorView> &fancy_dims_data,
                 const std::vector<bool> &out_axis_is_fancy,
                 const std::vector<py::ssize_t> &out_axis_target) {
  _require_cpu(x, "_C.gather_from_axes");

  // TODO(#15) (optim): optimize with chunk memcopy
  py::ssize_t src_ndim = x.shape.size();
  py::ssize_t ind_ndim = fancy_dims_data.empty()
                             ? 0
                             : (py::ssize_t)fancy_dims_data[0].shape.size();
  py::ssize_t out_ndim = new_shape.size();
  py::ssize_t numel = numel_from_shape(new_shape);
  auto new_storage = Storage::allocate(numel, x.storage->dtype(), Device::Cpu);

  auto indexers_data = std::vector<const int64_t *>(fancy_dims_data.size());
  for (size_t q = 0; q < indexers_data.size(); ++q)
    indexers_data[q] =
        static_cast<const int64_t *>(fancy_dims_data[q].storage->data());

  auto fancy_axes = std::vector<py::ssize_t>(src_ndim, -1);
  for (size_t p = 0; p < fancy_dims_in_src.size(); ++p)
    fancy_axes[fancy_dims_in_src[p]] = p;

  auto loc_out = std::vector<py::ssize_t>(out_ndim);
  auto loc_src_basic = std::vector<py::ssize_t>(src_ndim);
  auto loc_ind = std::vector<py::ssize_t>(ind_ndim);
  return DispatchAll::run(x.storage->dtype(), [&]<class T>() {
    auto old_data = static_cast<const T *>(x.storage->data());
    auto new_data = static_cast<T *>(new_storage->data());
    for (py::ssize_t idx_out = 0; idx_out < numel; ++idx_out) {

      // Set output loc wrt basic/fancy index combination
      for (py::ssize_t i = 0; i < out_ndim; ++i) {
        py::ssize_t src_dim = out_axis_target[i];
        if (out_axis_is_fancy[i])
          loc_ind[src_dim] = loc_out[i];
        else
          loc_src_basic[src_dim] = loc_out[i];
      }

      // Compute index in source storage
      py::ssize_t idx_src = x.offset;
      for (py::ssize_t i = 0; i < src_ndim; ++i) {
        py::ssize_t p = fancy_axes[i];
        py::ssize_t src_coord_i;
        if (p == -1)
          src_coord_i = loc_src_basic[i];
        else {
          const TensorView &indexer = fancy_dims_data[p];
          py::ssize_t flat_idx = indexer.offset;
          for (size_t d = 0; d < indexer.shape.size(); ++d)
            flat_idx += indexer.strides[d] * loc_ind[d];
          src_coord_i = indexers_data[p][flat_idx];
        }
        idx_src += x.strides[i] * src_coord_i;
      }

      new_data[idx_out] = old_data[idx_src];

      // Carry over
      py::ssize_t i = out_ndim - 1;
      while (i >= 0) {
        loc_out[i] += 1;
        if (loc_out[i] < new_shape[i])
          break;
        loc_out[i] = 0;
        i -= 1;
      }
    }
    return new_storage;
  });
}

template <class T>
__global__ void _copy_kernel(py::ssize_t n, const T *ptr_in, T *ptr_out,
                             TensorViewStatic view_in,
                             TensorViewStatic view_out) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  ptr_out[unravel(i, view_out)] = ptr_in[unravel(i, view_in)];
}

template <class T>
void _cuda_copy_view(const T *ptr_in, T *ptr_out, const TensorView &view_in,
                     const TensorView &view_out) {
  auto numel = numel_from_shape(view_in.shape);
  launch_1d(numel, _copy_kernel<T>, numel, ptr_in, ptr_out,
            tensor_view_to_static(view_in), tensor_view_to_static(view_out));
}

template <class T>
void _cpu_copy_view(const T *ptr_in, T *ptr_out, const TensorView &view_in,
                    const TensorView &view_out) {
  for (py::ssize_t i = 0; i < numel_from_shape(view_in.shape); ++i)
    ptr_out[unravel(i, view_out)] = ptr_in[unravel(i, view_in)];
}

void copy_view(const TensorView &src, const TensorView &out) {
  _require_same_device(src.storage, out.storage, "copy_view");
  _require_same_dtype(src.storage, out.storage, "copy_view");
  _require_same_shape(src, out, "copy_view");
  DispatchAll::run(out.storage->dtype(), [&]<class T>() {
    auto *ptr_src = static_cast<const T *>(src.storage->data());
    auto *ptr_out = static_cast<T *>(out.storage->data());
    switch (src.storage->device()) {
    case Device::Cpu:
      _cpu_copy_view<T>(ptr_src, ptr_out, src, out);
      break;
    case Device::Cuda:
      _cuda_copy_view<T>(ptr_src, ptr_out, src, out);
      break;
    }
  });
}

std::shared_ptr<Storage> clone_contiguous_view_from(const TensorView &x) {
  auto storage_out = Storage::allocate(numel_from_shape(x.shape),
                                       x.storage->dtype(), x.storage->device());
  auto ndim = x.shape.size();
  std::vector<py::ssize_t> out_strides(ndim);
  if (ndim > 0)
    out_strides[ndim - 1] = 1;
  for (py::ssize_t j = ndim - 2; j >= 0; --j)
    out_strides[j] = x.shape[j + 1] * out_strides[j + 1];
  auto view_out = TensorView(storage_out, x.shape, out_strides, 0);
  copy_view(x, view_out);
  return storage_out;
}

void bind_ops_(py::module_ &m) {
  m.def(
      "cast", [](const Storage &s, Dtype t) { return s.cast(t); },
      "Cast a storage to another dtype.", py::arg("storage"), py::arg("dtype"));
  m.def(
      "to", [](const Storage &s, Device d) { return s.to(d); },
      "Move a storage to another device.", py::arg("storage"),
      py::arg("device"));
  m.def("copy_view", &copy_view, "Copy source view into target view.",
        py::arg("src"), py::arg("out"));
  m.def("clone_contiguous_view_from", &clone_contiguous_view_from,
        "Clone contiguous view.", py::arg("src"));
  m.def("scatter_to_axes", &scatter_to_axes,
        "Scatter elements along fancy indices.", py::arg("src"), py::arg("out"),
        py::arg("fancy_dims_in_src"), py::arg("fancy_dims_data"),
        py::arg("out_axis_is_fancy"), py::arg("out_axis_target"));
  m.def("gather_from_axes", &gather_from_axes,
        "Extract elements along fancy indices.", py::arg("x"),
        py::arg("new_shape"), py::arg("fancy_dims_in_src"),
        py::arg("fancy_dims_data"), py::arg("out_axis_is_fancy"),
        py::arg("out_axis_target"));
}