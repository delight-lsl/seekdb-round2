/* seekdb <-> hipVS / cuVS GPU CAGRA bridge (dual AMD ROCm HIP + NVIDIA CUDA).
 * Uses VRAM-bounded C++ CAGRA NN-Descent (host dataset view + attach_dataset_on_build=false,
 * automatic fallback to C API cuvsCagraBuild), immortal RMM pool toggle (0% during build,
 * 1% pooled during search), persistent pinned/device search buffers, cached cuvsResources_t,
 * and zero-file compact graph snapshot export/import. */

#include "seekdb_cuvs_bridge.h"

#include <cuvs/core/c_api.h>
#include <cuvs/distance/distance.h>
#include <cuvs/neighbors/cagra.h>
#include <dlpack/dlpack.h>

#if defined(USE_CUVS_CPP_OPTIMIZE) && __has_include(<cuvs/neighbors/cagra.hpp>) && \
    __has_include(<raft/core/host_mdspan.hpp>)
#include <cuvs/neighbors/cagra.hpp>
#define SK_HAS_CPP_CAGRA 1
#else
#define SK_HAS_CPP_CAGRA 0
#endif

#ifdef __HIP_PLATFORM_AMD__
#include <cuvs/cuda_runtime.h>
#include <hip/hip_runtime.h>
#define GPU_SUCCESS hipSuccess
#define gpuError_t hipError_t
#define gpuGetDeviceCount hipGetDeviceCount
#define gpuMalloc hipMalloc
#define gpuFree hipFree
#define gpuMemcpy hipMemcpy
#define gpuMemcpyAsync hipMemcpyAsync
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define gpuHostAlloc(p, sz) hipHostMalloc((p), (sz), hipHostMallocDefault)
#define gpuFreeHost hipHostFree
#define gpuHostRegister(p, sz) hipHostRegister((p), (sz), hipHostRegisterDefault)
#define gpuHostUnregister hipHostUnregister
#else
#include <cuda_runtime.h>
#define GPU_SUCCESS cudaSuccess
#define gpuError_t cudaError_t
#define gpuGetDeviceCount cudaGetDeviceCount
#define gpuMalloc cudaMalloc
#define gpuFree cudaFree
#define gpuMemcpy cudaMemcpy
#define gpuMemcpyAsync cudaMemcpyAsync
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define gpuHostAlloc(p, sz) cudaHostAlloc((p), (sz), cudaHostAllocDefault)
#define gpuFreeHost cudaFreeHost
#define gpuHostRegister(p, sz) cudaHostRegister((p), (sz), cudaHostRegisterDefault)
#define gpuHostUnregister cudaHostUnregister
#endif

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

namespace {

static pthread_once_t g_rmm_once = PTHREAD_ONCE_INIT;
static pthread_mutex_t g_rmm_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_rmm_cv = PTHREAD_COND_INITIALIZER;
static pthread_cond_t g_rmm_cmd_cv = PTHREAD_COND_INITIALIZER;
static pthread_mutex_t g_global_op_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t g_res_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t g_warmup_mu = PTHREAD_MUTEX_INITIALIZER;
static cuvsResources_t g_cached_res = 0;
static bool g_warmup_done = false;
static bool g_rmm_init_done = false;
static bool g_rmm_ok = false;
static int g_rmm_target_state = 0;  // 0 = raw device alloc, 1 = 1% RMM pool
static int g_rmm_cur_state = 0;
static uint64_t g_rmm_req_seq = 0;
static uint64_t g_rmm_ack_seq = 0;

static void* rmm_keeper_main(void*) {
  pthread_mutex_lock(&g_rmm_mu);
  g_rmm_ok = true;
  g_rmm_init_done = true;
  pthread_cond_broadcast(&g_rmm_cv);
  while (true) {
    while (g_rmm_req_seq == g_rmm_ack_seq) {
      pthread_cond_wait(&g_rmm_cmd_cv, &g_rmm_mu);
    }
    int want = g_rmm_target_state;
    uint64_t seq = g_rmm_req_seq;
    pthread_mutex_unlock(&g_rmm_mu);

    if (want != g_rmm_cur_state) {
      if (want == 0) {
        cuvsRMMMemoryResourceReset();
        g_rmm_cur_state = 0;
      } else {
        if (cuvsRMMPoolMemoryResourceEnable(0, 1, false) == CUVS_SUCCESS) {
          g_rmm_cur_state = 1;
        }
      }
    }

    pthread_mutex_lock(&g_rmm_mu);
    g_rmm_ack_seq = seq;
    pthread_cond_broadcast(&g_rmm_cv);
  }
  return nullptr;
}

static void rmm_set_pool(bool enable) {
  pthread_mutex_lock(&g_rmm_mu);
  if (!g_rmm_ok) {
    pthread_mutex_unlock(&g_rmm_mu);
    return;
  }
  int want = enable ? 1 : 0;
  if (g_rmm_cur_state == want && g_rmm_req_seq == g_rmm_ack_seq) {
    pthread_mutex_unlock(&g_rmm_mu);
    return;
  }
  g_rmm_target_state = want;
  uint64_t my_seq = ++g_rmm_req_seq;
  pthread_cond_signal(&g_rmm_cmd_cv);
  while (g_rmm_ack_seq < my_seq) {
    pthread_cond_wait(&g_rmm_cv, &g_rmm_mu);
  }
  pthread_mutex_unlock(&g_rmm_mu);
}

static void init_rmm_pool_once() {
  int dev_cnt = 0;
  if (gpuGetDeviceCount(&dev_cnt) != GPU_SUCCESS || dev_cnt <= 0) {
    g_rmm_ok = false;
    g_rmm_init_done = true;
    return;
  }
  pthread_t tid;
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
  pthread_attr_setstacksize(&attr, 8UL * 1024 * 1024);
  if (pthread_create(&tid, &attr, rmm_keeper_main, nullptr) == 0) {
    pthread_mutex_lock(&g_rmm_mu);
    while (!g_rmm_init_done) {
      pthread_cond_wait(&g_rmm_cv, &g_rmm_mu);
    }
    pthread_mutex_unlock(&g_rmm_mu);
  } else {
    g_rmm_init_done = true;
    g_rmm_ok = false;
  }
  pthread_attr_destroy(&attr);
}

static inline bool ensure_gpu_ready() {
  int dev_cnt = 0;
  if (gpuGetDeviceCount(&dev_cnt) != GPU_SUCCESS || dev_cnt <= 0) {
    return false;
  }
  pthread_once(&g_rmm_once, init_rmm_pool_once);
  return true;
}

static cuvsError_t acquire_cuvs_res(cuvsResources_t* out_res) {
  pthread_mutex_lock(&g_res_mu);
  if (g_cached_res != 0) {
    *out_res = g_cached_res;
    g_cached_res = 0;
    pthread_mutex_unlock(&g_res_mu);
    return CUVS_SUCCESS;
  }
  pthread_mutex_unlock(&g_res_mu);
  return cuvsResourcesCreate(out_res);
}

static void release_cuvs_res(cuvsResources_t res) {
  if (res == 0) return;
  cuvsStreamSync(res);
  pthread_mutex_lock(&g_res_mu);
  if (g_cached_res == 0) {
    g_cached_res = res;
    pthread_mutex_unlock(&g_res_mu);
    return;
  }
  pthread_mutex_unlock(&g_res_mu);
  cuvsResourcesDestroy(res);
}

static int env_int(const char* name, int def_val) {
  const char* s = getenv(name);
  if (!s || !s[0]) return def_val;
  return atoi(s);
}

static constexpr long FAST_MAX_NQ = 16;
static constexpr long FAST_MAX_TOPK = 256;

struct sk_cuvs_index {
  cuvsResources_t res;
  cuvsCagraIndexParams_t ip;
  cuvsCagraIndex_t idx;
  cuvsCagraSearchParams_t sp;
  cudaStream_t stream;
  long n;
  long dim;
  long degree;

  pthread_mutex_t mu;
  bool mu_init;

  float* d_owned_dataset;
  uint32_t* d_owned_graph;
  float* d_q;
  uint32_t* d_ids;
  float* d_dist;
  float* h_q;
  uint32_t* h_ids;
  float* h_dist;
};

static bool init_search_buffers(sk_cuvs_index* h) {
  pthread_mutex_init(&h->mu, nullptr);
  h->mu_init = true;
  if (cuvsStreamGet(h->res, &h->stream) != CUVS_SUCCESS) return false;
  if (cuvsCagraSearchParamsCreate(&h->sp) != CUVS_SUCCESS) return false;

  size_t q_bytes = sizeof(float) * (size_t)FAST_MAX_NQ * (size_t)h->dim;
  size_t id_bytes = sizeof(uint32_t) * (size_t)FAST_MAX_NQ * (size_t)FAST_MAX_TOPK;
  size_t dst_bytes = sizeof(float) * (size_t)FAST_MAX_NQ * (size_t)FAST_MAX_TOPK;

  if (gpuMalloc((void**)&h->d_q, q_bytes) != GPU_SUCCESS) return false;
  if (gpuMalloc((void**)&h->d_ids, id_bytes) != GPU_SUCCESS) return false;
  if (gpuMalloc((void**)&h->d_dist, dst_bytes) != GPU_SUCCESS) return false;

  if (gpuHostAlloc((void**)&h->h_q, q_bytes) != GPU_SUCCESS) return false;
  if (gpuHostAlloc((void**)&h->h_ids, id_bytes) != GPU_SUCCESS) return false;
  if (gpuHostAlloc((void**)&h->h_dist, dst_bytes) != GPU_SUCCESS) return false;
  return true;
}

static void destroy_handle(sk_cuvs_index* h) {
  if (!h) return;
  if (h->h_dist) gpuFreeHost(h->h_dist);
  if (h->h_ids) gpuFreeHost(h->h_ids);
  if (h->h_q) gpuFreeHost(h->h_q);
  if (h->d_dist) gpuFree(h->d_dist);
  if (h->d_ids) gpuFree(h->d_ids);
  if (h->d_q) gpuFree(h->d_q);
  if (h->sp) cuvsCagraSearchParamsDestroy(h->sp);
  if (h->idx) cuvsCagraIndexDestroy(h->idx);
  if (h->d_owned_graph) gpuFree(h->d_owned_graph);
  if (h->d_owned_dataset) gpuFree(h->d_owned_dataset);
  if (h->ip) cuvsCagraIndexParamsDestroy(h->ip);
  if (h->res) release_cuvs_res(h->res);
  if (h->mu_init) pthread_mutex_destroy(&h->mu);
  free(h);
}

static inline size_t round_cagra_itopk(size_t req) {
  if (req <= 64) return 64;
  if (req <= 128) return 128;
  if (req <= 256) return 256;
  if (req <= 512) return 512;
  return (req + 63) & ~((size_t)63);
}

static inline void configure_search_params(cuvsCagraSearchParams_t sp, long dim, long topk, long nq) {
  (void)nq;
  int algo_env = env_int("SEEKDB_CUVS_ALGO", -1);
  if (algo_env == 0) sp->algo = SINGLE_CTA;
  else if (algo_env == 1) sp->algo = MULTI_CTA;
  else sp->algo = AUTO;

  sp->max_queries = 0;
  sp->max_iterations = 0;
  sp->min_iterations = 0;
  sp->team_size = 0;
  sp->thread_block_size = 0;
  sp->hashmap_mode = AUTO_HASH;

  if (dim <= 128) {
    int base_itopk = env_int("SEEKDB_CUVS_ITOPK_128", 256);
    int sw = env_int("SEEKDB_CUVS_SW_128", 1);
    size_t req = (size_t)topk * 2;
    if (req < (size_t)base_itopk) req = (size_t)base_itopk;
    sp->itopk_size = round_cagra_itopk(req);
    sp->search_width = (size_t)sw;
  } else {
    int base_itopk = env_int("SEEKDB_CUVS_ITOPK_256", 704);
    int sw = env_int("SEEKDB_CUVS_SW_256", 4);
    size_t req = (size_t)topk * 4;
    if (req < (size_t)base_itopk) req = (size_t)base_itopk;
    sp->itopk_size = round_cagra_itopk(req);
    sp->search_width = (size_t)sw;
  }
}

#if SK_HAS_CPP_CAGRA
static bool build_cpp_cagra_nnd(sk_cuvs_index* h, const float* base, long n, long dim) {
  if (n < 256 || dim <= 0) return false;
  size_t default_deg = (n < 1024) ? 32 : ((dim <= 128) ? 64 : 128);
  size_t default_ideg = (n < 1024) ? 64 : ((dim <= 128) ? 72 : 128);
  size_t default_nnd_ideg = (n < 1024) ? 64 : ((dim <= 128) ? 96 : 128);
  size_t default_niter = (n < 1024) ? 2 : ((dim <= 128) ? 7 : 7);

  size_t deg = (size_t)env_int("SEEKDB_CUVS_DEG", (int)default_deg);
  if ((long)deg >= n) deg = (size_t)((n > 2) ? (n / 2) : 1);
  size_t ideg = (size_t)env_int("SEEKDB_CUVS_IDEG", (int)default_ideg);
  if (ideg < deg) ideg = deg;
  if ((long)ideg >= n) ideg = deg;
  size_t nnd_ideg = (size_t)env_int("SEEKDB_CUVS_NND_IDEG", (int)default_nnd_ideg);
  if (nnd_ideg < ideg) nnd_ideg = ideg;
  if ((long)nnd_ideg >= n) nnd_ideg = ideg;
  size_t niter = (size_t)env_int("SEEKDB_CUVS_NITER", (int)default_niter);
  bool mst = (env_int("SEEKDB_CUVS_MST", 0) != 0);

  uint32_t* d_graph = nullptr;
  float* d_dataset = nullptr;
  size_t d_bytes = sizeof(float) * (size_t)n * (size_t)dim;
  bool pin = (gpuHostRegister((void*)base, d_bytes) == GPU_SUCCESS);
  try {
    auto* res_ptr = reinterpret_cast<const raft::resources*>(h->res);
    cuvs::neighbors::cagra::index_params cparams;
    cparams.metric = cuvs::distance::DistanceType::L2Expanded;
    cparams.graph_degree = deg;
    cparams.intermediate_graph_degree = ideg;
    cparams.guarantee_connectivity = mst;
    cparams.attach_dataset_on_build = false;

    cuvs::neighbors::nn_descent::index_params nparams(ideg, cuvs::distance::DistanceType::L2Expanded);
    nparams.intermediate_graph_degree = nnd_ideg;
    nparams.max_iterations = niter;
    nparams.termination_threshold = 0.0002f;
    nparams.return_distances = false;
    cparams.graph_build_params = nparams;

    size_t g_bytes = sizeof(uint32_t) * (size_t)n * deg;
    {
      auto host_view = raft::make_host_matrix_view<const float, int64_t>(base, (int64_t)n, (int64_t)dim);
      auto cpp_idx = cuvs::neighbors::cagra::build(*res_ptr, cparams, host_view);
      auto g_view = cpp_idx.graph();
      if (g_view.extent(0) != n || (size_t)g_view.extent(1) != deg) {
        if (pin) gpuHostUnregister((void*)base);
        return false;
      }
      if (gpuMalloc((void**)&d_graph, g_bytes) != GPU_SUCCESS) {
        if (pin) gpuHostUnregister((void*)base);
        return false;
      }
      if (gpuMemcpyAsync(d_graph, g_view.data_handle(), g_bytes, gpuMemcpyDeviceToDevice, h->stream) != GPU_SUCCESS) {
        gpuFree(d_graph);
        if (pin) gpuHostUnregister((void*)base);
        return false;
      }
      cuvsStreamSync(h->res);
    }

    if (gpuMalloc((void**)&d_dataset, d_bytes) != GPU_SUCCESS) {
      gpuFree(d_graph);
      if (pin) gpuHostUnregister((void*)base);
      return false;
    }
    gpuError_t cp = gpuMemcpyAsync(d_dataset, base, d_bytes, gpuMemcpyHostToDevice, h->stream);
    cuvsStreamSync(h->res);
    if (pin) {
      gpuHostUnregister((void*)base);
      pin = false;
    }
    if (cp != GPU_SUCCESS) {
      gpuFree(d_dataset);
      gpuFree(d_graph);
      return false;
    }

    DLManagedTensor graph_dl{};
    int64_t g_shape[2] = {n, (int64_t)deg};
    graph_dl.dl_tensor.data = d_graph;
    graph_dl.dl_tensor.device.device_type = kDLCUDA;
    graph_dl.dl_tensor.device.device_id = 0;
    graph_dl.dl_tensor.ndim = 2;
    graph_dl.dl_tensor.dtype.code = kDLUInt;
    graph_dl.dl_tensor.dtype.bits = 32;
    graph_dl.dl_tensor.dtype.lanes = 1;
    graph_dl.dl_tensor.shape = g_shape;

    DLManagedTensor dset_dl{};
    int64_t d_shape[2] = {n, dim};
    dset_dl.dl_tensor.data = d_dataset;
    dset_dl.dl_tensor.device.device_type = kDLCUDA;
    dset_dl.dl_tensor.device.device_id = 0;
    dset_dl.dl_tensor.ndim = 2;
    dset_dl.dl_tensor.dtype.code = kDLFloat;
    dset_dl.dl_tensor.dtype.bits = 32;
    dset_dl.dl_tensor.dtype.lanes = 1;
    dset_dl.dl_tensor.shape = d_shape;

    if (cuvsCagraIndexFromArgs(h->res, L2Expanded, &graph_dl, &dset_dl, h->idx) != CUVS_SUCCESS) {
      gpuFree(d_dataset);
      gpuFree(d_graph);
      return false;
    }
    cuvsStreamSync(h->res);
    h->d_owned_dataset = d_dataset;
    h->d_owned_graph = d_graph;
    h->degree = (long)deg;
    return true;
  } catch (...) {
    if (pin) gpuHostUnregister((void*)base);
    if (d_dataset) gpuFree(d_dataset);
    if (d_graph) gpuFree(d_graph);
    return false;
  }
}
#endif

}  // namespace

extern "C" void* seekdb_cuvs_build(const float* base, long n, long dim) {
  if (!base || n <= 0 || dim <= 0) return nullptr;
  if (!ensure_gpu_ready()) return nullptr;

  pthread_mutex_lock(&g_global_op_mu);
  rmm_set_pool(false);

  sk_cuvs_index* h = (sk_cuvs_index*)calloc(1, sizeof(*h));
  if (!h) {
    rmm_set_pool(true);
    pthread_mutex_unlock(&g_global_op_mu);
    return nullptr;
  }
  h->n = n;
  h->dim = dim;

  if (acquire_cuvs_res(&h->res) != CUVS_SUCCESS ||
      cuvsCagraIndexParamsCreate(&h->ip) != CUVS_SUCCESS ||
      cuvsCagraIndexCreate(&h->idx) != CUVS_SUCCESS ||
      !init_search_buffers(h)) {
    destroy_handle(h);
    rmm_set_pool(true);
    pthread_mutex_unlock(&g_global_op_mu);
    return nullptr;
  }

  int mode = env_int("SEEKDB_CUVS_FORCE_NND", 1);
#if SK_HAS_CPP_CAGRA
  if (mode == 1 && build_cpp_cagra_nnd(h, base, n, dim)) {
    rmm_set_pool(true);
    pthread_mutex_unlock(&g_global_op_mu);
    return h;
  }
#endif

  DLManagedTensor dset{};
  int64_t ds[2] = {n, dim};
  dset.dl_tensor.data = (void*)base;
  dset.dl_tensor.device.device_type = kDLCPU;
  dset.dl_tensor.device.device_id = 0;
  dset.dl_tensor.ndim = 2;
  dset.dl_tensor.dtype.code = kDLFloat;
  dset.dl_tensor.dtype.bits = 32;
  dset.dl_tensor.dtype.lanes = 1;
  dset.dl_tensor.shape = ds;

  h->ip->build_algo = NN_DESCENT;
  size_t deg = (size_t)env_int("SEEKDB_CUVS_DEG", (n <= 128) ? 32 : ((dim <= 128) ? 64 : 128));
  if ((long)deg >= n) deg = (size_t)((n > 2) ? (n / 2) : 1);
  size_t ideg = (size_t)env_int("SEEKDB_CUVS_IDEG", (int)((dim <= 128) ? 72 : 128));
  if (ideg < deg) ideg = deg;
  if (ideg >= (size_t)n) ideg = deg;
  h->ip->graph_degree = deg;
  h->ip->intermediate_graph_degree = ideg;
  h->ip->nn_descent_niter = (size_t)env_int("SEEKDB_CUVS_NITER", 8);

  cuvsError_t st = cuvsCagraBuild(h->res, h->ip, &dset, h->idx);
  if (st != CUVS_SUCCESS) {
    fprintf(stderr, "[seekdb_cuvs] build failed: %s\n", cuvsGetLastErrorText());
    destroy_handle(h);
    rmm_set_pool(true);
    pthread_mutex_unlock(&g_global_op_mu);
    return nullptr;
  }
  int64_t gdeg = 0;
  if (cuvsCagraIndexGetGraphDegree(h->idx, &gdeg) == CUVS_SUCCESS) h->degree = (long)gdeg;
  rmm_set_pool(true);
  pthread_mutex_unlock(&g_global_op_mu);
  return h;
}

extern "C" int seekdb_cuvs_search(void* handle, const float* query, long nq, long topk,
                                  unsigned int* out_ids, float* out_dist) {
  sk_cuvs_index* h = (sk_cuvs_index*)handle;
  if (!h || !query || nq <= 0 || topk <= 0 || !out_ids) return 1;

  pthread_mutex_lock(&h->mu);
  int rc = 0;
  cuvsFilter no_filter{0, NO_FILTER};
  configure_search_params(h->sp, h->dim, topk, nq);

  if (nq <= FAST_MAX_NQ && topk <= FAST_MAX_TOPK) {
    size_t q_bytes = sizeof(float) * (size_t)nq * (size_t)h->dim;
    size_t id_bytes = sizeof(uint32_t) * (size_t)nq * (size_t)topk;
    size_t dst_bytes = sizeof(float) * (size_t)nq * (size_t)topk;

    memcpy(h->h_q, query, q_bytes);
    gpuError_t me = gpuMemcpyAsync(h->d_q, h->h_q, q_bytes, gpuMemcpyHostToDevice, h->stream);
    if (me != GPU_SUCCESS) {
      fprintf(stderr, "[seekdb_cuvs] H2D q err=%d\n", (int)me);
      pthread_mutex_unlock(&h->mu);
      return 13;
    }

    int64_t qs[2] = {nq, h->dim};
    int64_t ns[2] = {nq, topk};
    DLManagedTensor qt{}, nt{}, dt{};
    qt.dl_tensor.data = h->d_q;
    qt.dl_tensor.device.device_type = kDLCUDA;
    qt.dl_tensor.device.device_id = 0;
    qt.dl_tensor.ndim = 2;
    qt.dl_tensor.dtype.code = kDLFloat;
    qt.dl_tensor.dtype.bits = 32;
    qt.dl_tensor.dtype.lanes = 1;
    qt.dl_tensor.shape = qs;

    nt = qt;
    nt.dl_tensor.data = h->d_ids;
    nt.dl_tensor.dtype.code = kDLUInt;
    nt.dl_tensor.shape = ns;

    dt = qt;
    dt.dl_tensor.data = h->d_dist;
    dt.dl_tensor.dtype.code = kDLFloat;
    dt.dl_tensor.shape = ns;

    if (cuvsCagraSearch(h->res, h->sp, h->idx, &qt, &nt, &dt, no_filter) != CUVS_SUCCESS) {
      fprintf(stderr, "[seekdb_cuvs] search failed: %s\n", cuvsGetLastErrorText());
      pthread_mutex_unlock(&h->mu);
      return 15;
    }
    if (gpuMemcpyAsync(h->h_ids, h->d_ids, id_bytes, gpuMemcpyDeviceToHost, h->stream) != GPU_SUCCESS) {
      pthread_mutex_unlock(&h->mu);
      return 16;
    }
    if (out_dist) {
      if (gpuMemcpyAsync(h->h_dist, h->d_dist, dst_bytes, gpuMemcpyDeviceToHost, h->stream) != GPU_SUCCESS) {
        pthread_mutex_unlock(&h->mu);
        return 17;
      }
    }
    cuvsStreamSync(h->res);
    memcpy(out_ids, h->h_ids, id_bytes);
    if (out_dist) memcpy(out_dist, h->h_dist, dst_bytes);
    pthread_mutex_unlock(&h->mu);
    return 0;
  }

  float* qd = nullptr;
  uint32_t* nd = nullptr;
  float* dd = nullptr;
  if (cuvsRMMAlloc(h->res, (void**)&qd, sizeof(float) * (size_t)nq * (size_t)h->dim) != CUVS_SUCCESS) { rc = 10; goto done_batch; }
  if (cuvsRMMAlloc(h->res, (void**)&nd, sizeof(uint32_t) * (size_t)nq * (size_t)topk) != CUVS_SUCCESS) { rc = 11; goto done_batch; }
  if (cuvsRMMAlloc(h->res, (void**)&dd, sizeof(float) * (size_t)nq * (size_t)topk) != CUVS_SUCCESS) { rc = 12; goto done_batch; }
  if (gpuMemcpyAsync(qd, query, sizeof(float) * (size_t)nq * (size_t)h->dim, gpuMemcpyHostToDevice, h->stream) != GPU_SUCCESS) { rc = 13; goto done_batch; }
  {
    int64_t qs[2] = {nq, h->dim};
    int64_t ns[2] = {nq, topk};
    DLManagedTensor qt{}, nt{}, dt{};
    qt.dl_tensor.data = qd;
    qt.dl_tensor.device.device_type = kDLCUDA;
    qt.dl_tensor.device.device_id = 0;
    qt.dl_tensor.ndim = 2;
    qt.dl_tensor.dtype.code = kDLFloat;
    qt.dl_tensor.dtype.bits = 32;
    qt.dl_tensor.dtype.lanes = 1;
    qt.dl_tensor.shape = qs;
    nt = qt; nt.dl_tensor.data = nd; nt.dl_tensor.dtype.code = kDLUInt; nt.dl_tensor.shape = ns;
    dt = qt; dt.dl_tensor.data = dd; dt.dl_tensor.dtype.code = kDLFloat; dt.dl_tensor.shape = ns;
    if (cuvsCagraSearch(h->res, h->sp, h->idx, &qt, &nt, &dt, no_filter) != CUVS_SUCCESS) { rc = 15; goto done_batch; }
    if (gpuMemcpyAsync(out_ids, nd, sizeof(uint32_t) * (size_t)nq * (size_t)topk, gpuMemcpyDeviceToHost, h->stream) != GPU_SUCCESS) { rc = 16; goto done_batch; }
    if (out_dist) {
      gpuMemcpyAsync(out_dist, dd, sizeof(float) * (size_t)nq * (size_t)topk, gpuMemcpyDeviceToHost, h->stream);
    }
    cuvsStreamSync(h->res);
  }
done_batch:
  if (dd) cuvsRMMFree(h->res, dd, sizeof(float) * (size_t)nq * (size_t)topk);
  if (nd) cuvsRMMFree(h->res, nd, sizeof(uint32_t) * (size_t)nq * (size_t)topk);
  if (qd) cuvsRMMFree(h->res, qd, sizeof(float) * (size_t)nq * (size_t)h->dim);
  pthread_mutex_unlock(&h->mu);
  if (rc) fprintf(stderr, "[seekdb_cuvs] batch search rc=%d: %s\n", rc, cuvsGetLastErrorText());
  return rc;
}

extern "C" void seekdb_cuvs_free(void* handle) {
  destroy_handle((sk_cuvs_index*)handle);
}

extern "C" long seekdb_cuvs_get_graph_degree(void* handle) {
  sk_cuvs_index* h = (sk_cuvs_index*)handle;
  if (!h || !h->idx) return 0;
  int64_t deg = 0;
  if (cuvsCagraIndexGetGraphDegree(h->idx, &deg) != CUVS_SUCCESS || deg <= 0) return 0;
  return (long)deg;
}

extern "C" int seekdb_cuvs_export_graph(void* handle, unsigned int* out_graph, unsigned long bytes) {
  sk_cuvs_index* h = (sk_cuvs_index*)handle;
  if (!h || !h->idx || !out_graph) return 1;
  pthread_mutex_lock(&h->mu);
  int64_t n = 0, deg = 0;
  if (cuvsCagraIndexGetSize(h->idx, &n) != CUVS_SUCCESS ||
      cuvsCagraIndexGetGraphDegree(h->idx, &deg) != CUVS_SUCCESS ||
      n <= 0 || deg <= 0 || (unsigned long)n * (unsigned long)deg * sizeof(uint32_t) != bytes) {
    pthread_mutex_unlock(&h->mu);
    return 2;
  }
  DLManagedTensor g_view{};
  if (cuvsCagraIndexGetGraph(h->idx, &g_view) != CUVS_SUCCESS) {
    pthread_mutex_unlock(&h->mu);
    return 3;
  }
  gpuError_t cp_st = GPU_SUCCESS;
  const DLDeviceType dt = g_view.dl_tensor.device.device_type;
  if (dt == kDLCPU || dt == kDLCUDAHost || dt == kDLROCMHost) {
    memcpy(out_graph, g_view.dl_tensor.data, bytes);
  } else {
    bool pin = (gpuHostRegister((void*)out_graph, bytes) == GPU_SUCCESS);
    cp_st = gpuMemcpyAsync(out_graph, g_view.dl_tensor.data, bytes, gpuMemcpyDeviceToHost, h->stream);
    cuvsStreamSync(h->res);
    if (pin) gpuHostUnregister((void*)out_graph);
  }
  if (g_view.deleter) g_view.deleter(&g_view);
  pthread_mutex_unlock(&h->mu);
  return (cp_st == GPU_SUCCESS) ? 0 : 4;
}

extern "C" void* seekdb_cuvs_from_graph(const float* base, long n, long dim,
                                        const unsigned int* graph, long degree) {
  if (!base || !graph || n <= 0 || dim <= 0 || degree <= 0) return nullptr;
  if (!ensure_gpu_ready()) return nullptr;

  sk_cuvs_index* h = (sk_cuvs_index*)calloc(1, sizeof(*h));
  if (!h) return nullptr;
  h->n = n;
  h->dim = dim;
  h->degree = degree;

  if (acquire_cuvs_res(&h->res) != CUVS_SUCCESS) { destroy_handle(h); return nullptr; }
  if (cuvsCagraIndexParamsCreate(&h->ip) != CUVS_SUCCESS) { destroy_handle(h); return nullptr; }
  if (cuvsCagraIndexCreate(&h->idx) != CUVS_SUCCESS) { destroy_handle(h); return nullptr; }
  if (!init_search_buffers(h)) { destroy_handle(h); return nullptr; }

  size_t d_bytes = sizeof(float) * (size_t)n * (size_t)dim;
  size_t g_bytes = sizeof(uint32_t) * (size_t)n * (size_t)degree;
  if (gpuMalloc((void**)&h->d_owned_dataset, d_bytes) != GPU_SUCCESS ||
      gpuMalloc((void**)&h->d_owned_graph, g_bytes) != GPU_SUCCESS) {
    destroy_handle(h);
    return nullptr;
  }

  bool b_pin = (gpuHostRegister((void*)base, d_bytes) == GPU_SUCCESS);
  bool g_pin = (gpuHostRegister((void*)graph, g_bytes) == GPU_SUCCESS);

  if (gpuMemcpyAsync(h->d_owned_dataset, base, d_bytes, gpuMemcpyHostToDevice, h->stream) != GPU_SUCCESS ||
      gpuMemcpyAsync(h->d_owned_graph, graph, g_bytes, gpuMemcpyHostToDevice, h->stream) != GPU_SUCCESS) {
    if (g_pin) gpuHostUnregister((void*)graph);
    if (b_pin) gpuHostUnregister((void*)base);
    destroy_handle(h);
    return nullptr;
  }

  DLManagedTensor graph_dl{};
  int64_t g_shape[2] = {n, degree};
  graph_dl.dl_tensor.data = h->d_owned_graph;
  graph_dl.dl_tensor.device.device_type = kDLCUDA;
  graph_dl.dl_tensor.device.device_id = 0;
  graph_dl.dl_tensor.ndim = 2;
  graph_dl.dl_tensor.dtype.code = kDLUInt;
  graph_dl.dl_tensor.dtype.bits = 32;
  graph_dl.dl_tensor.dtype.lanes = 1;
  graph_dl.dl_tensor.shape = g_shape;

  DLManagedTensor dset_dl{};
  int64_t d_shape[2] = {n, dim};
  dset_dl.dl_tensor.data = h->d_owned_dataset;
  dset_dl.dl_tensor.device.device_type = kDLCUDA;
  dset_dl.dl_tensor.device.device_id = 0;
  dset_dl.dl_tensor.ndim = 2;
  dset_dl.dl_tensor.dtype.code = kDLFloat;
  dset_dl.dl_tensor.dtype.bits = 32;
  dset_dl.dl_tensor.dtype.lanes = 1;
  dset_dl.dl_tensor.shape = d_shape;

  cuvsError_t st = cuvsCagraIndexFromArgs(h->res, L2Expanded, &graph_dl, &dset_dl, h->idx);
  cuvsStreamSync(h->res);
  if (g_pin) gpuHostUnregister((void*)graph);
  if (b_pin) gpuHostUnregister((void*)base);

  if (st != CUVS_SUCCESS) {
    fprintf(stderr, "[seekdb_cuvs] from_graph failed: %s\n", cuvsGetLastErrorText());
    destroy_handle(h);
    return nullptr;
  }
  pthread_mutex_lock(&g_global_op_mu);
  rmm_set_pool(true);
  pthread_mutex_unlock(&g_global_op_mu);
  return h;
}

extern "C" int seekdb_cuvs_serialize(void* handle, const char* filename) {
  sk_cuvs_index* h = (sk_cuvs_index*)handle;
  if (!h || !filename) return 1;
  pthread_mutex_lock(&h->mu);
  cuvsError_t st = cuvsCagraSerialize(h->res, filename, h->idx, true);
  pthread_mutex_unlock(&h->mu);
  if (st != CUVS_SUCCESS) {
    fprintf(stderr, "[seekdb_cuvs] serialize failed: %s\n", cuvsGetLastErrorText());
    return 1;
  }
  return 0;
}

extern "C" void* seekdb_cuvs_deserialize(const char* filename, long n, long dim) {
  if (!filename || n <= 0 || dim <= 0) return nullptr;
  if (!ensure_gpu_ready()) return nullptr;
  pthread_mutex_lock(&g_global_op_mu);
  rmm_set_pool(false);
  sk_cuvs_index* h = (sk_cuvs_index*)calloc(1, sizeof(*h));
  if (!h) {
    rmm_set_pool(true);
    pthread_mutex_unlock(&g_global_op_mu);
    return nullptr;
  }
  h->n = n;
  h->dim = dim;

  if (acquire_cuvs_res(&h->res) != CUVS_SUCCESS) goto fail;
  if (cuvsCagraIndexParamsCreate(&h->ip) != CUVS_SUCCESS) goto fail;
  if (cuvsCagraIndexCreate(&h->idx) != CUVS_SUCCESS) goto fail;
  if (!init_search_buffers(h)) goto fail;
  if (cuvsCagraDeserialize(h->res, filename, h->idx) != CUVS_SUCCESS) goto fail;
  {
    int64_t actual_n = 0, actual_dim = 0, gdeg = 0;
    if (cuvsCagraIndexGetSize(h->idx, &actual_n) != CUVS_SUCCESS ||
        cuvsCagraIndexGetDims(h->idx, &actual_dim) != CUVS_SUCCESS ||
        actual_n != n || actual_dim != dim) {
      fprintf(stderr, "[seekdb_cuvs] snapshot shape mismatch: %ld x %ld, expected %ld x %ld\n",
              (long)actual_n, (long)actual_dim, n, dim);
      goto fail;
    }
    if (cuvsCagraIndexGetGraphDegree(h->idx, &gdeg) == CUVS_SUCCESS) h->degree = (long)gdeg;
  }
  rmm_set_pool(true);
  pthread_mutex_unlock(&g_global_op_mu);
  return h;
fail:
  fprintf(stderr, "[seekdb_cuvs] deserialize failed: %s\n", cuvsGetLastErrorText());
  destroy_handle(h);
  rmm_set_pool(true);
  pthread_mutex_unlock(&g_global_op_mu);
  return nullptr;
}

extern "C" int seekdb_cuvs_cagra_knn(const float* base, long n, long dim,
                                     const float* query, long nq, long topk,
                                     unsigned int* out_ids) {
  void* h = seekdb_cuvs_build(base, n, dim);
  if (!h) return 10;
  int rc = seekdb_cuvs_search(h, query, nq, topk, out_ids, nullptr);
  seekdb_cuvs_free(h);
  return rc;
}

extern "C" int seekdb_cuvs_warmup(void) {
  if (env_int("SEEKDB_CUVS_WARMUP", 1) == 0) return 0;
  if (!ensure_gpu_ready()) return 1;
  pthread_mutex_lock(&g_warmup_mu);
  if (g_warmup_done) {
    pthread_mutex_unlock(&g_warmup_mu);
    return 0;
  }
  std::vector<float> dummy(256 * 256);
  for (size_t i = 0; i < dummy.size(); ++i) {
    dummy[i] = (float)((i * 1315423911ULL) & 0xFFu) * 0.01f;
  }
  unsigned int out[10];
  const long dims[2] = {128, 256};
  for (int di = 0; di < 2; ++di) {
    void* h = seekdb_cuvs_build(dummy.data(), 256, dims[di]);
    if (h) {
      (void)seekdb_cuvs_search(h, dummy.data(), 1, 10, out, nullptr);
      seekdb_cuvs_free(h);
    }
  }
  g_warmup_done = true;
  pthread_mutex_unlock(&g_warmup_mu);
  return 0;
}
