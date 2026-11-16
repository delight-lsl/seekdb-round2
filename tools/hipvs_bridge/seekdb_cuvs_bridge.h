#ifndef SEEKDB_CUVS_BRIDGE_H
#define SEEKDB_CUVS_BRIDGE_H
#ifdef __cplusplus
extern "C" {
#endif
/* One-shot: build a CAGRA index over base (n x dim, row-major f32) on the GPU and
 * search queries (nq x dim), writing top-k uint32 neighbor ids to out_ids
 * (length nq*topk). Returns 0 on success, non-zero on error. */
int seekdb_cuvs_cagra_knn(const float* base, long n, long dim,
                          const float* query, long nq, long topk,
                          unsigned int* out_ids);

/* Build-once / search-many (used by seekdb's ob_vsag_adaptor GPU path). */
void* seekdb_cuvs_build(const float* base, long n, long dim);
int   seekdb_cuvs_search(void* handle, const float* query, long nq, long topk,
                         unsigned int* out_ids, float* out_dist);
void  seekdb_cuvs_free(void* handle);
int   seekdb_cuvs_serialize(void* handle, const char* filename);
void* seekdb_cuvs_deserialize(const char* filename, long n, long dim);

/* Zero-file in-memory compact graph snapshot API (avoids duplicate dataset & temp file I/O). */
long  seekdb_cuvs_get_graph_degree(void* handle);
int   seekdb_cuvs_export_graph(void* handle, unsigned int* out_graph, unsigned long bytes);
void* seekdb_cuvs_from_graph(const float* base, long n, long dim,
                             const unsigned int* graph, long degree);

/* Optional background pre-warmup of GPU context, RMM pool, and hipVS/cuVS CAGRA kernels. */
int   seekdb_cuvs_warmup(void);
#ifdef __cplusplus
}
#endif
#endif
