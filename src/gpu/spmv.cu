#include "cuda_common.h"
#include "spmv_internal.h"

static __device__ __forceinline__ double csr_row_dot_device(const HPRQPGPUCsrIndex *row_ptr,
                                                            const HPRQPGPUCsrIndex *col_ind,
                                                            const double *values,
                                                            const double *vector,
                                                            size_t row)
{
    size_t idx;
    size_t row_start = (size_t)row_ptr[row];
    size_t row_end = (size_t)row_ptr[row + 1];
    double sum = 0.0;

    for (idx = row_start; idx < row_end; ++idx) {
        sum += values[idx] * vector[(size_t)col_ind[idx]];
    }

    return sum;
}

__global__ static void gpu_csr_spmv_kernel(double *output,
                                           const HPRQPGPUCsrIndex *row_ptr,
                                           const HPRQPGPUCsrIndex *col_ind,
                                           const double *values,
                                           const double *input,
                                           size_t nrow)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;

    if (i >= nrow) {
        return;
    }

    output[i] = csr_row_dot_device(row_ptr, col_ind, values, input, i);
}

__global__ static void gpu_csr_spmv_warp_kernel(
    double *output,
    const HPRQPGPUCsrIndex *row_ptr,
    const HPRQPGPUCsrIndex *col_ind,
    const double *values,
    const double *input,
    size_t nrow)
{
    const unsigned int lane = threadIdx.x & 31U;
    const size_t warp_in_grid =
        ((size_t)blockIdx.x * (size_t)blockDim.x +
         (size_t)threadIdx.x) >> 5;
    double sum = 0.0;
    size_t idx;
    size_t row_start = 0;
    size_t row_end = 0;

    if (warp_in_grid < nrow) {
        row_start = (size_t)row_ptr[warp_in_grid];
        row_end = (size_t)row_ptr[warp_in_grid + 1];
        for (idx = row_start + (size_t)lane;
             idx < row_end;
             idx += 32U) {
            sum += values[idx] * input[(size_t)col_ind[idx]];
        }
    }
    sum += __shfl_down_sync(0xffffffffU, sum, 16);
    sum += __shfl_down_sync(0xffffffffU, sum, 8);
    sum += __shfl_down_sync(0xffffffffU, sum, 4);
    sum += __shfl_down_sync(0xffffffffU, sum, 2);
    sum += __shfl_down_sync(0xffffffffU, sum, 1);
    if (lane == 0U && warp_in_grid < nrow) {
        output[warp_in_grid] = sum;
    }
}

void gpu_spmv_a_init(HPRQPGPUSpMVA *spmv)
{
    if (spmv == NULL) {
        return;
    }
    std::memset(spmv, 0, sizeof(*spmv));
}

void gpu_spmv_at_init(HPRQPGPUSpMVAT *spmv)
{
    if (spmv == NULL) {
        return;
    }
    std::memset(spmv, 0, sizeof(*spmv));
}

void gpu_spmv_q_init(HPRQPGPUSpMVQ *spmv)
{
    if (spmv == NULL) {
        return;
    }
    std::memset(spmv, 0, sizeof(*spmv));
}

void gpu_spmv_a_free(HPRQPGPUSpMVA *spmv)
{
    if (spmv == NULL) {
        return;
    }
    if (spmv->graph_exec_x_hat != NULL) {
        cudaGraphExecDestroy((cudaGraphExec_t)spmv->graph_exec_x_hat);
    }
    if (spmv->graph_x_hat != NULL) {
        cudaGraphDestroy((cudaGraph_t)spmv->graph_x_hat);
    }
    if (spmv->desc_x_bar != NULL) {
        cusparseDestroyDnVec((cusparseDnVecDescr_t)spmv->desc_x_bar);
    }
    if (spmv->desc_x_hat != NULL) {
        cusparseDestroyDnVec((cusparseDnVecDescr_t)spmv->desc_x_hat);
    }
    if (spmv->desc_dx != NULL) {
        cusparseDestroyDnVec((cusparseDnVecDescr_t)spmv->desc_dx);
    }
    if (spmv->desc_tempv != NULL) {
        cusparseDestroyDnVec((cusparseDnVecDescr_t)spmv->desc_tempv);
    }
    if (spmv->desc_Ax != NULL) {
        cusparseDestroyDnVec((cusparseDnVecDescr_t)spmv->desc_Ax);
    }
    if (spmv->desc_A != NULL) {
        cusparseDestroySpMat((cusparseSpMatDescr_t)spmv->desc_A);
    }
    if (spmv->buffer != NULL) {
        cudaFree(spmv->buffer);
    }
    if (spmv->handle != NULL) {
        cusparseDestroy((cusparseHandle_t)spmv->handle);
    }
    gpu_spmv_a_init(spmv);
}

void gpu_spmv_at_free(HPRQPGPUSpMVAT *spmv)
{
    if (spmv == NULL) {
        return;
    }
    if (spmv->graph_exec_y != NULL) {
        cudaGraphExecDestroy((cudaGraphExec_t)spmv->graph_exec_y);
    }
    if (spmv->graph_y != NULL) {
        cudaGraphDestroy((cudaGraph_t)spmv->graph_y);
    }
    if (spmv->desc_y_bar != NULL) {
        cusparseDestroyDnVec((cusparseDnVecDescr_t)spmv->desc_y_bar);
    }
    if (spmv->desc_y != NULL) {
        cusparseDestroyDnVec((cusparseDnVecDescr_t)spmv->desc_y);
    }
    if (spmv->desc_ATy_bar != NULL) {
        cusparseDestroyDnVec((cusparseDnVecDescr_t)spmv->desc_ATy_bar);
    }
    if (spmv->desc_ATy != NULL) {
        cusparseDestroyDnVec((cusparseDnVecDescr_t)spmv->desc_ATy);
    }
    if (spmv->desc_ATdy != NULL) {
        cusparseDestroyDnVec((cusparseDnVecDescr_t)spmv->desc_ATdy);
    }
    if (spmv->desc_AT != NULL) {
        cusparseDestroySpMat((cusparseSpMatDescr_t)spmv->desc_AT);
    }
    if (spmv->buffer != NULL) {
        cudaFree(spmv->buffer);
    }
    if (spmv->handle != NULL) {
        cusparseDestroy((cusparseHandle_t)spmv->handle);
    }
    gpu_spmv_at_init(spmv);
}

void gpu_spmv_q_free(HPRQPGPUSpMVQ *spmv)
{
    if (spmv == NULL) {
        return;
    }
    if (spmv->desc_w != NULL) {
        cusparseDestroyDnVec((cusparseDnVecDescr_t)spmv->desc_w);
    }
    if (spmv->desc_w_bar != NULL) {
        cusparseDestroyDnVec((cusparseDnVecDescr_t)spmv->desc_w_bar);
    }
    if (spmv->desc_x_bar != NULL) {
        cusparseDestroyDnVec((cusparseDnVecDescr_t)spmv->desc_x_bar);
    }
    if (spmv->desc_Qw != NULL) {
        cusparseDestroyDnVec((cusparseDnVecDescr_t)spmv->desc_Qw);
    }
    if (spmv->desc_Qw_bar != NULL) {
        cusparseDestroyDnVec((cusparseDnVecDescr_t)spmv->desc_Qw_bar);
    }
    if (spmv->desc_Qx != NULL) {
        cusparseDestroyDnVec((cusparseDnVecDescr_t)spmv->desc_Qx);
    }
    if (spmv->desc_Q != NULL) {
        cusparseDestroySpMat((cusparseSpMatDescr_t)spmv->desc_Q);
    }
    if (spmv->buffer != NULL) {
        cudaFree(spmv->buffer);
    }
    if (spmv->handle != NULL) {
        cusparseDestroy((cusparseHandle_t)spmv->handle);
    }
    gpu_spmv_q_init(spmv);
}

static HPRQPStatus create_csr_descriptor(cusparseSpMatDescr_t *descriptor, const HPRQPGPUMatrixCSR *matrix)
{
    return map_cusparse_status(cusparseCreateCsr(descriptor,
                                                 (int64_t)matrix->nrow,
                                                 (int64_t)matrix->ncol,
                                                 (int64_t)matrix->nnz,
                                                 matrix->row_ptr,
                                                 matrix->col_ind,
                                                 matrix->values,
                                                 CUSPARSE_INDEX_32I,
                                                 CUSPARSE_INDEX_32I,
                                                 CUSPARSE_INDEX_BASE_ZERO,
                                                 CUDA_R_64F));
}

static HPRQPStatus create_vector_descriptor(cusparseDnVecDescr_t *descriptor, const HPRQPGPUVector *vector)
{
    return map_cusparse_status(cusparseCreateDnVec(descriptor,
                                                   (int64_t)vector->length,
                                                   vector->values,
                                                   CUDA_R_64F));
}

static HPRQPStatus allocate_spmv_buffer(void **buffer, size_t *buffer_size)
{
    if (buffer == NULL || buffer_size == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    *buffer = NULL;
    if (*buffer_size == 0) {
        return HPRQP_STATUS_OK;
    }
    return map_cuda_status(cudaMalloc(buffer, *buffer_size));
}

static HPRQPStatus prepare_spmv_common(cusparseHandle_t handle,
                                       cusparseSpMatDescr_t matrix_desc,
                                       cusparseDnVecDescr_t input_desc,
                                       cusparseDnVecDescr_t output_desc,
                                       void **buffer,
                                       size_t *buffer_size)
{
    size_t local_buffer_size = 0;
    HPRQPStatus status;

    status = map_cusparse_status(cusparseSpMV_bufferSize(handle,
                                                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                         &kSpmvAlpha,
                                                         matrix_desc,
                                                         input_desc,
                                                         &kSpmvBeta,
                                                         output_desc,
                                                         CUDA_R_64F,
                                                         CUSPARSE_SPMV_CSR_ALG2,
                                                         &local_buffer_size));
    if (status != HPRQP_STATUS_OK) {
        return status;
    }

    *buffer_size = local_buffer_size;
    status = allocate_spmv_buffer(buffer, buffer_size);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }

#if defined(CUSPARSE_VERSION) && CUSPARSE_VERSION >= 12400
    status = map_cusparse_status(cusparseSpMV_preprocess(handle,
                                                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                         &kSpmvAlpha,
                                                         matrix_desc,
                                                         input_desc,
                                                         &kSpmvBeta,
                                                         output_desc,
                                                         CUDA_R_64F,
                                                         CUSPARSE_SPMV_CSR_ALG2,
                                                         *buffer));
    if (status != HPRQP_STATUS_OK) {
        if (*buffer != NULL) {
            cudaFree(*buffer);
            *buffer = NULL;
        }
        *buffer_size = 0;
        return status;
    }
#endif

    return HPRQP_STATUS_OK;
}

static HPRQPStatus prepare_spmv_A(HPRQPGPUSpMVA *spmv,
                                  const HPRQPGPUMatrixCSR *matrix,
                                  const HPRQPGPUWorkspace *workspace)
{
    HPRQPStatus status;
    cusparseHandle_t handle = NULL;

    status = map_cusparse_status(cusparseCreate(&handle));
    if (status != HPRQP_STATUS_OK) {
        return status;
    }
    spmv->handle = handle;

    status = create_csr_descriptor((cusparseSpMatDescr_t *)&spmv->desc_A, matrix);
    if (status != HPRQP_STATUS_OK) {
        gpu_spmv_a_free(spmv);
        return status;
    }
    status = create_vector_descriptor((cusparseDnVecDescr_t *)&spmv->desc_x_bar, &workspace->x_bar);
    if (status != HPRQP_STATUS_OK) {
        gpu_spmv_a_free(spmv);
        return status;
    }
    status = create_vector_descriptor((cusparseDnVecDescr_t *)&spmv->desc_x_hat, &workspace->x_hat);
    if (status != HPRQP_STATUS_OK) {
        gpu_spmv_a_free(spmv);
        return status;
    }
    status = create_vector_descriptor((cusparseDnVecDescr_t *)&spmv->desc_dx, &workspace->dx);
    if (status != HPRQP_STATUS_OK) {
        gpu_spmv_a_free(spmv);
        return status;
    }
    status = create_vector_descriptor((cusparseDnVecDescr_t *)&spmv->desc_tempv, &workspace->tempv);
    if (status != HPRQP_STATUS_OK) {
        gpu_spmv_a_free(spmv);
        return status;
    }
    status = create_vector_descriptor((cusparseDnVecDescr_t *)&spmv->desc_Ax, &workspace->Ax);
    if (status != HPRQP_STATUS_OK) {
        gpu_spmv_a_free(spmv);
        return status;
    }

    status = prepare_spmv_common(handle,
                                 (cusparseSpMatDescr_t)spmv->desc_A,
                                 (cusparseDnVecDescr_t)spmv->desc_x_bar,
                                 (cusparseDnVecDescr_t)spmv->desc_Ax,
                                 &spmv->buffer,
                                 &spmv->buffer_size);
    if (status != HPRQP_STATUS_OK) {
        gpu_spmv_a_free(spmv);
        return status;
    }

    spmv->prepared = 1;
    return HPRQP_STATUS_OK;
}

static HPRQPStatus prepare_spmv_AT(HPRQPGPUSpMVAT *spmv,
                                   const HPRQPGPUMatrixCSR *matrix,
                                   const HPRQPGPUWorkspace *workspace)
{
    HPRQPStatus status;
    cusparseHandle_t handle = NULL;

    status = map_cusparse_status(cusparseCreate(&handle));
    if (status != HPRQP_STATUS_OK) {
        return status;
    }
    spmv->handle = handle;

    status = create_csr_descriptor((cusparseSpMatDescr_t *)&spmv->desc_AT, matrix);
    if (status != HPRQP_STATUS_OK) {
        gpu_spmv_at_free(spmv);
        return status;
    }
    status = create_vector_descriptor((cusparseDnVecDescr_t *)&spmv->desc_y_bar, &workspace->y_bar);
    if (status != HPRQP_STATUS_OK) {
        gpu_spmv_at_free(spmv);
        return status;
    }
    status = create_vector_descriptor((cusparseDnVecDescr_t *)&spmv->desc_y, &workspace->y);
    if (status != HPRQP_STATUS_OK) {
        gpu_spmv_at_free(spmv);
        return status;
    }
    status = create_vector_descriptor((cusparseDnVecDescr_t *)&spmv->desc_ATy_bar, &workspace->ATy_bar);
    if (status != HPRQP_STATUS_OK) {
        gpu_spmv_at_free(spmv);
        return status;
    }
    status = create_vector_descriptor((cusparseDnVecDescr_t *)&spmv->desc_ATy, &workspace->ATy);
    if (status != HPRQP_STATUS_OK) {
        gpu_spmv_at_free(spmv);
        return status;
    }
    status = create_vector_descriptor((cusparseDnVecDescr_t *)&spmv->desc_ATdy, &workspace->ATdy);
    if (status != HPRQP_STATUS_OK) {
        gpu_spmv_at_free(spmv);
        return status;
    }

    status = prepare_spmv_common(handle,
                                 (cusparseSpMatDescr_t)spmv->desc_AT,
                                 (cusparseDnVecDescr_t)spmv->desc_y_bar,
                                 (cusparseDnVecDescr_t)spmv->desc_ATy_bar,
                                 &spmv->buffer,
                                 &spmv->buffer_size);
    if (status != HPRQP_STATUS_OK) {
        gpu_spmv_at_free(spmv);
        return status;
    }

    spmv->prepared = 1;
    return HPRQP_STATUS_OK;
}

static HPRQPStatus prepare_spmv_Q(HPRQPGPUSpMVQ *spmv,
                                  const HPRQPGPUMatrixCSR *matrix,
                                  const HPRQPGPUWorkspace *workspace)
{
    HPRQPStatus status;
    cusparseHandle_t handle = NULL;

    status = map_cusparse_status(cusparseCreate(&handle));
    if (status != HPRQP_STATUS_OK) {
        return status;
    }
    spmv->handle = handle;

    status = create_csr_descriptor((cusparseSpMatDescr_t *)&spmv->desc_Q, matrix);
    if (status != HPRQP_STATUS_OK) {
        gpu_spmv_q_free(spmv);
        return status;
    }
    status = create_vector_descriptor((cusparseDnVecDescr_t *)&spmv->desc_w, &workspace->w);
    if (status != HPRQP_STATUS_OK) {
        gpu_spmv_q_free(spmv);
        return status;
    }
    status = create_vector_descriptor((cusparseDnVecDescr_t *)&spmv->desc_w_bar, &workspace->w_bar);
    if (status != HPRQP_STATUS_OK) {
        gpu_spmv_q_free(spmv);
        return status;
    }
    status = create_vector_descriptor((cusparseDnVecDescr_t *)&spmv->desc_x_bar, &workspace->x_bar);
    if (status != HPRQP_STATUS_OK) {
        gpu_spmv_q_free(spmv);
        return status;
    }
    status = create_vector_descriptor((cusparseDnVecDescr_t *)&spmv->desc_Qw, &workspace->Qw);
    if (status != HPRQP_STATUS_OK) {
        gpu_spmv_q_free(spmv);
        return status;
    }
    status = create_vector_descriptor((cusparseDnVecDescr_t *)&spmv->desc_Qw_bar, &workspace->Qw_bar);
    if (status != HPRQP_STATUS_OK) {
        gpu_spmv_q_free(spmv);
        return status;
    }
    status = create_vector_descriptor((cusparseDnVecDescr_t *)&spmv->desc_Qx, &workspace->Qx);
    if (status != HPRQP_STATUS_OK) {
        gpu_spmv_q_free(spmv);
        return status;
    }

    status = prepare_spmv_common(handle,
                                 (cusparseSpMatDescr_t)spmv->desc_Q,
                                 (cusparseDnVecDescr_t)spmv->desc_w,
                                 (cusparseDnVecDescr_t)spmv->desc_Qw,
                                 &spmv->buffer,
                                 &spmv->buffer_size);
    if (status != HPRQP_STATUS_OK) {
        gpu_spmv_q_free(spmv);
        return status;
    }

    spmv->prepared = 1;
    return HPRQP_STATUS_OK;
}

static HPRQPStatus apply_spmv(cusparseHandle_t handle,
                              cusparseSpMatDescr_t matrix_desc,
                              cusparseDnVecDescr_t input_desc,
                              cusparseDnVecDescr_t output_desc,
                              void *buffer)
{
    return map_cusparse_status(cusparseSpMV(handle,
                                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &kSpmvAlpha,
                                            matrix_desc,
                                            input_desc,
                                            &kSpmvBeta,
                                            output_desc,
                                            CUDA_R_64F,
                                            CUSPARSE_SPMV_CSR_ALG2,
                                            buffer));
}

/*
 * Capture one fixed-descriptor cuSPARSE product.  The graph contains the
 * same CSR_ALG2 kernels and memory operations as apply_spmv; only their host
 * submission is cached.  Failure is a performance fallback, not a solver
 * error, so callers retain the ordinary cuSPARSE path.
 */
static int prepare_fixed_spmv_graph(cusparseHandle_t handle,
                                    cusparseSpMatDescr_t matrix_desc,
                                    cusparseDnVecDescr_t input_desc,
                                    cusparseDnVecDescr_t output_desc,
                                    void *buffer,
                                    void **graph_out,
                                    void **graph_exec_out)
{
    cudaStream_t previous_stream = NULL;
    cudaStream_t capture_stream = NULL;
    cudaGraph_t graph = NULL;
    cudaGraphExec_t graph_exec = NULL;
    int capture_started = 0;
    int success = 0;

    if (graph_out == NULL || graph_exec_out == NULL) return 0;
    *graph_out = NULL;
    *graph_exec_out = NULL;
    if (cusparseGetStream(handle, &previous_stream) != CUSPARSE_STATUS_SUCCESS) {
        return 0;
    }

    /* Load any lazy cuSPARSE modules before stream capture. */
    if (apply_spmv(handle, matrix_desc, input_desc, output_desc, buffer) !=
        HPRQP_STATUS_OK) {
        return 0;
    }
    if (cudaStreamSynchronize(previous_stream) != cudaSuccess) return 0;
    if (cudaStreamCreateWithFlags(&capture_stream, cudaStreamNonBlocking) !=
        cudaSuccess) {
        return 0;
    }
    if (cusparseSetStream(handle, capture_stream) != CUSPARSE_STATUS_SUCCESS) {
        goto cleanup;
    }
    if (cudaStreamBeginCapture(capture_stream,
                               cudaStreamCaptureModeThreadLocal) !=
        cudaSuccess) {
        goto cleanup;
    }
    capture_started = 1;
    if (apply_spmv(handle, matrix_desc, input_desc, output_desc, buffer) !=
        HPRQP_STATUS_OK) {
        goto cleanup;
    }
    if (cudaStreamEndCapture(capture_stream, &graph) != cudaSuccess) {
        capture_started = 0;
        graph = NULL;
        goto cleanup;
    }
    capture_started = 0;
    if (graph == NULL ||
        cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0U) !=
            cudaSuccess) {
        goto cleanup;
    }
    *graph_out = graph;
    *graph_exec_out = graph_exec;
    graph = NULL;
    graph_exec = NULL;
    success = 1;

cleanup:
    if (capture_started) {
        cudaGraph_t invalid_graph = NULL;
        (void)cudaStreamEndCapture(capture_stream, &invalid_graph);
        if (invalid_graph != NULL) cudaGraphDestroy(invalid_graph);
    }
    (void)cusparseSetStream(handle, previous_stream);
    if (capture_stream != NULL) cudaStreamDestroy(capture_stream);
    if (graph_exec != NULL) cudaGraphExecDestroy(graph_exec);
    if (graph != NULL) cudaGraphDestroy(graph);
    return success;
}

static HPRQPStatus apply_spmv_dynamic(cusparseHandle_t handle,
                                      cusparseSpMatDescr_t matrix_desc,
                                      double *input_values,
                                      size_t input_length,
                                      double *output_values,
                                      size_t output_length,
                                      void *buffer)
{
    cusparseDnVecDescr_t input_desc = NULL;
    cusparseDnVecDescr_t output_desc = NULL;
    HPRQPStatus status;

    status = map_cusparse_status(cusparseCreateDnVec(&input_desc, (int64_t)input_length, input_values, CUDA_R_64F));
    if (status != HPRQP_STATUS_OK) {
        return status;
    }
    status = map_cusparse_status(cusparseCreateDnVec(&output_desc, (int64_t)output_length, output_values, CUDA_R_64F));
    if (status != HPRQP_STATUS_OK) {
        cusparseDestroyDnVec(input_desc);
        return status;
    }

    status = apply_spmv(handle, matrix_desc, input_desc, output_desc, buffer);
    cusparseDestroyDnVec(output_desc);
    cusparseDestroyDnVec(input_desc);
    return status;
}

static HPRQPStatus apply_spmv_custom(double *output,
                                     const HPRQPGPUMatrixCSR *matrix,
                                     const double *input,
                                     HPRQPGPUCustomSpmvKernel kernel)
{
    unsigned int threads;
    unsigned int blocks;

    if (output == NULL || matrix == NULL || input == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    if (matrix->nrow == 0) {
        return HPRQP_STATUS_OK;
    }

    threads = 256U;
    if (kernel == HPRQP_GPU_CUSTOM_SPMV_WARP) {
        const size_t warps_per_block = (size_t)threads / 32U;
        blocks = (unsigned int)((matrix->nrow + warps_per_block - 1) /
                                warps_per_block);
        gpu_csr_spmv_warp_kernel<<<blocks, threads>>>(output,
                                                       matrix->row_ptr,
                                                       matrix->col_ind,
                                                       matrix->values,
                                                       input,
                                                       matrix->nrow);
    } else {
        blocks = (unsigned int)((matrix->nrow + threads - 1) / threads);
        gpu_csr_spmv_kernel<<<blocks, threads>>>(output,
                                                 matrix->row_ptr,
                                                 matrix->col_ind,
                                                 matrix->values,
                                                 input,
                                                 matrix->nrow);
    }
    return HPRQP_STATUS_OK;
}

HPRQPStatus gpu_workspace_prepare_spmv(HPRQPGPUWorkspace *workspace,
                                       const HPRQPGPUProblem *model,
                                       int verbose)
{
    HPRQPStatus status;

    if (workspace == NULL || model == NULL || !workspace->allocated || !model->uploaded) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    gpu_spmv_a_free(&workspace->spmv_A);
    gpu_spmv_at_free(&workspace->spmv_AT);
    gpu_spmv_q_free(&workspace->spmv_Q);
    workspace->spmv_prepared = 0;

    if (!workspace->empty_A) {
        status = prepare_spmv_A(&workspace->spmv_A, &model->A, workspace);
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
        status = prepare_spmv_AT(&workspace->spmv_AT, &model->AT, workspace);
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
    }

    if (model->Q.nnz > 0) {
        status = prepare_spmv_Q(&workspace->spmv_Q, &model->Q, workspace);
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
    }

    workspace->spmv_prepared = 1;
    return HPRQP_STATUS_OK;
}

HPRQPStatus gpu_workspace_prepare_selected_spmv_graphs(
    HPRQPGPUWorkspace *workspace,
    const HPRQPGPUProblem *model,
    int verbose)
{
    int graph_a;
    int graph_at;
    int eligible;

    if (workspace == NULL || model == NULL || !workspace->spmv_prepared) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    eligible = !workspace->empty_A && workspace->p > 0U &&
        model->Q.nnz > 0U && workspace->Q_is_diag &&
        workspace->parallel_q_exact_single_diagonal &&
        workspace->spmv_mode_q == HPRQP_SPMV_MODE_CUSTOMIZED &&
        workspace->custom_spmv_q_kernel == HPRQP_GPU_CUSTOM_SPMV_ROW &&
        workspace->qc_bt_short_rows &&
        workspace->spmv_mode_a == HPRQP_SPMV_MODE_CUSPARSE;
    if (!eligible) {
        if (verbose && workspace->p > 0U) {
            std::printf("  QCQP cuSPARSE graphs: not selected\n");
        }
        return HPRQP_STATUS_OK;
    }

    graph_a = prepare_fixed_spmv_graph(
        (cusparseHandle_t)workspace->spmv_A.handle,
        (cusparseSpMatDescr_t)workspace->spmv_A.desc_A,
        (cusparseDnVecDescr_t)workspace->spmv_A.desc_x_hat,
        (cusparseDnVecDescr_t)workspace->spmv_A.desc_Ax,
        workspace->spmv_A.buffer,
        &workspace->spmv_A.graph_x_hat,
        &workspace->spmv_A.graph_exec_x_hat);
    graph_at = prepare_fixed_spmv_graph(
        (cusparseHandle_t)workspace->spmv_AT.handle,
        (cusparseSpMatDescr_t)workspace->spmv_AT.desc_AT,
        (cusparseDnVecDescr_t)workspace->spmv_AT.desc_y,
        (cusparseDnVecDescr_t)workspace->spmv_AT.desc_ATy,
        workspace->spmv_AT.buffer,
        &workspace->spmv_AT.graph_y,
        &workspace->spmv_AT.graph_exec_y);
    (void)cudaMemset(workspace->Ax.values, 0,
                     workspace->m * sizeof(double));
    (void)cudaMemset(workspace->ATy.values, 0,
                     workspace->n * sizeof(double));
    if (verbose) {
        std::printf("  QCQP cuSPARSE graphs: A*x_hat=%s, AT*y=%s\n",
                    graph_a ? "ready" : "fallback",
                    graph_at ? "ready" : "fallback");
    }
    return HPRQP_STATUS_OK;
}

HPRQPStatus gpu_spmv_apply_q_w(HPRQPGPUWorkspace *workspace, const HPRQPGPUProblem *model)
{
    if (workspace->spmv_mode_q == HPRQP_SPMV_MODE_CUSTOMIZED) {
        return apply_spmv_custom(workspace->Qw.values,
                                 &model->Q,
                                 workspace->w.values,
                                 workspace->custom_spmv_q_kernel);
    }
    cusparseSpMV((cusparseHandle_t)workspace->spmv_Q.handle,
                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                 &kSpmvAlpha,
                 (cusparseSpMatDescr_t)workspace->spmv_Q.desc_Q,
                 (cusparseDnVecDescr_t)workspace->spmv_Q.desc_w,
                 &kSpmvBeta,
                 (cusparseDnVecDescr_t)workspace->spmv_Q.desc_Qw,
                 CUDA_R_64F,
                 CUSPARSE_SPMV_CSR_ALG2,
                 workspace->spmv_Q.buffer);
    return HPRQP_STATUS_OK;
}

HPRQPStatus gpu_spmv_apply_q_w_bar(HPRQPGPUWorkspace *workspace, const HPRQPGPUProblem *model)
{
    if (workspace->spmv_mode_q == HPRQP_SPMV_MODE_CUSTOMIZED) {
        return apply_spmv_custom(workspace->Qw_bar.values,
                                 &model->Q,
                                 workspace->w_bar.values,
                                 workspace->custom_spmv_q_kernel);
    }
    cusparseSpMV((cusparseHandle_t)workspace->spmv_Q.handle,
                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                 &kSpmvAlpha,
                 (cusparseSpMatDescr_t)workspace->spmv_Q.desc_Q,
                 (cusparseDnVecDescr_t)workspace->spmv_Q.desc_w_bar,
                 &kSpmvBeta,
                 (cusparseDnVecDescr_t)workspace->spmv_Q.desc_Qw_bar,
                 CUDA_R_64F,
                 CUSPARSE_SPMV_CSR_ALG2,
                 workspace->spmv_Q.buffer);
    return HPRQP_STATUS_OK;
}

HPRQPStatus gpu_spmv_apply_q_x_bar(HPRQPGPUWorkspace *workspace, const HPRQPGPUProblem *model)
{
    if (workspace->spmv_mode_q == HPRQP_SPMV_MODE_CUSTOMIZED) {
        return apply_spmv_custom(workspace->Qx.values,
                                 &model->Q,
                                 workspace->x_bar.values,
                                 workspace->custom_spmv_q_kernel);
    }
    cusparseSpMV((cusparseHandle_t)workspace->spmv_Q.handle,
                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                 &kSpmvAlpha,
                 (cusparseSpMatDescr_t)workspace->spmv_Q.desc_Q,
                 (cusparseDnVecDescr_t)workspace->spmv_Q.desc_x_bar,
                 &kSpmvBeta,
                 (cusparseDnVecDescr_t)workspace->spmv_Q.desc_Qx,
                 CUDA_R_64F,
                 CUSPARSE_SPMV_CSR_ALG2,
                 workspace->spmv_Q.buffer);
    return HPRQP_STATUS_OK;
}

HPRQPStatus gpu_spmv_apply_q_dw(HPRQPGPUWorkspace *workspace, const HPRQPGPUProblem *model)
{
    if (workspace->spmv_mode_q == HPRQP_SPMV_MODE_CUSTOMIZED) {
        return apply_spmv_custom(workspace->dQw.values,
                                 &model->Q,
                                 workspace->dw.values,
                                 workspace->custom_spmv_q_kernel);
    }
    return apply_spmv_dynamic((cusparseHandle_t)workspace->spmv_Q.handle,
                              (cusparseSpMatDescr_t)workspace->spmv_Q.desc_Q,
                              workspace->dw.values,
                              workspace->n,
                              workspace->dQw.values,
                              workspace->n,
                              workspace->spmv_Q.buffer);
}

HPRQPStatus gpu_spmv_apply_q_atdy(HPRQPGPUWorkspace *workspace, const HPRQPGPUProblem *model)
{
    if (workspace->spmv_mode_q == HPRQP_SPMV_MODE_CUSTOMIZED) {
        return apply_spmv_custom(workspace->QATdy.values,
                                 &model->Q,
                                 workspace->ATdy.values,
                                 workspace->custom_spmv_q_kernel);
    }
    return apply_spmv_dynamic((cusparseHandle_t)workspace->spmv_Q.handle,
                              (cusparseSpMatDescr_t)workspace->spmv_Q.desc_Q,
                              workspace->ATdy.values,
                              workspace->n,
                              workspace->QATdy.values,
                              workspace->n,
                              workspace->spmv_Q.buffer);
}

HPRQPStatus gpu_spmv_apply_a_tempv(HPRQPGPUWorkspace *workspace, const HPRQPGPUProblem *model)
{
    if (workspace->spmv_mode_a == HPRQP_SPMV_MODE_CUSTOMIZED) {
        return apply_spmv_custom(workspace->Ax.values,
                                 &model->A,
                                 workspace->tempv.values,
                                 workspace->custom_spmv_a_kernel);
    }
    cusparseSpMV((cusparseHandle_t)workspace->spmv_A.handle,
                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                 &kSpmvAlpha,
                 (cusparseSpMatDescr_t)workspace->spmv_A.desc_A,
                 (cusparseDnVecDescr_t)workspace->spmv_A.desc_tempv,
                 &kSpmvBeta,
                 (cusparseDnVecDescr_t)workspace->spmv_A.desc_Ax,
                 CUDA_R_64F,
                 CUSPARSE_SPMV_CSR_ALG2,
                 workspace->spmv_A.buffer);
    return HPRQP_STATUS_OK;
}

HPRQPStatus gpu_spmv_apply_a_x_hat(HPRQPGPUWorkspace *workspace, const HPRQPGPUProblem *model)
{
    if (workspace->spmv_mode_a == HPRQP_SPMV_MODE_CUSTOMIZED) {
        return apply_spmv_custom(workspace->Ax.values,
                                 &model->A,
                                 workspace->x_hat.values,
                                 workspace->custom_spmv_a_kernel);
    }
    return apply_spmv((cusparseHandle_t)workspace->spmv_A.handle,
                      (cusparseSpMatDescr_t)workspace->spmv_A.desc_A,
                      (cusparseDnVecDescr_t)workspace->spmv_A.desc_x_hat,
                      (cusparseDnVecDescr_t)workspace->spmv_A.desc_Ax,
                      workspace->spmv_A.buffer);
}

HPRQPStatus gpu_spmv_apply_a_x_hat_graph(
    HPRQPGPUWorkspace *workspace,
    const HPRQPGPUProblem *model)
{
    if (workspace->spmv_mode_a == HPRQP_SPMV_MODE_CUSTOMIZED ||
        workspace->spmv_A.graph_exec_x_hat == NULL) {
        return gpu_spmv_apply_a_x_hat(workspace, model);
    }
    return map_cuda_status(cudaGraphLaunch(
        (cudaGraphExec_t)workspace->spmv_A.graph_exec_x_hat,
        (cudaStream_t)0));
}

HPRQPStatus gpu_spmv_apply_a_x_bar(HPRQPGPUWorkspace *workspace, const HPRQPGPUProblem *model)
{
    if (workspace->spmv_mode_a == HPRQP_SPMV_MODE_CUSTOMIZED) {
        return apply_spmv_custom(workspace->Ax.values,
                                 &model->A,
                                 workspace->x_bar.values,
                                 workspace->custom_spmv_a_kernel);
    }
    cusparseSpMV((cusparseHandle_t)workspace->spmv_A.handle,
                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                 &kSpmvAlpha,
                 (cusparseSpMatDescr_t)workspace->spmv_A.desc_A,
                 (cusparseDnVecDescr_t)workspace->spmv_A.desc_x_bar,
                 &kSpmvBeta,
                 (cusparseDnVecDescr_t)workspace->spmv_A.desc_Ax,
                 CUDA_R_64F,
                 CUSPARSE_SPMV_CSR_ALG2,
                 workspace->spmv_A.buffer);
    return HPRQP_STATUS_OK;
}

HPRQPStatus gpu_spmv_apply_at_y(HPRQPGPUWorkspace *workspace, const HPRQPGPUProblem *model)
{
    if (workspace->spmv_mode_a == HPRQP_SPMV_MODE_CUSTOMIZED) {
        return apply_spmv_custom(workspace->ATy.values,
                                 &model->AT,
                                 workspace->y.values,
                                 workspace->custom_spmv_a_kernel);
    }
    cusparseSpMV((cusparseHandle_t)workspace->spmv_AT.handle,
                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                 &kSpmvAlpha,
                 (cusparseSpMatDescr_t)workspace->spmv_AT.desc_AT,
                 (cusparseDnVecDescr_t)workspace->spmv_AT.desc_y,
                 &kSpmvBeta,
                 (cusparseDnVecDescr_t)workspace->spmv_AT.desc_ATy,
                 CUDA_R_64F,
                 CUSPARSE_SPMV_CSR_ALG2,
                 workspace->spmv_AT.buffer);
    return HPRQP_STATUS_OK;
}

HPRQPStatus gpu_spmv_apply_at_y_graph(
    HPRQPGPUWorkspace *workspace,
    const HPRQPGPUProblem *model)
{
    if (workspace->spmv_mode_a == HPRQP_SPMV_MODE_CUSTOMIZED ||
        workspace->spmv_AT.graph_exec_y == NULL) {
        return gpu_spmv_apply_at_y(workspace, model);
    }
    return map_cuda_status(cudaGraphLaunch(
        (cudaGraphExec_t)workspace->spmv_AT.graph_exec_y,
        (cudaStream_t)0));
}

HPRQPStatus gpu_spmv_apply_at_y_bar(HPRQPGPUWorkspace *workspace, const HPRQPGPUProblem *model)
{
    if (workspace->spmv_mode_a == HPRQP_SPMV_MODE_CUSTOMIZED) {
        return apply_spmv_custom(workspace->ATy_bar.values,
                                 &model->AT,
                                 workspace->y_bar.values,
                                 workspace->custom_spmv_a_kernel);
    }
    cusparseSpMV((cusparseHandle_t)workspace->spmv_AT.handle,
                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                 &kSpmvAlpha,
                 (cusparseSpMatDescr_t)workspace->spmv_AT.desc_AT,
                 (cusparseDnVecDescr_t)workspace->spmv_AT.desc_y_bar,
                 &kSpmvBeta,
                 (cusparseDnVecDescr_t)workspace->spmv_AT.desc_ATy_bar,
                 CUDA_R_64F,
                 CUSPARSE_SPMV_CSR_ALG2,
                 workspace->spmv_AT.buffer);
    return HPRQP_STATUS_OK;
}

HPRQPStatus gpu_spmv_apply_at_dy(HPRQPGPUWorkspace *workspace, const HPRQPGPUProblem *model)
{
    if (workspace->spmv_mode_a == HPRQP_SPMV_MODE_CUSTOMIZED) {
        return apply_spmv_custom(workspace->ATdy.values,
                                 &model->AT,
                                 workspace->dy.values,
                                 workspace->custom_spmv_a_kernel);
    }
    return apply_spmv_dynamic((cusparseHandle_t)workspace->spmv_AT.handle,
                              (cusparseSpMatDescr_t)workspace->spmv_AT.desc_AT,
                              workspace->dy.values,
                              workspace->m,
                              workspace->ATdy.values,
                              workspace->n,
                              workspace->spmv_AT.buffer);
}

HPRQPStatus gpu_spmv_apply_at_y_bar_residual(HPRQPGPUWorkspace *workspace, const HPRQPGPUProblem *model)
{
    if (workspace->spmv_mode_a == HPRQP_SPMV_MODE_CUSTOMIZED) {
        return apply_spmv_custom(workspace->ATdy.values,
                                 &model->AT,
                                 workspace->y_bar.values,
                                 workspace->custom_spmv_a_kernel);
    }
    cusparseSpMV((cusparseHandle_t)workspace->spmv_AT.handle,
                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                 &kSpmvAlpha,
                 (cusparseSpMatDescr_t)workspace->spmv_AT.desc_AT,
                 (cusparseDnVecDescr_t)workspace->spmv_AT.desc_y_bar,
                 &kSpmvBeta,
                 (cusparseDnVecDescr_t)workspace->spmv_AT.desc_ATdy,
                 CUDA_R_64F,
                 CUSPARSE_SPMV_CSR_ALG2,
                 workspace->spmv_AT.buffer);
    return HPRQP_STATUS_OK;
}
