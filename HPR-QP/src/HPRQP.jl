module HPRQP

using QPSReader
using SparseArrays
using LinearAlgebra
using CUDA
using CUDA.CUSPARSE
using CUDA.CUBLAS: symm!
using Printf
using CSV
using DataFrames
using Random
using Logging
using MAT
using HDF5
using Dates

include("structs.jl")
include("utils.jl")
include("kernels.jl")
include("algorithm.jl")

# Export main functions and types for public API usage
export HPRQP_parameters, HPRQP_results
export build_from_mps, build_from_QAbc, build_from_mat
export build_from_ABST, build_from_Ab_lambda  # New Q operator builders
export optimize

end
