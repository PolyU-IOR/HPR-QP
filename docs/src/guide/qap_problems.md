# QAP Problems

The Quadratic Assignment Problem (QAP) is a fundamental combinatorial optimization problem. HPRQP provides a specialized operator for efficiently solving QAP and its relaxations.

## Problem Formulation

The QAP aims to assign ``n`` facilities to ``n`` locations to minimize:

```math
\min_{\pi} \quad \sum_{i=1}^n \sum_{j=1}^n f_{ij} \cdot d_{\pi(i),\pi(j)}
```

where:
- ``f_{ij}`` is the flow between facilities ``i`` and ``j`` (flow matrix ``F``)
- ``d_{kl}`` is the distance between locations ``k`` and ``l`` (distance matrix ``D``)
- ``\pi`` is a permutation (assignment)

### Linearized Formulation

Using decision variables ``x_{ik} \in \{0, 1\}`` where ``x_{ik} = 1`` if facility ``i`` is assigned to location ``k``:

```math
\begin{array}{ll}
\min \quad & \sum_{i=1}^n \sum_{j=1}^n \sum_{k=1}^n \sum_{l=1}^n f_{ij} \cdot d_{kl} \cdot x_{ik} \cdot x_{jl} \\
\text{s.t.} \quad & \sum_{k=1}^n x_{ik} = 1, \quad \forall i \\
& \sum_{i=1}^n x_{ik} = 1, \quad \forall k \\
& x_{ik} \in \{0, 1\}
\end{array}
```

This is a quadratic program with the quadratic term having Kronecker product structure: ``Q = F \otimes D``.

### Continuous Relaxation

Replace ``x_{ik} \in \{0,1\}`` with ``x_{ik} \in [0,1]`` to get a QP that HPRQP can solve:

```math
\begin{array}{ll}
\min \quad & \text{vec}(X)^T (F \otimes D) \text{vec}(X) \\
\text{s.t.} \quad & X \mathbf{1} = \mathbf{1}, \quad X^T \mathbf{1} = \mathbf{1} \\
& 0 \leq X \leq 1
\end{array}
```

## Why Use the QAP Operator?

The QAP operator exploits the Kronecker product structure ``Q = F \otimes D``:

**Benefits:**
- ✅ **Memory efficient**: Stores ``F`` and ``D`` (``2n^2``) instead of full ``Q`` (``n^4`` entries)
- ✅ **Faster computation**: Exploits structure in matrix-vector products
- ✅ **Numerically stable**: Avoids explicitly forming huge Kronecker product

**Memory comparison for ``n = 100``:**
- Full Q matrix: ``n^4 = 100,000,000`` entries (800 MB)
- F and D matrices: ``2n^2 = 20,000`` entries (0.16 MB)

## Basic Usage

```julia
using HPRQP
using SparseArrays

# Problem size
n = 10  # Facilities/locations

# Flow matrix (traffic between facilities)
F = rand(n, n)
F = (F + F') / 2  # Make symmetric

# Distance matrix (distance between locations)
D = rand(n, n)
D = (D + D') / 2  # Make symmetric

# Create QAP operator
Q_qap = QAPOperatorCPU(F, D, n)

# Build QAP constraints
# Vectorize X: x = vec(X) has length n^2
n2 = n * n

# Row sum constraints: X*1 = 1
A_row = sparse(zeros(n, n2))
for i in 1:n
    for k in 1:n
        A_row[i, (i-1)*n + k] = 1.0
    end
end

# Column sum constraints: X'*1 = 1
A_col = sparse(zeros(n, n2))
for k in 1:n
    for i in 1:n
        A_col[k, (i-1)*n + k] = 1.0
    end
end

# Combine constraints
A = vcat(A_row, A_col)
AL = ones(2n)
AU = ones(2n)

# No linear term
c = zeros(n2)

# Box constraints: 0 <= x <= 1
l = zeros(n2)
u = ones(n2)

# Build and solve
model = build_from_QAbc(Q_qap, A, c, AL, AU, l, u)

params = HPRQP_parameters()
params.use_gpu = true
params.stoptol = 1e-4

result = optimize(model, params)

# Extract solution matrix
X_vec = result.x
X = reshape(X_vec, n, n)

println("QAP relaxation objective: ", result.primal_obj)
println("Solution matrix X:")
display(X)
```

## Constructing QAP Constraints

### Helper Function

```julia
using SparseArrays

function build_qap_constraints(n::Int)
    """Build constraint matrix for QAP with n facilities/locations"""
    n2 = n * n
    
    # Row constraints: sum over k of x[i,k] = 1 for each i
    rows_row = Int[]
    cols_row = Int[]
    vals_row = Float64[]
    
    for i in 1:n
        for k in 1:n
            idx = (i-1)*n + k  # vec(X) index
            push!(rows_row, i)
            push!(cols_row, idx)
            push!(vals_row, 1.0)
        end
    end
    A_row = sparse(rows_row, cols_row, vals_row, n, n2)
    
    # Column constraints: sum over i of x[i,k] = 1 for each k
    rows_col = Int[]
    cols_col = Int[]
    vals_col = Float64[]
    
    for k in 1:n
        for i in 1:n
            idx = (i-1)*n + k
            push!(rows_col, k)
            push!(cols_col, idx)
            push!(vals_col, 1.0)
        end
    end
    A_col = sparse(rows_col, cols_col, vals_col, n, n2)
    
    # Combine
    A = vcat(A_row, A_col)
    AL = ones(2n)
    AU = ones(2n)
    
    return A, AL, AU
end

# Use it
A, AL, AU = build_qap_constraints(n)
```

## Complete Example: Facility Location

```julia
using HPRQP
using SparseArrays
using LinearAlgebra

# Problem: Assign 5 facilities to 5 locations
n = 5

# Flow matrix: how much traffic between facilities
# Example: factories with material flow
F = [
    0  10  5   2   1;
    10  0  8   3   2;
    5   8  0   6   4;
    2   3  6   0   7;
    1   2  4   7   0
]
F = Float64.(F)

# Distance matrix: distance between locations
# Example: geographical distances
D = [
    0.0  1.0  2.0  3.0  4.0;
    1.0  0.0  1.0  2.0  3.0;
    2.0  1.0  0.0  1.0  2.0;
    3.0  2.0  1.0  0.0  1.0;
    4.0  3.0  2.0  1.0  0.0
]

# Create QAP operator
Q_qap = QAPOperatorCPU(F, D, n)

# Build constraints
A, AL, AU = build_qap_constraints(n)

# Solve relaxation
model = build_from_QAbc(Q_qap, A, zeros(n^2), AL, AU, zeros(n^2), ones(n^2))

params = HPRQP_parameters()
params.use_gpu = false  # Small problem
result = optimize(model, params)

# Display solution
X = reshape(result.x, n, n)
println("\nQAP Relaxation Solution:")
println("Objective value: ", result.primal_obj)
println("\nAssignment matrix X:")
display(round.(X, digits=3))

# Find near-integer assignments
println("\nNear-integer assignments (> 0.9):")
for i in 1:n, k in 1:n
    if X[i,k] > 0.9
        println("  Facility $i → Location $k (x = ", round(X[i,k], digits=3), ")")
    end
end
```

## Rounding to Integer Solution

The continuous relaxation often gives fractional solutions. Common rounding strategies:

### Greedy Rounding

```julia
function greedy_rounding(X::Matrix)
    n = size(X, 1)
    assignment = zeros(Int, n)
    facilities_assigned = Set{Int}()
    locations_used = Set{Int}()
    
    # Flatten and sort by value
    entries = [(i, k, X[i,k]) for i in 1:n, k in 1:n]
    sort!(entries, by=x->x[3], rev=true)
    
    # Greedily assign
    for (i, k, val) in entries
        if !(i in facilities_assigned) && !(k in locations_used)
            assignment[i] = k
            push!(facilities_assigned, i)
            push!(locations_used, k)
        end
    end
    
    return assignment
end

# Use it
X = reshape(result.x, n, n)
assignment = greedy_rounding(X)
println("Greedy assignment: ", assignment)

# Evaluate objective
obj = sum(F[i,j] * D[assignment[i], assignment[j]] for i in 1:n, j in 1:n)
println("Integer objective: ", obj)
```

### Randomized Rounding

```julia
function randomized_rounding(X::Matrix; n_samples=100)
    n = size(X, 1)
    best_obj = Inf
    best_assignment = nothing
    
    for _ in 1:n_samples
        assignment = zeros(Int, n)
        locations = collect(1:n)
        
        for i in 1:n
            # Sample location proportional to X[i,:]
            probs = X[i, locations]
            probs ./= sum(probs)
            
            k_idx = sample(1:length(locations), Weights(probs))
            k = locations[k_idx]
            
            assignment[i] = k
            deleteat!(locations, k_idx)
        end
        
        # Evaluate
        obj = sum(F[i,j] * D[assignment[i], assignment[j]] for i in 1:n, j in 1:n)
        if obj < best_obj
            best_obj = obj
            best_assignment = assignment
        end
    end
    
    return best_assignment, best_obj
end
```

## Applications

### Factory Layout

```julia
# Arrange machines to minimize material handling
n_machines = 8

# Material flow (parts/hour between machines)
F = rand(0:10, n_machines, n_machines)
F = (F + F') / 2
F[diagind(F)] .= 0

# Distance between locations (meters)
D = rand(1.0:20.0, n_machines, n_machines)
D = (D + D') / 2
D[diagind(D)] .= 0

Q_qap = QAPOperatorCPU(F, D, n_machines)
# ... solve as above
```

### Communication Network

```julia
# Assign tasks to processors to minimize communication cost
n_tasks = 12

# Communication volume between tasks (MB)
F = rand(0:100, n_tasks, n_tasks)
F = (F + F') / 2

# Network latency between processors (ms)
D = rand(1.0:10.0, n_tasks, n_tasks)
D = (D + D') / 2

Q_qap = QAPOperatorCPU(F, D, n_tasks)
# ... solve
```

## Performance Tips

1. **GPU for Large Problems**:
   ```julia
   params = HPRQP_parameters()
   params.use_gpu = true  # Essential for n > 20
   ```

2. **Warm-Start from Heuristics**:
   ```julia
   # Get initial assignment from greedy heuristic
   x0 = zeros(n^2)
   # ... fill x0 based on heuristic
   
   params.initial_x = x0
   ```

3. **Hierarchical Approach**:
   ```julia
   # Solve small problem first, use to warm-start larger
   n_small = 5
   n_large = 10
   
   # Solve small
   Q_small = QAPOperatorCPU(F[1:n_small, 1:n_small], D[1:n_small, 1:n_small], n_small)
   # ... solve
   
   # Extend solution to larger problem
   # Use as warm-start for full problem
   ```

## See Also

- [Q Operators Overview](q_operators_overview.md) - Understanding Q operators
- [Sparse Matrix QP](sparse_matrix_qp.md) - Alternative approach
- [Direct API](direct_api.md) - Building models with operators
