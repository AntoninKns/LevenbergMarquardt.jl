"""
    SimpleNLSModel <: AbstractNLSModel
Simple NLSModel for testing purposes.
Modified problem 20 in the Hock-Schittkowski Suite.
     min   ½‖F(x)‖²
where
    F(x) = [1 - x₁; 10 (x₂ - x₁²)]
x₀ = ones(n).
Modified SimpleNLSModel.
"""
mutable struct SimpleNLSModel{T, S} <: AbstractNLSModel{T, S}
  meta::NLPModelMeta{T, S}
  nls_meta::NLSMeta{T, S}
  counters::NLSCounters
  ##### Jacobians data #####
  rows::Vector{Int}
  cols::Vector{Int}
  vals::Vector{T}
end

function SimpleNLSModel(::Type{T}) where {T}
  meta = NLPModelMeta(
    2,
    x0 = 2*ones(T, 2),
    name = "Simple NLS Model",
  )
  nls_meta = NLSMeta{T, Vector{T}}(2, 2, nnzj = 3)
  rows = Vector{Int}(undef, nls_meta.nnzj)
  cols = Vector{Int}(undef, nls_meta.nnzj)
  vals = Vector{T}(undef, nls_meta.nnzj)

  return SimpleNLSModel(meta, nls_meta, NLSCounters(), rows, cols, vals)
end

SimpleNLSModel() = SimpleNLSModel(Float64)

function NLPModels.residual!(nls::SimpleNLSModel, x::AbstractVector, Fx::AbstractVector)
  @lencheck 2 x Fx
  increment!(nls, :neval_residual)
  Fx .= [1 - x[1]; 10 * (x[2] - x[1]^2)]
  return Fx
end

# Jx = [-1  0; -20x₁  10]
function NLPModels.jac_structure_residual!(
  nls::SimpleNLSModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  @lencheck 3 rows cols
  rows .= [1, 2, 2]
  cols .= [1, 1, 2]
  return rows, cols
end

function NLPModels.jac_coord_residual!(nls::SimpleNLSModel, x::AbstractVector, vals::AbstractVector)
  @lencheck 2 x
  @lencheck 3 vals
  increment!(nls, :neval_jac_residual)
  vals .= [-1, -20x[1], 10]
  return vals
end

function NLPModels.jprod_residual!(
  nls::SimpleNLSModel,
  x::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  @lencheck 2 x v Jv
  increment!(nls, :neval_jprod_residual)
  Jv .= [-v[1]; -20 * x[1] * v[1] + 10 * v[2]]
  return Jv
end

function NLPModels.jtprod_residual!(
  nls::SimpleNLSModel,
  x::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  @lencheck 2 x v Jtv
  increment!(nls, :neval_jtprod_residual)
  Jtv .= [-v[1] - 20 * x[1] * v[2]; 10 * v[2]]
  return Jtv
end
