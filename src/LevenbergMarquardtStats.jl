"""
Type for statistics returned by the LevenbergMarquardt solvers, the attributes are:
  - model
  - status
  - solution
  - rNorm
  - rNorm0
  - ArNorm
  - ArNorm0
  - iter
  - inner_iter
  - elapsed_time
"""
mutable struct LMStats
  model :: AbstractNLSModel
  status :: Symbol
  solution :: AbstractVector
  rNorm :: AbstractFloat
  rNorm0 :: AbstractFloat
  ArNorm :: AbstractFloat
  ArNorm0 :: AbstractFloat
  iter :: Int
  inner_iter :: Int
  elapsed_time :: Float64
end