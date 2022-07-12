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
mutable struct LMStatsAD{T,S}
  model :: AbstractNLSModel
  status :: Symbol
  solution :: S
  rNorm :: T
  rNorm0 :: T
  ArNorm :: T
  ArNorm0 :: T
  iter :: Int
  inner_iter :: Int
  elapsed_time :: Float64
end