"""
Type for statistics returned by the LevenbergMarquardt solvers, the attributes are:
  - model
  - status
  - solution
  - rNorm0
  - rNorm
  - ArNorm0
  - ArNorm
  - iter
  - inner_iter
  - elapsed_time
"""
mutable struct LMStatsAD{T,S}
  model :: AbstractNLSModel
  status :: Symbol
  solution :: S
  rNorm0 :: T
  rNorm :: T
  ArNorm0 :: T
  ArNorm :: T
  iter :: Int
  inner_iter :: Int
  elapsed_time :: Float64
end