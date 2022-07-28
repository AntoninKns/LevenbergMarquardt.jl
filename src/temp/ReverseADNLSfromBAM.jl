export ADBundleAdjustmentModel

"""
Function that transforms a BundleAdjustmentModel into a ReverseADNLSModel
compatible with LevenbergMarquardt.jl methods.
"""
function ADBundleAdjustmentModel(model::BundleAdjustmentModel)
  # residual function for evaluation
  T = eltype(model.meta.x0)
  k = similar(model.meta.x0)
  P1 = similar(model.meta.x0)
  resid!(rx, x) = BundleAdjustmentModels.residuals!(x, rx, model.cams_indices, model.pnts_indices, model.nobs, model.npnts, k, P1, model.pt2d)

  # residual function for forward AD
  k_fwd = Vector{SparseDiffTools.Dual{ForwardDiff.Tag{SparseDiffTools.DeivVecTag, T}, T, 1}}(undef, model.meta.nvar)
  P1_fwd = Vector{SparseDiffTools.Dual{ForwardDiff.Tag{SparseDiffTools.DeivVecTag, T}, T, 1}}(undef, model.meta.nvar)
  resid_fwd!(rx, x) = BundleAdjustmentModels.residuals!(x, rx, model.cams_indices, model.pnts_indices, model.nobs, model.npnts, k_fwd, P1_fwd, model.pt2d)

  # residual function for reverse AD
  k_rev = Vector{ReverseDiff.TrackedReal{T, T, Nothing}}(undef, model.meta.nvar)
  P1_rev = Vector{ReverseDiff.TrackedReal{T, T, Nothing}}(undef, model.meta.nvar)
  resid_rev!(rx, x) = BundleAdjustmentModels.residuals!(x, rx, model.cams_indices, model.pnts_indices, model.nobs, model.npnts, k_rev, P1_rev, model.pt2d)

  return ReverseADNLSModel(resid!, model.nls_meta.nequ, model.meta.x0, r_fwd! = resid_fwd!, r_rev! = resid_rev!, name=model.meta.name)
end