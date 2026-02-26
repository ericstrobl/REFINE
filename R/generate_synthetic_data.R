generate_synthetic_data <- function(N, d, q, T) {
  # Generate a minimal synthetic longitudinal dataset for REFINE.
  #
  # Inputs
  #   N: number of subjects
  #   d: number of questionnaire items at baseline (X0)
  #   q: number of baseline nuisance covariates (Z0)
  #   T: number of follow-up time points
  #
  # Output (list)
  #   Y0      : N x (d+q) baseline matrix [X0 | Z0]
  #   followups: list of length T, each element is:
  #              - Yt   : n_t x d follow-up item matrix
  #              - idx0 : length n_t integer indices into rows of Y0 (overlap subjects)
  #   z_idx   : column indices of Z0 within Y0
  
  # ---- Baseline: correlated questionnaire items + independent nuisance covariates ----
  latent <- matrix(rnorm(N * 2), N, 2)
  L <- matrix(rnorm(2 * d), 2, d)
  X0 <- latent %*% L + 0.3 * matrix(rnorm(N * d), N, d)
  Z0 <- matrix(rnorm(N * q), N, q)
  
  Y0 <- cbind(X0, Z0)
  z_idx <- (d + 1):(d + q)
  
  # Create time-varying linear dynamics that gradually shrink with t, plus noise.
  A_list <- lapply(seq_len(T), function(t) {
    shrink <- 0.8^t
    diag(d) * shrink + matrix(rnorm(d * d, sd = 0.05), d, d)
  })
  
  # Simulate follow-up item matrices (each is N x d before missingness)
  X_list <- lapply(seq_len(T), function(t) {
    X0 %*% A_list[[t]] + 0.3 * matrix(rnorm(N * d), N, d)
  })
  
  # Simulate follow-up missingness by sub-sampling subjects at each timepoint
  # (earlier times observed for more subjects).
  keep_frac <- seq(0.80, 0.60, length.out = T)
  idx_list <- lapply(seq_len(T), function(t) {
    sample(seq_len(N), size = round(keep_frac[t] * N))
  })
  
  followups <- lapply(seq_len(T), function(t) {
    idx <- idx_list[[t]]
    list(Yt = X_list[[t]][idx, , drop = FALSE], idx0 = idx)
  })
  
  list(Y0 = Y0, followups = followups, z_idx = z_idx)
}
