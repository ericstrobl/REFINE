# ==============================================================================
# REFINE reference implementation (baseline-only inference, follow-up–informed training)
#
# Notation (matches the paper):
#   - Y0 : N x p baseline matrix containing questionnaire items X0 and optional nuisances Z0
#   - X0 : baseline questionnaire items (subset of columns of Y0)
#   - Z0 : baseline nuisance covariates (remaining columns of Y0, indexed by z_idx)
#   - For each follow-up time t:
#       * Yt   : n_t x d_t follow-up item matrix observed for a subset of subjects
#       * idx0 : length n_t integer vector mapping rows of Yt to rows in Y0 (overlap set)
#
# REFINE decomposes the conditional mean into:
#   (X0, Z0) --h_t--> X0bar^(t) --beta_t--> X_t
#
# where:
#   - B_t (d_t x dX) linearly reconstructs X0 from Yt on the overlap subjects
#   - a_t (length dX) is the intercept recovered via centering (means)
#   - X0bar^(t) = a_t + Yt B_t is the follow-up–informed proxy target in the *X0 coordinate system*
#   - h_t is a nonlinear “preprocessor” that predicts X0bar^(t) from baseline inputs (X0, Z0)
#   - beta_t is the linear decoder from stabilized baseline items to follow-up items
#     (here estimated via an SVD pseudoinverse of B_t for numerical robustness)
# ==============================================================================


# ---------- helper: Moore–Penrose pseudoinverse via SVD ----------
# Computes A^+ using an SVD threshold to drop near-zero singular values.
#   - A: matrix to pseudoinvert
#   - tol: singular-value cutoff (defaults to a standard machine-precision heuristic)
pinv_svd <- function(A, tol = NULL) {
  s <- svd(A)
  d <- s$d
  if (is.null(tol)) tol <- max(dim(A)) * max(d) * .Machine$double.eps
  d_inv <- ifelse(d > tol, 1 / d, 0)
  s$v %*% (t(s$u) * d_inv)
}


# ---------- internal: split baseline into questionnaire items (X0) vs nuisances (Z0) ----------
# Y0: N x p baseline matrix.
# z_idx: integer indices (1..p) identifying nuisance covariates within Y0.
#        - NULL / empty means "no nuisance covariates"
#
# Returns:
#   - X0    : N x dX matrix (columns not in z_idx)
#   - Z0    : N x q matrix (columns in z_idx) or NULL if q=0
#   - x_idx : indices of X0 columns in Y0
#   - z_idx : validated nuisance indices (possibly empty)
refine_split_baseline <- function(Y0, z_idx = NULL) {
  stopifnot(is.matrix(Y0))
  p <- ncol(Y0)
  
  if (is.null(z_idx) || length(z_idx) == 0L) {
    z_idx <- integer(0)
  } else {
    z_idx <- sort(unique(as.integer(z_idx)))
    stopifnot(all(z_idx >= 1L), all(z_idx <= p))
  }
  
  x_idx <- setdiff(seq_len(p), z_idx)
  stopifnot(length(x_idx) >= 1L)  # must retain at least one questionnaire item
  
  X0 <- Y0[, x_idx, drop = FALSE]
  Z0 <- if (length(z_idx) > 0L) Y0[, z_idx, drop = FALSE] else NULL
  
  list(X0 = X0, Z0 = Z0, x_idx = x_idx, z_idx = z_idx)
}


# ---------- stage 1: estimate follow-up→baseline reconstruction (B_t) on overlaps ----------
# For each follow-up time k:
#   - Inputs:
#       * Yt   : n_t x d_t follow-up items
#       * idx0 : length n_t mapping rows of Yt to baseline rows in Y0 (subjects in common)
#   - We fit centered OLS on the overlap subjects:
#         X0 = 1 a_t^T + Yt B_t + noise
#     Centering (subtracting means) handles the intercept:
#         (X0 - muX) ≈ (Yt - muY) B_t
#     so B_t is estimated from cross-products; a_t is recovered from means:
#         a_t = muX - muY B_t
#
# Output per timepoint:
#   - B_t    : d_t x dX reconstruction matrix (follow-up items → baseline items)
#   - a_t    : length dX intercept vector (baseline scale)
#   - beta_t : dX x d_t decoder matrix computed as B_t^+ (SVD pseudoinverse)
#
# Notes:
#   - ridge is only used to stabilize Yc'Yc when it is rank-deficient.
#   - beta_t uses a pseudoinverse so the code can proceed even when B_t is not square/full-rank.
refine_fit_B <- function(Y0, followups, z_idx = NULL,
                         ridge = 1e-6, pinv_tol = NULL) {
  stopifnot(is.matrix(Y0), is.list(followups), length(followups) >= 1L)
  stopifnot(is.numeric(ridge), ridge >= 0)
  
  sp <- refine_split_baseline(Y0, z_idx = z_idx)
  X0 <- sp$X0
  dX <- ncol(X0)
  
  Tn <- length(followups)
  
  B_list <- vector("list", Tn)
  a_list <- vector("list", Tn)
  beta_list <- vector("list", Tn)
  
  for (k in seq_len(Tn)) {
    Yt   <- followups[[k]]$Yt
    idx0 <- as.integer(followups[[k]]$idx0)
    
    stopifnot(is.matrix(Yt), length(idx0) == nrow(Yt))
    
    # Baseline questionnaire items for subjects observed at time k
    X0k <- X0[idx0, , drop = FALSE]
    
    # Center both sides so we can estimate B_t without explicitly fitting an intercept
    muX <- colMeans(X0k)
    muY <- colMeans(Yt)
    Xc <- sweep(X0k, 2, muX, "-")
    Yc <- sweep(Yt,  2, muY, "-")
    
    # Normal equations: (Yc'Yc) B_t = (Yc'Xc)
    A <- crossprod(Yc)         # d_t x d_t
    C <- crossprod(Yc, Xc)     # d_t x dX
    
    dt_k <- ncol(Yt)
    rkA <- qr(A)$rank
    if (rkA < dt_k) {
      warning(sprintf("Time %d: Yt'Yt is rank-deficient (rank=%d < dt=%d); adding ridge=%g.",
                      k, rkA, dt_k, ridge))
      A <- A + diag(ridge, dt_k)
    }
    
    # B_t: follow-up items → baseline items (in X0 coordinates)
    Bk <- solve(A, C)          # d_t x dX
    B_list[[k]] <- Bk
    
    # Diagnostics only: warn if B_t itself is low rank (can destabilize decoding)
    rkB <- qr(Bk)$rank
    if (rkB < min(dt_k, dX)) {
      warning(sprintf("Time %d: estimated B_t rank %d < min(dt=%d, dX=%d); inversion may be unstable.",
                      k, rkB, dt_k, dX))
    }
    
    # Decoder beta_t maps stabilized baseline items → follow-up items.
    # We use an SVD pseudoinverse to tolerate non-square / ill-conditioned B_t.
    beta_list[[k]] <- pinv_svd(Bk, tol = pinv_tol)  # dX x d_t
    
    # Recover intercept a_t on the original (uncentered) scale:
    #   a_t = muX - muY B_t
    a_list[[k]] <- as.numeric(muX - (muY %*% Bk))   # length dX
  }
  
  list(B_list = B_list, a_list = a_list, beta_list = beta_list,
       x_idx = sp$x_idx, z_idx = sp$z_idx)
}


# ---------- stage 2: fit nonlinear stabilizer h_t with multivariate random forests ----------
# h_t learns the supervised "preprocessing" map:
#     h_t : [X0, Z0]  ->  X0bar^(t)
# where X0bar^(t) is the follow-up–informed proxy target built on overlap subjects.
#
# Implementation detail:
#   - Uses randomForestSRC's multivariate regression:
#       cbind(y1,...,yd) ~ .
#   - X must be an n x p matrix of predictors; Y an n x d matrix of multivariate targets.
refine_fit_h_rf <- function(X, Y,
                            ntree = 1000,
                            nodesize = 5,
                            mtry = NULL,
                            bootstrap = "by.root",
                            samptype = "swr",
                            na.action = "na.omit",
                            rf_params = list()) {
  stopifnot(is.matrix(X), is.matrix(Y))
  if (!requireNamespace("randomForestSRC", quietly = TRUE)) {
    stop("Package randomForestSRC is required. Install via install.packages('randomForestSRC').")
  }
  
  p  <- ncol(X)
  dY <- ncol(Y)
  if (is.null(mtry)) mtry <- max(1L, floor(sqrt(p)))
  
  # Build a single data.frame with named outcome columns (y*) and predictor columns (x*)
  df <- data.frame(Y, X)
  colnames(df) <- c(paste0("y", seq_len(dY)), paste0("x", seq_len(p)))
  
  # Multivariate regression formula for randomForestSRC
  form <- stats::as.formula(
    paste0("cbind(", paste0("y", seq_len(dY), collapse = ","), ") ~ .")
  )
  
  # Allow user to override/extend RF settings via rf_params
  args <- modifyList(list(
    formula   = form,
    data      = df,
    ntree     = ntree,
    nodesize  = nodesize,
    mtry      = mtry,
    bootstrap = bootstrap,
    samptype  = samptype,
    na.action = na.action
  ), rf_params)
  
  rf <- do.call(randomForestSRC::rfsrc, args)
  
  structure(list(rf = rf, p = p, dY = dY), class = "refine_h_rf")
}

# Predict h_t([X0,Z0]) on new baseline inputs.
# Returns an n x d matrix of stabilized baseline item proxies (X0bar-hat).
refine_predict_h_rf <- function(h_model, X_new) {
  stopifnot(inherits(h_model, "refine_h_rf"), is.matrix(X_new))
  if (!requireNamespace("randomForestSRC", quietly = TRUE)) {
    stop("Package randomForestSRC is required. Install via install.packages('randomForestSRC').")
  }
  
  stopifnot(ncol(X_new) == h_model$p)
  
  df_new <- data.frame(X_new)
  colnames(df_new) <- paste0("x", seq_len(ncol(X_new)))
  
  pr <- stats::predict(h_model$rf, newdata = df_new)
  
  # randomForestSRC has multiple prediction formats; handle both robustly
  H <- tryCatch(
    randomForestSRC::get.mv.predicted(pr),
    error = function(e) pr$predicted
  )
  
  as.matrix(H)
}


# ---------- full training wrapper: fit {B_t, a_t, beta_t, h_t} for each follow-up time ----------
# Inputs:
#   - Y0       : N x p baseline matrix with both questionnaire items and optional nuisances
#   - followups: list over timepoints k=1..T:
#       * followups[[k]]$Yt   : n_t x d_t follow-up matrix
#       * followups[[k]]$idx0 : subject-row mapping into Y0
#   - z_idx    : baseline nuisance column indices within Y0 (used only as predictors in h_t)
#
# Procedure:
#   1) Estimate B_t (and intercept a_t) by centered OLS on overlap subjects.
#   2) Construct proxy targets on overlaps:
#         X0bar^(t) = a_t + Yt B_t
#   3) Fit nonlinear h_t: [X0,Z0] -> X0bar^(t) using multivariate RF.
#   4) Store beta_t = B_t^+ for downstream prediction.
#
# Output:
#   A "refine_model" object containing everything needed for baseline-only inference.
refine_fit <- function(Y0, followups, z_idx = NULL,
                       ridge = 1e-6, pinv_tol = NULL,
                       ntree = 1000, nodesize = 5, mtry = NULL,
                       bootstrap = "by.root", samptype = "swr",
                       na.action = "na.omit",
                       rf_params = list()) {
  
  stopifnot(is.matrix(Y0), is.list(followups), length(followups) >= 1L)
  
  sp <- refine_split_baseline(Y0, z_idx = z_idx)
  X0 <- sp$X0
  Z0 <- sp$Z0
  
  # Stage 1: linear reconstruction and decoder calibration on overlaps (X0 space only)
  Bt <- refine_fit_B(Y0, followups, z_idx = z_idx, ridge = ridge, pinv_tol = pinv_tol)
  
  Tn <- length(followups)
  dX <- ncol(X0)
  
  # For each timepoint, build:
  #   - X0bar^(t): follow-up–informed proxy in baseline item coordinates
  #   - X inputs : baseline predictors [X0, Z0] for subjects with observed follow-up at time t
  X0bar_list <- vector("list", Tn)
  Xinp_list  <- vector("list", Tn)
  
  for (k in seq_len(Tn)) {
    Yt   <- followups[[k]]$Yt
    idx0 <- as.integer(followups[[k]]$idx0)
    
    X0k <- X0[idx0, , drop = FALSE]
    Z0k <- if (!is.null(Z0)) Z0[idx0, , drop = FALSE] else NULL
    
    Bk <- Bt$B_list[[k]]   # d_t x dX
    ak <- Bt$a_list[[k]]   # length dX
    
    # Proxy target in X0 coordinates (same interpretation as baseline items)
    X0bar <- sweep(Yt %*% Bk, 2, ak, "+")   # n_t x dX
    X0bar_list[[k]] <- X0bar
    
    # Baseline predictors used by h_t (paper: h_t(X0, Z))
    Xinp_list[[k]] <- if (is.null(Z0k)) X0k else cbind(X0k, Z0k)
  }
  
  # Stage 2: fit one nonlinear stabilizer per follow-up timepoint
  h_models_list <- vector("list", Tn)
  for (k in seq_len(Tn)) {
    h_models_list[[k]] <- refine_fit_h_rf(
      X = Xinp_list[[k]],
      Y = X0bar_list[[k]],
      ntree = ntree,
      nodesize = nodesize,
      mtry = mtry,
      bootstrap = bootstrap,
      samptype = samptype,
      na.action = na.action,
      rf_params = rf_params
    )
  }
  
  structure(list(
    B_list = Bt$B_list,
    a_list = Bt$a_list,
    beta_list = Bt$beta_list,
    h_models_list = h_models_list,
    x_idx = sp$x_idx,
    z_idx = sp$z_idx,
    p_in = ncol(if (is.null(Z0)) X0 else cbind(X0, Z0)),
    dX = dX
  ), class = "refine_model")
}


# ---------- baseline-only inference: predict follow-up items at each timepoint ----------
# Given:
#   - model  : output of refine_fit()
#   - Y0_new : n x p baseline matrix with the same column layout as training Y0
#
# For each timepoint k:
#   1) Compute stabilized baseline representation:
#         Hhat = h_t([X0,Z0])    (n x dX)
#   2) Remove intercept to match the centered formulation:
#         Hc = Hhat - a_t
#   3) Decode to follow-up space:
#         Xhat_t = Hc %*% beta_t (n x d_t)
#
# Returns:
#   - list of length T, with out[[k]] an n x d_t matrix of predictions.
refine_predict <- function(model, Y0_new) {
  stopifnot(inherits(model, "refine_model"), is.matrix(Y0_new))
  
  # Recreate baseline predictors [X0,Z0] using the same column split as training
  X0_new <- Y0_new[, model$x_idx, drop = FALSE]
  Z0_new <- if (length(model$z_idx) > 0L) Y0_new[, model$z_idx, drop = FALSE] else NULL
  Xinp_new <- if (is.null(Z0_new)) X0_new else cbind(X0_new, Z0_new)
  
  stopifnot(ncol(Xinp_new) == model$p_in)
  
  Tn <- length(model$h_models_list)
  out <- vector("list", Tn)
  
  for (k in seq_len(Tn)) {
    # Stabilize baseline items in X0 coordinates
    Hhat  <- refine_predict_h_rf(model$h_models_list[[k]], Xinp_new)  # n x dX
    
    # Apply the same intercept correction used in proxy construction
    ak    <- model$a_list[[k]]                                        # length dX
    betak <- model$beta_list[[k]]                                     # dX x d_t
    
    Hc <- sweep(Hhat, 2, ak, "-")
    out[[k]] <- Hc %*% betak                                          # n x d_t
  }
  
  out
}
