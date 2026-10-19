##' Collapse STH observations into groups sharing the same worm-burden mean
##'
##' Individuals with the same location, survey time, covariate row, offset,
##' standardisation factor and MDA history have identical \eqn{\mu_W}, aggregation and prevalence
##' probability for every Monte Carlo sample of \eqn{S}. Their likelihood
##' contributions can therefore be summed exactly at group level, which shrinks
##' the TMB tape of the MCML objective from O(n * n_samples) to roughly
##' O((G + n_pos) * n_samples), with G the number of groups.
##'
##' Grouping keys use the exact hexadecimal representation of each value, so
##' only bit-identical rows are merged and the objective is unchanged.
##'
##' @return A list of data items consumed by \code{src/dsgm_tmb.cpp}.
##' @keywords internal
.sth_obs_groups <- function(y_prev, intensity_data, D, cov_offset, ID_coords,
                            survey_times, int_mat, std_factor = NULL) {

  if (is.null(std_factor)) std_factor <- rep(1, length(y_prev))
  hx  <- function(x) sprintf("%a", as.numeric(x))
  key <- do.call(paste, c(list(ID_coords, hx(survey_times), hx(cov_offset),
                               hx(std_factor)),
                          lapply(seq_len(ncol(D)),       function(j) hx(D[, j])),
                          lapply(seq_len(ncol(int_mat)), function(j) hx(int_mat[, j])),
                          sep = "|"))

  grp <- match(key, unique(key))                 # 1..G, first-occurrence order
  G   <- max(grp)

  pos <- which(y_prev == 1)
  pg  <- grp[pos]                                # group of each positive
  stopifnot(length(pg) == length(intensity_data))

  pgrp   <- sort(unique(pg))                     # groups holding positives
  pos_pg <- match(pg, pgrp)                      # 1..Gp
  Gp     <- length(pgrp)

  c_shift <- intensity_data - 1                  # NegBin branch: C - 1
  y_shift <- pmax(intensity_data - 1, 0)         # Gamma branch, as clamped in TMB
  nz      <- c_shift != 0                        # lgamma(c + phi) - lgamma(phi) = 0 at c = 0

  gsum <- function(x) as.numeric(rowsum(x, pos_pg, reorder = TRUE)[, 1])

  list(
    grp_rep      = as.integer(match(seq_len(G), grp) - 1L),
    grp_npos     = as.numeric(tabulate(grp[y_prev == 1], G)),
    grp_nneg     = as.numeric(tabulate(grp[y_prev != 1], G)),
    pgrp         = as.integer(pgrp - 1L),
    pgrp_n       = as.numeric(tabulate(pos_pg, Gp)),
    pgrp_nlg     = as.numeric(tabulate(pos_pg[nz], Gp)),
    pgrp_sumc    = gsum(c_shift),
    pgrp_const   = -gsum(lgamma(c_shift + 1)),
    pgrp_sumy    = gsum(y_shift),
    pgrp_sumlogy = gsum(log(y_shift + 1e-10)),
    lg_c         = as.numeric(c_shift[nz]),
    lg_pg        = as.integer(pos_pg[nz] - 1L),
    n_groups     = G
  )
}
