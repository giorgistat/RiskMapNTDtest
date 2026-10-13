#include <TMB.hpp>

// =============================================================================
// dsgm_tmb.cpp
//
// TMB template for the doubly stochastic geostatistical model for joint
// prevalence-intensity modelling of soil-transmitted helminths (STH).
//
// Hierarchical model:
//   W_j(x_i) | S(x_i) ~ NegBin(mu(x_i), omega_i)
//     log mu*(x_i) = D_i * beta + S(x_i)
//     mu(x_i)      = mu*(x_i) * MDA_effect(x_i, t)
//
//   Aggregation parameter (vary_k controls which form is used):
//     vary_k = 0:  omega_i = exp(log_k)                      [constant, default]
//     vary_k = 1:  log(omega_i) = log_k + omega1*log(mu_W_i) [varies with burden]
//
//   omega1 is a free, unpenalised regression slope — well-identified from data.
//
//   Y_ij | W_j(x_i) ~ Poisson(rho * W_j(x_i))
//
// Intensity likelihood for C | C > 0 (controlled by intensity_family):
//   0 = Shifted Gamma:           C - 1 ~ Gamma(kappa_C, theta_C)
//   1 = Shifted NegBin2:         C - 1 ~ NB2(mu_C - 1, phi_C)
//
// MDA effect (exponential decay):
//   MDA_effect(i) = prod_{m: t_i > u_m} [1 - alpha_W * cov_m * exp(-(t_i-u_m)/gamma_W)]
//
// Inference: MCML with importance sampling.
//   Step 1: compute_denominator_only = 1  -> report log_f_vals at theta_0
//   Step 2: compute_denominator_only = 0  -> MCML objective
//
// Memory design (tape size of the MCML objective):
//   * R^{-1} and log|R| via atomic::matinvpd, and all quadratic forms
//     S_s' R^{-1} S_s via one atomic::matmul, as in inst/tmb/dsgm_mdiag.cpp.
//     The previous AD Cholesky plus per-sample triangular solves cost
//     O(n_loc^3 + n_samples * n_loc^2) taped operations.
//   * Observations are collapsed into groups that share mu_W (same location,
//     survey time, covariates, offset and MDA history; see
//     R/sth_obs_groups.R). Prevalence and intensity terms are summed exactly
//     per group, so the Monte Carlo loop costs O(G + n_pos) per sample instead
//     of O(n). Only the NegBin term lgamma(c + phi_C) remains per positive,
//     and only for c = C - 1 != 0.
//   The objective is identical to the ungrouped dense-Cholesky version up to
//   floating-point summation order.
// =============================================================================

template<class Type>
Type get_distance_sth(int i, int j, const vector<Type>& dist_vec, int n_loc) {
  if (i == j) return Type(0);
  if (i > j)  { int tmp = i; i = j; j = tmp; }
  int idx = i * n_loc - (i * (i + 1)) / 2 + (j - i - 1);
  return dist_vec(idx);
}

template<class Type>
Type objective_function<Type>::operator() ()
{
  // ===========================================================================
  // DATA
  // ===========================================================================

  DATA_VECTOR(y_prev);

  DATA_MATRIX(D);
  DATA_VECTOR(cov_offset);

  DATA_MATRIX(S_samples);
  DATA_IVECTOR(ID_coords);

  DATA_VECTOR(dist_vec);
  DATA_INTEGER(n_loc);

  // MDA block
  DATA_VECTOR(survey_times);
  DATA_VECTOR(mda_times);
  DATA_IVECTOR(mda_i);
  DATA_IVECTOR(mda_j);
  DATA_VECTOR(mda_coverage);
  DATA_INTEGER(n_mda_pairs);

  // Observation groups (built by .sth_obs_groups() in R)
  DATA_IVECTOR(grp_rep);       // representative observation of each group (0-based)
  DATA_VECTOR(grp_npos);       // number of egg-positives per group
  DATA_VECTOR(grp_nneg);       // number of egg-negatives per group
  DATA_IVECTOR(pgrp);          // groups holding at least one positive (0-based)
  DATA_VECTOR(pgrp_n);         // positives per positive-group
  DATA_VECTOR(pgrp_nlg);       // positives with C - 1 != 0 per positive-group
  DATA_VECTOR(pgrp_sumc);      // sum of (C - 1)
  DATA_VECTOR(pgrp_const);     // -sum lgamma(C)            [NegBin constant]
  DATA_VECTOR(pgrp_sumy);      // sum of max(C - 1, 0)      [Gamma]
  DATA_VECTOR(pgrp_sumlogy);   // sum of log(max(C-1,0)+1e-10) [Gamma]
  DATA_VECTOR(lg_c);           // C - 1 for positives with C - 1 != 0
  DATA_IVECTOR(lg_pg);         // their positive-group index (0-based)

  // Alpha penalty
  DATA_INTEGER(use_alpha_penalty);
  DATA_INTEGER(alpha_penalty_type);  // 1 = Beta(a,b); 2 = Normal on logit
  DATA_SCALAR(alpha_param1);
  DATA_SCALAR(alpha_param2);

  // Gamma_W penalty
  DATA_INTEGER(use_gamma_penalty);
  DATA_INTEGER(gamma_penalty_type);  // 1 = Gamma; 2 = Normal; 3 = Log-Normal [preferred]
  DATA_SCALAR(gamma_param1);
  DATA_SCALAR(gamma_param2);

  // Rho penalty
  DATA_INTEGER(use_rho_penalty);
  DATA_INTEGER(rho_penalty_type);    // 1 = Gamma; 2 = Normal; 3 = Log-Normal [preferred]
  DATA_SCALAR(rho_param1);
  DATA_SCALAR(rho_param2);

  // Importance sampling
  DATA_INTEGER(compute_denominator_only);
  DATA_VECTOR(log_denominator_vals);

  // Intensity likelihood family: 0 = shifted Gamma, 1 = shifted NegBin2
  DATA_INTEGER(intensity_family);

  // Aggregation flag: 0 = constant omega, 1 = omega varies with mu_W
  DATA_INTEGER(vary_k);

  // ===========================================================================
  // PARAMETERS
  // ===========================================================================

  PARAMETER_VECTOR(beta);
  PARAMETER(log_k);       // log baseline aggregation (omega_0)
  PARAMETER(log_rho);
  PARAMETER(logit_alpha);
  PARAMETER(log_gamma);
  PARAMETER(log_sigma2);
  PARAMETER(log_phi);
  PARAMETER(omega1);      // slope of log(omega) on log(mu_W); fixed at 0 when vary_k=0

  const Type k       = exp(log_k);
  const Type rho     = exp(log_rho);
  const Type alpha_W = Type(1.0) / (Type(1.0) + exp(-logit_alpha));
  const Type gamma_W = exp(log_gamma);
  const Type sigma2  = exp(log_sigma2);
  const Type phi     = exp(log_phi);

  int n         = y_prev.size();
  int n_samples = S_samples.rows();
  int G         = grp_rep.size();
  int Gp        = pgrp.size();
  int n_lg      = lg_c.size();

  // ===========================================================================
  // MDA EFFECT
  // ===========================================================================

  vector<Type> mda_effect(n);
  mda_effect.setOnes();

  for (int idx = 0; idx < n_mda_pairs; idx++) {
    int  i        = mda_i(idx);
    int  m        = mda_j(idx);
    Type coverage = mda_coverage(idx);
    Type dt       = survey_times(i) - mda_times(m);
    if (dt > Type(0))
      mda_effect(i) *= (Type(1.0) - alpha_W * coverage * exp(-dt / gamma_W));
  }

  // ===========================================================================
  // CORRELATION MATRIX R, INVERSE AND LOG-DETERMINANT
  // ===========================================================================

  matrix<Type> R(n_loc, n_loc);
  for (int i = 0; i < n_loc; i++) {
    R(i, i) = Type(1.0);
    for (int j = i + 1; j < n_loc; j++) {
      Type r  = exp(-get_distance_sth(i, j, dist_vec, n_loc) / phi);
      R(i, j) = r;
      R(j, i) = r;
    }
  }

  Type log_det_R;
  matrix<Type> Rinv = atomic::matinvpd(R, log_det_R);

  // ===========================================================================
  // BATCHED QUADRATIC FORMS   quad(s) = S_s' R^{-1} S_s
  // ===========================================================================

  matrix<Type> St = S_samples.transpose();      // n_loc x n_samples
  matrix<Type> M  = atomic::matmul(Rinv, St);   // n_loc x n_samples, one node

  vector<Type> quad(n_samples);
  for (int s = 0; s < n_samples; s++) {
    Type q = Type(0.0);
    for (int i = 0; i < n_loc; i++) q += St(i, s) * M(i, s);
    quad(s) = q;
  }

  const Type log_prior_const = Type(n_loc) * log(sigma2) + log_det_R;

  // ===========================================================================
  // FIXED-EFFECTS LINEAR PREDICTOR (group representatives only)
  // ===========================================================================

  vector<Type> mu_fixed = D * beta + cov_offset;
  const Type   c_rho    = Type(1.0) - exp(-rho);

  vector<Type> eta_g(G);   // fixed-effects predictor, per group
  vector<Type> mda_g(G);   // MDA effect, per group
  vector<int>  loc_g(G);
  for (int g = 0; g < G; g++) {
    int i    = grp_rep(g);
    eta_g(g) = mu_fixed(i);
    mda_g(g) = mda_effect(i);
    loc_g(g) = ID_coords(i);
  }

  // ===========================================================================
  // MONTE CARLO LOOP
  // ===========================================================================

  vector<Type> log_f_vals(n_samples);
  vector<Type> mu_g(G), k_g(G), praw_g(G);
  vector<Type> phiC_h(Gp);

  for (int s = 0; s < n_samples; s++) {

    Type ll = Type(0.0);

    // ----- Prevalence likelihood (per group) -----
    for (int g = 0; g < G; g++) {
      Type mu_W = exp(eta_g(g) + S_samples(s, loc_g(g))) * mda_g(g);
      Type k_i  = (vary_k == 1) ?
        exp(log_k + omega1 * log(mu_W + Type(1e-10))) : k;

      Type ratio = k_i / (k_i + mu_W * c_rho);
      Type pr_r  = Type(1.0) - pow(ratio, k_i);
      mu_g(g) = mu_W; k_g(g) = k_i; praw_g(g) = pr_r;

      Type pr = CppAD::CondExpLt(pr_r, Type(1e-10),             Type(1e-10),             pr_r);
      pr      = CppAD::CondExpGt(pr,   Type(1.0) - Type(1e-10), Type(1.0) - Type(1e-10), pr);

      if (grp_npos(g) > Type(0.5)) ll += grp_npos(g) * log(pr);
      if (grp_nneg(g) > Type(0.5)) ll += grp_nneg(g) * log(Type(1.0) - pr);
    }

    // ----- Intensity likelihood (per positive-group) -----
    for (int h = 0; h < Gp; h++) {
      int  g    = pgrp(h);
      Type mu_i = mu_g(g);
      Type k_i  = k_g(g);
      Type pr_i = CppAD::CondExpLt(praw_g(g), Type(1e-10), Type(1e-10), praw_g(g));

      Type mu_C     = (rho * mu_i) / pr_i;
      Type sigma2_C = (rho * mu_i * (Type(1.0) + rho)) / pr_i
        + (rho * rho * mu_i * mu_i / pr_i)
        * (Type(1.0)/k_i + Type(1.0) - Type(1.0)/pr_i);
      sigma2_C = CppAD::CondExpLt(sigma2_C, Type(1e-6), Type(1e-6), sigma2_C);

      if (intensity_family == 0) {

        // Shifted Gamma: C - 1 ~ Gamma(kappa_C, theta_C), summed over the group
        Type mu_C1 = mu_C - Type(1.0);
        mu_C1 = CppAD::CondExpLt(mu_C1, Type(0.1), Type(0.1), mu_C1);

        Type kappa_C = (mu_C1 * mu_C1) / sigma2_C;
        Type theta_C = sigma2_C / mu_C1;
        kappa_C = CppAD::CondExpLt(kappa_C, Type(1e-6), Type(1e-6), kappa_C);
        theta_C = CppAD::CondExpLt(theta_C, Type(1e-6), Type(1e-6), theta_C);

        ll += (kappa_C - Type(1.0)) * pgrp_sumlogy(h)
          - pgrp_sumy(h) / theta_C
          - pgrp_n(h) * (kappa_C * log(theta_C) + lgamma(kappa_C));

      } else {

        // Shifted NegBin2: C - 1 ~ NB2(mu_C1, phi_C), group-level terms
        Type mu_C1 = mu_C - Type(1.0);
        mu_C1 = CppAD::CondExpLt(mu_C1, Type(0.1), Type(0.1), mu_C1);

        Type denom_nb = sigma2_C - mu_C1;
        denom_nb = CppAD::CondExpLt(denom_nb, Type(1e-6), Type(1e-6), denom_nb);
        Type phi_C = mu_C1 * mu_C1 / denom_nb;
        phi_C = CppAD::CondExpLt(phi_C, Type(1e-4), Type(1e-4), phi_C);
        phiC_h(h) = phi_C;

        Type log_r   = log(phi_C) - log(phi_C + mu_C1);
        Type log_1mr = log(mu_C1) - log(phi_C + mu_C1);

        ll += - pgrp_nlg(h) * lgamma(phi_C)
          + pgrp_n(h) * phi_C * log_r
          + pgrp_sumc(h) * log_1mr
          + pgrp_const(h);
      }
    }

    // ----- NegBin per-positive term lgamma(c + phi_C), c = C - 1 != 0 -----
    if (intensity_family != 0) {
      for (int j = 0; j < n_lg; j++)
        ll += lgamma(lg_c(j) + phiC_h(lg_pg(j)));
    }

    // ----- GP prior (quad(s) precomputed above) -----
    Type log_prior = -Type(0.5) * (log_prior_const + quad(s) / sigma2);

    Type log_num  = ll + log_prior;
    log_f_vals(s) = compute_denominator_only ?
    log_num : log_num - log_denominator_vals(s);
  }

  if (compute_denominator_only) {
    REPORT(log_f_vals);
    return Type(0);
  }

  // ===========================================================================
  // MC LOG-LIKELIHOOD (log-sum-exp stable)
  // ===========================================================================

  Type max_lf     = log_f_vals.maxCoeff();
  vector<Type> ef = exp(log_f_vals - max_lf);
  Type mc_loglik  = log(ef.mean()) + max_lf;

  // ===========================================================================
  // PENALTIES  (alpha_W, gamma_W, rho only — omega1 is free)
  // ===========================================================================

  Type penalty = Type(0.0);

  if (use_alpha_penalty == 1) {
    if (alpha_penalty_type == 1) {
      penalty -= (alpha_param1 - Type(1.0)) * log(alpha_W);
      penalty -= (alpha_param2 - Type(1.0)) * log(Type(1.0) - alpha_W);
    } else if (alpha_penalty_type == 2) {
      Type d = logit_alpha - alpha_param1;
      penalty += Type(0.5) * d * d / (alpha_param2 * alpha_param2);
    }
  }

  if (use_gamma_penalty == 1) {
    if (gamma_penalty_type == 1) {
      penalty -= (gamma_param1 - Type(1.0)) * log(gamma_W);
      penalty += gamma_param2 * gamma_W;
    } else if (gamma_penalty_type == 2) {
      Type d = gamma_W - gamma_param1;
      penalty += Type(0.5) * d * d / (gamma_param2 * gamma_param2);
    } else if (gamma_penalty_type == 3) {
      Type d = log_gamma - gamma_param1;
      penalty += Type(0.5) * d * d / (gamma_param2 * gamma_param2);
    }
  }

  if (use_rho_penalty == 1) {
    if (rho_penalty_type == 1) {
      penalty -= (rho_param1 - Type(1.0)) * log(rho);
      penalty += rho_param2 * rho;
    } else if (rho_penalty_type == 2) {
      Type d = rho - rho_param1;
      penalty += Type(0.5) * d * d / (rho_param2 * rho_param2);
    } else if (rho_penalty_type == 3) {
      Type d = log_rho - rho_param1;
      penalty += Type(0.5) * d * d / (rho_param2 * rho_param2);
    }
  }

  Type nll = -(mc_loglik - penalty);

  ADREPORT(k);
  ADREPORT(rho);
  ADREPORT(alpha_W);
  ADREPORT(gamma_W);
  ADREPORT(sigma2);
  ADREPORT(phi);
  ADREPORT(omega1);

  return nll;
}
