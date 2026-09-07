#include <TMB.hpp>

// =============================================================================
// dsgm_mdiag.cpp
//
// TMB template for the doubly stochastic geostatistical model with
// multiple diagnostics (lf_mdiag).
//
// Two Binomial outcomes per observation sharing a latent NB worm burden:
//
//   Parasitological (is_mf == 1):
//     P(Y > 0) = 1 - [omega / (omega + mu*(1-exp(-alpha)))]^omega
//
//   Serological (is_mf == 0):
//     P(Y = 1) = gamma_sens * {1 - [omega / (omega + mu)]^omega}
//
// Optional MDA effect (use_mda == 1):
//   mu_W(i) = mu_W_star(i) * phi(i)
//   phi(i)  = prod_{m: t_i > u_m} [1 - alpha_W * cov_m * exp(-(t_i-u_m)/gamma_W)]
//
// Penalties (active when use_mda == 1):
//   alpha_W : Beta(a,b) or Normal on logit scale
//   gamma_W : Gamma(shape,rate) or Normal
//
// Inference: MCML with importance sampling.
//   Step 1: compute_denominator_only = 1  -> report log_f_vals at theta_0
//   Step 2: compute_denominator_only = 0  -> MCML objective
//
// -----------------------------------------------------------------------------
// MEMORY / TAPE NOTES (rewrite of the original dense-Cholesky version)
// -----------------------------------------------------------------------------
// The previous implementation formed an Eigen::LLT of the n_loc x n_loc
// correlation matrix in AD arithmetic, then performed one triangular solve
// *per Monte Carlo sample*. Because every scalar operation of a dense
// factorisation and of every solve is recorded on the CppAD tape, the tape
// grew like
//
//     O(n_loc^3)  +  O(n_samples * n_loc^2)
//
// For n_loc = 1214 and n_samples = 1000 that is ~1e9 recorded operations,
// which exhausts memory (std::bad_alloc) at MakeADFun time.
//
// This version instead:
//   (1) uses atomic::matinvpd() for the inverse and log-determinant, so the
//       whole factorisation is a SINGLE tape node with an analytic reverse
//       rule (exact derivatives, not an approximation);
//   (2) computes ALL quadratic forms in one batched atomic matrix product
//         M = R^{-1} S^T          (n_loc x n_samples, one tape node)
//       and then recovers quad(s) = sum_i S(s,i) * M(i,s), which is only
//       O(n_samples * n_loc) taped multiply-adds;
//   (3) replaces pow(a, omega) with exp(omega * log(a)) and hoists log(omega),
//       1/phi and 1/sigma2 out of the inner loops.
//
// Resulting tape is dominated by the Binomial likelihood loop, O(n * n_samples),
// rather than by the linear algebra -- roughly a 50-70x reduction in tape size.
//
// NOTE: atomic::matinvpd forms an explicit inverse rather than keeping a
// Cholesky factor. With the nugget on the diagonal (R(i,i) = 1 + nu2) this is
// well conditioned in normal use, but if nu2 is driven to ~0 while phi is large
// the correlation matrix can become near-singular. If that occurs, bound nu2
// away from zero (or fix tau2) rather than reverting to the dense LLT.
// =============================================================================

template<class Type>
Type get_distance(int i, int j, const vector<Type>& dist_vec, int n_loc) {
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

  DATA_VECTOR(y);           // Binomial counts (cases), length n
  DATA_VECTOR(units_m);     // Binomial denominators, length n
  DATA_IVECTOR(is_mf);      // 1 = Parasitological (MF), 0 = Serological, length n
  DATA_MATRIX(D);           // Covariate matrix, n x p
  DATA_VECTOR(cov_offset);  // Linear predictor offset, length n

  DATA_MATRIX(S_samples);   // MCMC samples of S, n_samples x n_loc
  DATA_IVECTOR(ID_coords);  // 0-indexed location ID for each obs, length n

  DATA_VECTOR(dist_vec);    // Compressed upper-triangle distance vector
  DATA_INTEGER(n_loc);      // Number of unique spatial locations

  DATA_SCALAR(gamma_sens);  // Fixed sensitivity of serological test

  DATA_INTEGER(fix_omega);        // 0 = estimate omega, 1 = fix to fixed_omega_val
  DATA_SCALAR(fixed_omega_val);

  // --- MDA block (only used when use_mda == 1) --------------------------------
  DATA_INTEGER(use_mda);
  DATA_VECTOR(survey_times);      // survey time per obs, length n
  DATA_VECTOR(mda_times);         // MDA round times, length n_mda
  DATA_IVECTOR(mda_i);            // row indices (0-indexed) of non-zero int_mat
  DATA_IVECTOR(mda_j);            // col indices (0-indexed) of non-zero int_mat
  DATA_VECTOR(mda_coverage);      // coverage values for non-zero entries
  DATA_INTEGER(n_mda_pairs);      // number of non-zero entries

  // Penalties on MDA parameters
  DATA_INTEGER(use_alpha_W_penalty);
  DATA_INTEGER(alpha_W_penalty_type);  // 1 = Beta on alpha_W; 2 = Normal on logit_alpha_W
  DATA_SCALAR(alpha_W_param1);
  DATA_SCALAR(alpha_W_param2);

  DATA_INTEGER(use_gamma_W_penalty);
  DATA_INTEGER(gamma_W_penalty_type);  // 1 = Gamma on gamma_W; 2 = Normal on gamma_W
  DATA_SCALAR(gamma_W_param1);
  DATA_SCALAR(gamma_W_param2);

  // Importance sampling
  DATA_INTEGER(compute_denominator_only);
  DATA_VECTOR(log_denominator_vals);  // length n_samples

  // ===========================================================================
  // PARAMETERS
  // ===========================================================================

  PARAMETER_VECTOR(beta);      // fixed-effect coefficients
  PARAMETER(log_sigma2);
  PARAMETER(log_phi);
  PARAMETER(log_nu2);          // log(tau2 / sigma2)
  PARAMETER(log_omega);        // ignored when fix_omega == 1
  PARAMETER(log_alpha);        // log MF detection rate per worm

  // MDA parameters (declared always; inactive when use_mda == 0)
  PARAMETER(logit_alpha_W);    // logit immediate worm burden drop in (0,1)
  PARAMETER(log_gamma_W);      // log recovery rate

  // Natural-scale transforms
  const Type sigma2  = exp(log_sigma2);
  const Type phi     = exp(log_phi);
  const Type nu2     = exp(log_nu2);
  const Type omega   = (fix_omega == 1) ? fixed_omega_val : exp(log_omega);
  const Type alpha   = exp(log_alpha);
  const Type alpha_W = Type(1.0) / (Type(1.0) + exp(-logit_alpha_W));
  const Type gamma_W = exp(log_gamma_W);

  int n         = y.size();
  int n_samples = S_samples.rows();

  // ===========================================================================
  // MDA EFFECT  phi(i)
  // ===========================================================================

  vector<Type> mda_effect(n);
  mda_effect.setOnes();

  if (use_mda == 1) {
    for (int idx = 0; idx < n_mda_pairs; idx++) {
      int  i        = mda_i(idx);
      int  m        = mda_j(idx);
      Type coverage = mda_coverage(idx);
      Type dt       = survey_times(i) - mda_times(m);
      if (dt > Type(0))
        mda_effect(i) *= (Type(1.0) - alpha_W * coverage * exp(-dt / gamma_W));
    }
  }

  // ===========================================================================
  // CORRELATION MATRIX R
  //
  // Built once. Hoisting 1/phi out of the loop keeps this to ~2 taped
  // operations per off-diagonal element (one multiply, one exp).
  // ===========================================================================

  const Type inv_phi = Type(1.0) / phi;

  matrix<Type> R(n_loc, n_loc);
  for (int i = 0; i < n_loc; i++) {
    R(i, i) = Type(1.0) + nu2;
    for (int j = i + 1; j < n_loc; j++) {
      Type r  = exp(-get_distance(i, j, dist_vec, n_loc) * inv_phi);
      R(i, j) = r;
      R(j, i) = r;
    }
  }

  // ---------------------------------------------------------------------------
  // Atomic inverse + log-determinant.
  //
  // atomic::matinvpd() records the entire positive-definite factorisation as a
  // single tape node, supplying its own analytic reverse-mode rule. Derivatives
  // w.r.t. phi, nu2 remain exact; the O(n_loc^3) scalar operations are executed
  // in double precision and never taped.
  //
  // Sign convention matches TMB's density::MVNORM_t: log_det_R = log|R|.
  // ---------------------------------------------------------------------------
  Type log_det_R;
  matrix<Type> Rinv = atomic::matinvpd(R, log_det_R);

  // ===========================================================================
  // BATCHED QUADRATIC FORMS
  //
  //   quad(s) = S_s' R^{-1} S_s
  //
  // Computing these one at a time costs O(n_loc^2) taped operations per sample.
  // Instead form
  //     M = R^{-1} S^T        (n_loc x n_samples)
  // as ONE atomic matrix product, then contract:
  //     quad(s) = sum_i S^T(i,s) * M(i,s)
  // which is only O(n_samples * n_loc) taped multiply-adds. S_samples is data,
  // so each term is (constant * AD) -- a single node.
  // ===========================================================================

  matrix<Type> St = S_samples.transpose();      // n_loc x n_samples
  matrix<Type> M  = atomic::matmul(Rinv, St);   // n_loc x n_samples, one node

  vector<Type> quad(n_samples);
  for (int s = 0; s < n_samples; s++) {
    Type q = Type(0.0);
    for (int i = 0; i < n_loc; i++) q += St(i, s) * M(i, s);
    quad(s) = q;
  }

  // ===========================================================================
  // FIXED-EFFECTS LINEAR PREDICTOR
  // ===========================================================================

  vector<Type> mu_fixed = D * beta + cov_offset;

  const Type c_alpha    = Type(1.0) - exp(-alpha);
  const Type log_omega_v = log(omega);      // hoisted out of the sample loop
  const Type inv_sigma2  = Type(1.0) / sigma2;
  const Type eps         = Type(1e-10);

  // Constant part of the GP log-prior (independent of s)
  const Type log_prior_const = Type(n_loc) * log_sigma2 + log_det_R;

  // ===========================================================================
  // MONTE CARLO LOOP
  //
  // This is now the dominant contributor to the tape, at O(n * n_samples).
  // pow(a, omega) is written as exp(omega * log(a)) to avoid CppAD's generic
  // pow expansion, and log(omega) is hoisted above.
  // ===========================================================================

  vector<Type> log_f_vals(n_samples);

  for (int s = 0; s < n_samples; s++) {

    Type ll = Type(0.0);

    for (int i = 0; i < n; i++) {

      Type mu_W = exp(mu_fixed(i) + S_samples(s, ID_coords(i))) * mda_effect(i);

      Type p;
      if (is_mf(i) == 1) {
        // Parasitological: 1 - [omega/(omega + mu*c_alpha)]^omega
        p = Type(1.0) - exp(omega * (log_omega_v - log(omega + mu_W * c_alpha)));
      } else {
        // Serological: gamma_sens * {1 - [omega/(omega + mu)]^omega}
        p = gamma_sens * (Type(1.0) - exp(omega * (log_omega_v - log(omega + mu_W))));
      }

      p = CppAD::CondExpLt(p, eps, eps, p);
      p = CppAD::CondExpGt(p, Type(1.0) - eps, Type(1.0) - eps, p);

      ll += y(i) * log(p) + (units_m(i) - y(i)) * log(Type(1.0) - p);
    }

    Type log_prior = -Type(0.5) * (log_prior_const + quad(s) * inv_sigma2);
    Type log_num   = ll + log_prior;

    log_f_vals(s) = compute_denominator_only ?
    log_num : log_num - log_denominator_vals(s);
  }

  // ===========================================================================
  // DENOMINATOR PASS: report and exit
  // ===========================================================================

  if (compute_denominator_only) {
    REPORT(log_f_vals);
    return Type(0);
  }

  // ===========================================================================
  // MC LOG-LIKELIHOOD (log-sum-exp stable)
  // ===========================================================================

  Type max_lf      = log_f_vals.maxCoeff();
  vector<Type> ef  = exp(log_f_vals - max_lf);
  Type mc_loglik   = log(ef.mean()) + max_lf;

  // ---------------------------------------------------------------------------
  // Importance-sampling effective sample size, reported for diagnostics.
  //   ESS = (sum w)^2 / sum(w^2),  w_s = exp(log_f_vals(s) - max)
  // A collapse of ESS relative to n_samples indicates the optimiser has drifted
  // too far from theta_0 for the current proposal to support it.
  // ---------------------------------------------------------------------------
  Type sum_w   = ef.sum();
  Type sum_w2  = (ef * ef).sum();
  Type ess     = (sum_w * sum_w) / sum_w2;
  REPORT(ess);
  REPORT(log_f_vals);

  // ===========================================================================
  // MDA PENALTIES
  // ===========================================================================

  Type penalty = Type(0.0);

  if (use_mda == 1) {

    if (use_alpha_W_penalty == 1) {
      if (alpha_W_penalty_type == 1) {
        // Beta(a, b) on alpha_W
        penalty -= (alpha_W_param1 - Type(1.0)) * log(alpha_W);
        penalty -= (alpha_W_param2 - Type(1.0)) * log(Type(1.0) - alpha_W);
      } else if (alpha_W_penalty_type == 2) {
        // Normal(mean, sd) on logit(alpha_W)
        Type d = logit_alpha_W - alpha_W_param1;
        penalty += Type(0.5) * d * d / (alpha_W_param2 * alpha_W_param2);
      }
    }

    if (use_gamma_W_penalty == 1) {
      if (gamma_W_penalty_type == 1) {
        // Gamma(shape, rate) on gamma_W
        penalty -= (gamma_W_param1 - Type(1.0)) * log(gamma_W);
        penalty += gamma_W_param2 * gamma_W;
      } else if (gamma_W_penalty_type == 2) {
        Type d = gamma_W - gamma_W_param1;
        penalty += Type(0.5) * d * d / (gamma_W_param2 * gamma_W_param2);
      } else if (gamma_W_penalty_type == 3) {
        Type d = log_gamma_W - gamma_W_param1;
        penalty += Type(0.5) * d * d / (gamma_W_param2 * gamma_W_param2);
      }
    }
  }

  Type nll = -(mc_loglik - penalty);

  // ===========================================================================
  // REPORT ON NATURAL SCALE
  // ===========================================================================

  ADREPORT(sigma2);
  ADREPORT(phi);
  ADREPORT(omega);
  ADREPORT(alpha);
  Type tau2 = nu2 * sigma2;
  ADREPORT(tau2);

  if (use_mda == 1) {
    ADREPORT(alpha_W);
    ADREPORT(gamma_W);
  }

  return nll;
}
