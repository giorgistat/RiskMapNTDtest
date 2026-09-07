// =============================================================================
// dsgm_mdiag.stan
//
// Stan sampler for the doubly stochastic geostatistical model with
// multiple diagnostics.  Used to draw MCMC samples of the spatial process
// S at fixed parameter values theta_0, which are then passed to the TMB
// MCML engine (dsgm_mdiag.cpp) for parameter estimation.
//
// Two Binomial outcomes per observation, sharing a latent NB worm burden
// driven by a log-Gaussian spatial process S(x):
//
//   Parasitological (is_mf[i] == 1):
//     P(Y > 0) = 1 - [omega / (omega + mu*(1-exp(-alpha)))]^omega
//
//   Serological (is_mf[i] == 0):
//     P(Y = 1) = gamma_sens * {1 - [omega / (omega + mu)]^omega}
//
// Optional MDA effect (use_mda == 1):
//   mu_W[i] = exp(eta[i]) * mda_impact[i]
//   mda_impact is pre-computed in R via compute_mda_effect() and passed
//   as data, exactly as in the STH Stan sampler.
//
// Spatial process:
//   S = sqrt(sigma2) * L * S_raw,  S_raw ~ N(0, I)
//   R[i,j] = exp(-D_mat[i,j] / phi),  L = cholesky(R)
//   L is computed once in transformed_data since phi is fixed.
//
// All model parameters (sigma2, phi, omega, alpha, gamma_sens) are FIXED
// at their theta_0 values and passed as data.  Only S_raw is sampled.
// =============================================================================

functions {

  // P(detect >= 1 MF) via NB PGF evaluated at exp(-alpha)
  real p_mf(real mu_W, real omega, real alpha) {
    real c_alpha = 1.0 - exp(-alpha);
    real ratio   = omega / (omega + mu_W * c_alpha);
    real pr      = 1.0 - pow(ratio, omega);
    pr = fmax(pr, 1e-10);
    pr = fmin(pr, 1.0 - 1e-10);
    return pr;
  }

  // P(detect antigen) -- sensitivity-adjusted NB zero probability
  real p_cfa(real mu_W, real omega, real gamma_sens) {
    real ratio = omega / (omega + mu_W);
    real pr    = gamma_sens * (1.0 - pow(ratio, omega));
    pr = fmax(pr, 1e-10);
    pr = fmin(pr, 1.0 - 1e-10);
    return pr;
  }

}

data {

  int<lower=1> n;       // Total observations (both diagnostics)
  int<lower=1> n_loc;   // Unique spatial locations
  int<lower=1> p;       // Number of fixed-effect covariates

  // Binomial response
  array[n] int<lower=0>  y;        // Positive counts
  array[n] int<lower=1>  units_m;  // Individuals tested

  // Diagnostic type: 1 = Parasitological (MF), 0 = Serological
  array[n] int<lower=0, upper=1> is_mf;

  // Location mapping (1-indexed)
  array[n] int<lower=1, upper=n_loc> ID_coords;

  // Distance matrix between unique locations
  matrix[n_loc, n_loc] D_mat;

  // Fixed-effects linear predictor: D * beta  (computed in R, passed as data)
  vector[n] eta_fixed;

  // Fixed parameters at theta_0
  real<lower=0>          sigma2;      // GP variance
  real<lower=0>          phi;         // GP range
  real<lower=0>          omega;       // NB aggregation parameter
  real<lower=0>          alpha;       // MF detection rate per worm
  real<lower=0,upper=1>  gamma_sens;  // Serological sensitivity (fixed by user)

  // Optional MDA effect
  // use_mda = 0: no MDA; mda_impact is ignored (pass a vector of ones)
  // use_mda = 1: multiply exp(eta) by mda_impact[i]
  //   mda_impact must be pre-computed in R via compute_mda_effect() at theta_0
  int<lower=0, upper=1>  use_mda;
  vector<lower=0>[n]     mda_impact;  // MDA decay factor per obs (1.0 if use_mda=0)

}

transformed data {

  // Correlation matrix and its Cholesky factor, computed once at fixed phi.
  matrix[n_loc, n_loc] R;
  matrix[n_loc, n_loc] L;

  for (i in 1:n_loc) {
    R[i, i] = 1.0;
    for (j in (i + 1):n_loc) {
      R[i, j] = exp(-D_mat[i, j] / phi);
      R[j, i] = R[i, j];
    }
  }

  L = cholesky_decompose(R);

}

parameters {

  // Standardised spatial random effects.
  // Actual process: S = sqrt(sigma2) * L * S_raw ~ N(0, sigma2 * R)
  vector[n_loc] S_raw;

}

transformed parameters {

  vector[n_loc] S;
  vector[n]     mu_W;    // Expected worm burden per observation
  vector[n]     prob;    // Detection probability per observation

  S = sqrt(sigma2) * (L * S_raw);

  for (i in 1:n) {
    // Worm burden: fixed effects + spatial effect, optionally adjusted by MDA
    real eta_i = eta_fixed[i] + S[ID_coords[i]];
    mu_W[i] = use_mda ? exp(eta_i) * mda_impact[i] : exp(eta_i);

    // Detection probability depends on diagnostic type
    if (is_mf[i] == 1) {
      prob[i] = p_mf(mu_W[i], omega, alpha);
    } else {
      prob[i] = p_cfa(mu_W[i], omega, gamma_sens);
    }
  }

}

model {

  // Prior on standardised spatial process
  S_raw ~ std_normal();

  // Binomial likelihood -- identical form for both diagnostic types;
  // the difference is encoded in prob[] computed above.
  for (i in 1:n) {
    target += binomial_lpmf(y[i] | units_m[i], prob[i]);
  }

}
