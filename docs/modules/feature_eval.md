# Per-feature diagnostics for topic models

This page defines the per-feature diagnostics written to
`{prefix}.feature_residuals.tsv` when `--residuals` is enabled for LDA or the
Gamma–Poisson topic model.

## Model overview

A topic model represents each unit as a mixture of $K$ topics and each topic
as a profile over $V$ observed features. For unit $d$, feature $w$, and topic
$k$, the fitted marginal mean has the factorized form

$$
\mu_{dw}=c_d\sum_{k=1}^K\bar\theta_{dk}\bar\beta_{kw}.
$$

In the Gamma–Poisson model, $\bar\theta_{dk}$ and $\bar\beta_{kw}$ are
nonnegative posterior means and $c_d$ is the unit exposure. In LDA,
$\bar\theta_d$ and each topic profile are normalized probability vectors,
$c_d=n_d$ is the effective unit total, and
$\mu_{dw}=n_dP(w\mid d)$ is the equivalent fixed-exposure Poisson mean for
the multinomial model.

For every positive observation, local inference also supplies a soft topic
allocation

$$
\varphi_{dwk}\propto
\exp\left\{E_q[\log\theta_{dk}]+E_q[\log\beta_{kw}]\right\},
\qquad \sum_k\varphi_{dwk}=1.
$$

## Notation

| Symbol | Meaning |
|---|---|
| $D,V,K$ | Numbers of evaluated units, model features, and topics |
| $B$ | Maximum number of units held in a processing batch |
| $\text{nnz}$ | Number of represented unit-feature entries |
| $d,w,k$ | Unit, feature, and topic indices |
| $q$ | Converged variational posterior used for local inference |
| $n_{dw}$ | Effective observed count after any feature weighting |
| $n_d=\sum_w n_{dw}$ | Effective LDA unit total |
| $c_d$ | Gamma–Poisson exposure; equal to $n_d$ in the LDA representation |
| $\bar\theta_{dk}$ | Posterior mean local topic intensity or proportion |
| $\bar\beta_{kw}$ | Fitted topic-feature factor used in the marginal mean |
| $\varphi_{dwk}$ | Variational topic allocation for a positive observation |
| $\mu_{dw}$ | Fitted marginal mean count |
| $b_k=\sum_v\bar\beta_{kv}$ | Gamma–Poisson topic capacity; $b_k=1$ for normalized LDA profiles |
| $\hat\theta_d$ | Normalized topic proportions reported for unit $d$ in $\Delta^{K-1}$ |
| $\hat\beta_{kw}=\bar\beta_{kw}/b_k$ | Normalized topic profile in $\Delta^{M-1}$ |
| $N_w=\sum_d n_{dw}$ | Observed corpus total for feature $w$ |
| $U_w=\|\{d:n_{dw}>0\}\|$ | Number of evaluated units expressing feature $w$ |
| $S_k=\sum_d c_d\bar\theta_{dk}$ | Evaluated-corpus topic exposure |
| $M_w=\sum_d\mu_{dw}$ | Predicted corpus total for feature $w$ |
| $\pi_k$ | Topic prevalence in the evaluated corpus |
| $L_d$ | Number of distinct model features with positive count in unit $d$ |
| $\text{TV}(p,q)$ | $\frac{1}{2}\sum_k\|p_k-q_k\|$, total variation distance between two distributions |

For Gamma–Poisson output, topic intensities are normalized using their topic
capacities:

$$
\hat\theta_{dk}
=\frac{\bar\theta_{dk}b_k}{\sum_j\bar\theta_{dj}b_j}.
$$

If Gamma–Poisson feature dispersion is enabled,
$\bar\epsilon_{dw}=E_q[\epsilon_{dw}]$ denotes the observation-conditioned
dispersion mean for a positive cell. Unrepresented zero entries use
$\bar\epsilon_{dw}=1$ under the sparse approximation.

## Corpus totals and sparse zero-entry accounting

The predicted feature total can be computed without visiting every zero entry:

$$
M_w=\sum_d\mu_{dw}=\sum_k\bar\beta_{kw}S_k.
$$

## Marginal feature gain (`log2Gain`)

The diagnostic-only feature multiplier is

$$
\widehat a_w=\frac{N_w}{M_w}.
$$

`log2Gain` is $\log_2\widehat a_w$. On new test data it measures the
feature-wide abundance shift required to match the evaluated corpus. On
training data it remains a marginal calibration diagnostic, but it is not
used to weight training-mode `pull`.

## Marginal Poisson deviance (`marginalDev`)

The deviance attributable to the feature-total mismatch is

$$
D_w^{\mathrm{marginal}}
=2\left[N_w\log\frac{N_w}{M_w}-(N_w-M_w)\right].
$$

Use $0\log 0=0$. When $N_w=0$, the result is $2M_w$.

## Conditional Poisson deviance (`conditionalDev`)

After applying the common multiplier $\widehat a_w$,

$$
D_w^{\mathrm{conditional}}
=2\sum_d\left[
n_{dw}\log\frac{n_{dw}}{\widehat a_w\mu_{dw}}
-(n_{dw}-\widehat a_w\mu_{dw})
\right].
$$

Because $\widehat a_wM_w=N_w$, the linear terms cancel and the sparse form is

$$
D_w^{\mathrm{conditional}}
=2\sum_{d:n_{dw}>0}
n_{dw}\log\frac{n_{dw}}{\widehat a_w\mu_{dw}}.
$$

The fixed-model per-feature deviance decomposes exactly as

$$
D_w^{\mathrm{fixed}}
=D_w^{\mathrm{marginal}}+D_w^{\mathrm{conditional}}.
$$

## Topic-direction drift (`factorDrift`)

The observed soft topic count for feature $w$ is

$$
A_{kw}=\sum_{d:n_{dw}>0}n_{dw}\varphi_{dwk}.
$$

After marginal adjustment, its expected topic-specific count is

$$
E_{kw}=\widehat a_w\bar\beta_{kw}S_k.
$$

Both topic vectors sum to $N_w$. The reported topic deviance is

$$
G_w^{\mathrm{topic}}
=2\sum_k A_{kw}\log\frac{A_{kw}}{E_{kw}}.
$$

`factorDrift` is large when occurrences of a feature are allocated across
topics differently from the fixed model after matching total abundance.

## One-step deletion influence (`deletionTV`)

Deletion diagnostics are computed only for positive counts. They approximate
leave-one-feature-out inference with one local coordinate update rather than
fully refit the unit.

### Gamma–Poisson deletion

Let $s^\theta_{dk}$ and $r^\theta_{dk}$ be the converged Gamma shape and rate
for the local topic intensity. Holding the current allocation, dispersion
mean, exposure, and topic-feature factors fixed gives

$$
\theta^{(-w)}_{dk}
=\frac{s^\theta_{dk}-n_{dw}\varphi_{dwk}}
{r^\theta_{dk}-c_d\bar\epsilon_{dw}\bar\beta_{kw}}.
$$

Normalize $\theta_d^{(-w)}$ with the same capacity factors $b_k$ used for
$\hat\theta_d$ to obtain $\hat\theta_d^{(-w)}$.

### LDA deletion

For LDA, subtract the feature allocation from the likelihood contribution to
the local Dirichlet shape:

$$
\gamma^{(-w)}_{dk}
=\max\{0,\gamma_{dk}-n_{dw}\varphi_{dwk}\}.
$$

Normalize $\gamma_d^{(-w)}$ to obtain $\hat\theta_d^{(-w)}$.

### Reported summary

For either model, define

$$
J_{dw}^{+}
=\text{TV}(\hat\theta_d^{(-w)},\hat\theta_d), \qquad \text{deletionTV}_w
=\frac{1}{U_w}\sum_{d:n_{dw}>0}J_{dw}^{+}.
$$

Thus every unit expressing the feature contributes equally, regardless of its
feature count.

The statistic excludes the effect of deleting feature $w$ on zero entries in the Gamma-Poisson model, because an all-unit one-step deletion calculation would require $O(DKV)$ arithmetic.

## Residual-weighted deleted-background leverage (`pull`)

The directional term compares the feature's own allocation with the one-step
topic mixture after removing that feature:

$$
T_{dw}
=\text{TV}(\varphi_{dw\cdot},\hat\theta_d^{(-w)}).
$$

For ordinary transform-data diagnostics,

$$
\text{pull}_w
=\frac{1}{N_w}\sum_{d:n_{dw}>0}
|n_{dw}-\widehat a_w\mu_{dw}|T_{dw}.
$$

The associated positive-count residual rate is

$$
\text{adjAbsDiffRate}_w
=\frac{1}{N_w}\sum_{d:n_{dw}>0}
|n_{dw}-\widehat a_w\mu_{dw}|.
$$

When `--use-training-prevalence` selects training-corpus evaluation, `pull`
instead uses the raw weight $|n_{dw}-\mu_{dw}|$. It is accumulated in one pass,
and `adjAbsDiffRate` is not emitted.

This is not a fully leave-one-out statistic: $\mu_{dw}$,
$\widehat a_w$, and $\varphi_{dw}$ still come from inference with all
features, while $\hat\theta_d^{(-w)}$ is only a one-step deletion update.

## Corpus-relative specificity and cofeature agreement

The topic prevalence used by standalone transforms is estimated from all
retained evaluated units:

$$
\pi_k\propto
\begin{cases}
b_kS_k, & \text{Gamma–Poisson},\\
\sum_dn_d\hat\theta_{dk}, & \text{LDA}.
\end{cases}
$$

If the model is derived from the same data, use `--use-training-prevalence` to use the already-known fitted prevalence.

Define the topic signature of feature $w$ under the evaluated prevalence and fixed topic profiles:

$$
r_{wk}=P(k\mid w;\pi,\hat\beta)
=\frac{\pi_k\hat\beta_{kw}}{m_w},
\qquad
m_w=\sum_\ell\pi_\ell\hat\beta_{\ell w}.
$$

### Topic information (`topicInformation`)

The corpus-relative topic information is

$$
I_w=D_{\mathrm{KL}}(r_w\|\pi)
=\sum_k r_{wk}\log\frac{r_{wk}}{\pi_k}.
$$

It is large when observing the feature sharply changes topic odds relative to
the evaluated corpus.

### Cofeature agreement

For a positive feature in a unit with $L_d>1$, average the signatures of the
other positive model features without count weighting:

$$
\bar r_{d,-w,k}
=\frac{1}{L_d-1}
\sum_{\substack{v:n_{dv}>0\\v\ne w}}r_{vk}.
$$

The log overlap lift is

$$
A_{dw}=\log\left(
\sum_k\frac{r_{wk}\bar r_{d,-w,k}}{\pi_k}
\right).
$$

Let $\mathcal U_w^*=\{d:n_{dw}>0,\ L_d>1\}$. The two reported summaries are

$$
\text{cofeatureCorroboration}_w
=\frac{1}{|\mathcal U_w^*|}
\sum_{d\in\mathcal U_w^*}\max(A_{dw},0),
$$

$$
\text{cofeatureConflict}_w
=\frac{1}{|\mathcal U_w^*|}
\sum_{d\in\mathcal U_w^*}\max(-A_{dw},0).
$$

These values are `NA` when no eligible unit contains a positive cofeature.
High corroboration means that the feature points toward topics supported by
other present features. High conflict means that it points in a different
direction; this can indicate noise, a secondary signal, or model
misspecification.

## Output columns

| Column | Definition |
|---|---|
| `Feature` | Feature name |
| `absDiff` | $\sum_d\|n_{dw}-\mu_{dw}\|$, including collapsed zero-entry mass |
| `absDiffRate` | `absDiff / totCount`, or `0` when `totCount` is zero |
| `totCount` | $N_w$ |
| `nUnits` | Number of units with $n_{dw}>0$ |
| `log2Gain` | $\log_2(N_w/M_w)$ |
| `marginalDev` | Deviance due to the feature-total mismatch |
| `conditionalDev` | Remaining unit-level deviance after applying $\widehat a_w$ |
| `factorDrift` | Gain-adjusted topic-allocation deviance $G_w^{\mathrm{topic}}$ |
| `deletionTV` | Mean one-step deletion effect over units expressing the feature |
| `topicInformation` | Corpus-relative KL information $I_w$ |
| `cofeatureCorroboration` | Mean positive cofeature log-overlap lift |
| `cofeatureConflict` | Mean negative cofeature log-overlap lift |
| `adjAbsDiffRate` | Positive-count gain-adjusted absolute residual rate |
| `pull` | Residual-weighted deleted-background directional leverage |

The exact schema depends on the diagnostic mode:

| Mode | Final columns |
|---|---|
| Standalone transform, full | `adjAbsDiffRate`, `pull` |
| Standalone transform, `--feature-diagnostics-cheap` | Neither column |
| Training evaluation, `--use-training-prevalence` | `pull` only; the cheap flag has no effect |

Features with zero observed total have `log2Gain=-inf`; positive-count
deletion, cofeature, and pull summaries are `NA` when their required context
is absent.

## Computational characteristics

| Calculation | Work for all features | Additional storage | zero-count effect included? |
|---|---:|---:|---|
| Corpus totals and gain | $O(DK+KV+\text{nnz})$ | $O(V+K)$ | Yes, through $M_w$ |
| Marginal and conditional deviance | $O(DK+KV+K\text{nnz})$ | $O(V)$ | Yes |
| `factorDrift` | $O(DK+KV+K\text{nnz})$ | $O(KV)$ | Expected zero mass is collapsed |
| `deletionTV` | $O(K\text{nnz})$ | $O(V)$ | No |
| `topicInformation` and cofeature agreement | $O(KV+K\text{nnz})$ | $O(KV+BK)$ plus a compact transform-time presence spool | No |
| `pull` | $O(K\text{nnz})$ | $O(V)$ for training; the transform spool for full diagnostics | No |

For standalone transforms, prevalence is unavailable until all units have
been processed, so the implementation writes a compact spool beneath
`--temp-dir`. The system temporary directory is used when the option is
omitted, and the scoped spool directory is removed automatically.
