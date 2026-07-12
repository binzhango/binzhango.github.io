---
title: "Statistical Tests for Data Analysis: A Practical Onboarding Guide"
description: A senior data scientist's field guide to framing hypotheses, selecting statistical tests, checking assumptions, interpreting results, and communicating decisions.
excerpt: Learn how to choose and use statistical tests without turning analysis into a p-value checklist—from experimental design and assumptions to effect sizes, confidence intervals, and clear recommendations.
authors:
  - BZ
date: 2026-07-12
categories:
  - DATA SCIENCE
tags:
  - statistics
  - hypothesis testing
  - experimentation
  - data analysis
  - onboarding
---

Statistical tests are not buttons we press after looking at a dashboard. They are tools for answering **well-defined questions under explicit assumptions**.

This chapter is the guide I would give a data scientist joining my team. The familiar decision tree—compare means, compare proportions, study relationships, compare variances, or check a distribution—is a useful entry point. The real work begins one level earlier: understanding how the data was generated and what decision the analysis must support.

> **The test is rarely the first decision. First define the question, the unit of analysis, the estimand, and the data-generating process.**

## What you should be able to do after this chapter

By the end, you should be able to:

- translate a business question into a statistical hypothesis;
- identify the outcome type and whether observations are independent or paired;
- select a reasonable test and explain why it fits;
- check assumptions without applying normality tests mechanically;
- report an effect size and confidence interval alongside a p-value;
- recognize common failure modes such as pseudoreplication and multiple testing;
- write a conclusion that is useful to a decision-maker.

## The mental model: question first, test second

Most common analyses fit into five families.

| Question family | Typical business question | Outcome or inputs | Common starting point |
| --- | --- | --- | --- |
| **Compare means** | Did the new checkout reduce completion time? | Numeric outcome, categorical group | t-test or ANOVA |
| **Compare proportions** | Did conversion rate improve? | Binary outcome summarized as counts | Two-proportion test or chi-square test |
| **Check a relationship** | Does usage increase with tenure? | Two numeric variables, or two categorical variables | Correlation or chi-square independence test |
| **Compare variances** | Did process variability decrease? | Numeric outcome across groups | Levene's test or variance-ratio test |
| **Assess a distribution** | Do observed categories match the planned mix? | Counts or a numeric sample | Chi-square goodness-of-fit or distribution diagnostics |

The animation below preserves the reference chart as a dense field guide. Begin with the statistical question, choose one of the five analysis families, and read down that column to narrow the method by design, assumptions, sample structure, and robust alternatives.

<figure class="statistical-workflow">
  <video autoplay loop muted playsinline preload="metadata" poster="/assets/animations/statistical-tests/statistical-test-workflow-poster.png" aria-label="Animated workflow for choosing and reporting a statistical test">
    <source src="/assets/animations/statistical-tests/statistical-test-workflow.mp4" type="video/mp4" />
    <img src="/assets/animations/statistical-tests/statistical-test-workflow.gif" alt="Statistical test workflow moving from question and study design through test selection, assumption validation, and decision reporting" />
  </video>
  <figcaption>Follow one highlighted column at a time. The cards preserve the major parametric, rank-based, exact, variance, and distribution choices from the source chart; study design still determines the final method.</figcaption>
</figure>

This is a map, not an autopilot. Regression, hierarchical models, time-series methods, survival analysis, causal inference, and Bayesian models are often better choices once the design becomes more realistic.

## Step 1: write the question before touching the data

A good analysis starts with one sentence:

> Among **[population]**, what is the effect or association of **[exposure/intervention]** on **[outcome]** over **[time period]**, compared with **[baseline/control]**?

Suppose a product team asks, “Did the redesigned onboarding work?” That is too vague. A testable version is:

> Among newly registered US users in July, what is the difference in seven-day activation rate between the redesigned onboarding flow and the current flow?

Now define:

- **Population:** newly registered US users in July.
- **Experimental unit:** the user, not the click or event row.
- **Treatment:** redesigned onboarding.
- **Control:** current onboarding.
- **Outcome:** activated within seven days, a binary variable.
- **Estimand:** treatment activation rate minus control activation rate.
- **Direction:** usually two-sided unless a one-sided test was justified before seeing data.

This framing immediately suggests a comparison of two proportions. More importantly, it prevents us from accidentally treating thousands of events from the same user as thousands of independent observations.

## Step 2: understand how the data was generated

Before selecting a test, answer these questions.

### What is the unit of analysis?

The unit is the entity that can be considered an observation: user, account, store, machine, patient, or day. Repeated rows do not automatically create independent evidence.

If treatment is assigned by store but the analysis treats every customer as independently assigned, standard errors will usually be too small. This is **pseudoreplication**. Aggregate to the assignment unit or use a clustered or hierarchical model.

### Are observations independent, paired, repeated, or clustered?

- **Independent:** different users are assigned to treatment and control.
- **Paired:** the same user is measured before and after a change.
- **Repeated:** each user contributes measurements at many times.
- **Clustered:** users belong to stores, classrooms, households, or markets.

Paired and repeated data contain within-subject correlation. An independent-samples test ignores that structure. For two time points, a paired test may be enough. For several time points or unequal follow-up, consider a mixed-effects model or generalized estimating equations.

### Was the intervention randomized?

Randomization supports a causal interpretation because it breaks systematic links between treatment assignment and potential outcomes. Observational data can establish association, but a simple hypothesis test does not remove confounding.

If the data is observational, ask which variables influence both exposure and outcome. Regression adjustment, matching, weighting, interrupted time series, or another causal design may be needed.

### Was the outcome and analysis plan specified in advance?

Looking at many metrics and reporting only the smallest p-value inflates false positives. Before analysis, document the primary outcome, important segments, exclusion rules, minimum detectable effect, stopping rule, and planned statistical method.

## Step 3: classify the variables

Test selection depends on what the variables mean, not how they happen to be stored.

| Type | Examples | Useful summaries |
| --- | --- | --- |
| **Continuous numeric** | revenue, latency, temperature | mean, median, SD, IQR, quantiles |
| **Count** | purchases, incidents, visits | rate, mean count, exposure time |
| **Binary** | converted/not converted | proportion, risk difference, risk ratio |
| **Nominal categorical** | channel, region, plan | frequency and proportion |
| **Ordinal** | satisfaction level, severity | ordered frequencies, median, cumulative proportion |
| **Time-to-event** | time to churn or failure | survival curve, hazard, censoring rate |

Counts, ordinal outcomes, and time-to-event data often need purpose-built models. For example, a t-test on purchase counts can be misleading when most users purchase zero times and a few purchase many times.

## Step 4: state the hypotheses and estimand

The **null hypothesis** usually states no difference or no association. The **alternative hypothesis** states what would count as evidence against the null.

For a two-group mean comparison:

$$
H_0: \mu_T - \mu_C = 0
$$

$$
H_1: \mu_T - \mu_C \ne 0
$$

The quantity $\mu_T - \mu_C$ is the **estimand**: the population-level effect we want to estimate. The estimate from our sample might be $\bar{x}_T - \bar{x}_C$.

Do not let the null hypothesis define the whole analysis. A difference of exactly zero is rarely the business question. The more useful question is whether the plausible effect is large enough to matter.

## Choosing among the common tests

### Comparing a numeric outcome between two independent groups

Use **Welch's t-test** as the usual default for comparing two independent means. Unlike the classic pooled t-test, it does not assume equal population variances and performs well when sample sizes differ.

Key conditions:

- observations are independent within and between groups;
- the outcome is numeric;
- each group is reasonably well behaved, or the sample is large enough for the sampling distribution of the mean to be stable;
- extreme outliers are investigated.

If the outcome is strongly skewed, first ask whether the **mean** is still the estimand the business cares about. The Mann–Whitney U test is not simply “a t-test for medians.” It tests whether values from one group tend to rank higher than values from the other; interpreting it as a median difference requires additional shape assumptions.

Good alternatives include a bootstrap confidence interval for a mean or median difference, a permutation test aligned with the estimand, or a regression model using a suitable distribution.

### Comparing a numeric outcome across more than two groups

Use **one-way ANOVA** when comparing means across three or more independent groups under roughly equal variances. Use **Welch ANOVA** when variances or sample sizes differ.

The global test asks whether all population means are equal. A significant result does not tell you which groups differ. Follow it with planned contrasts or an appropriate post-hoc procedure:

- Tukey HSD after standard ANOVA for all pairwise comparisons;
- Games–Howell after Welch ANOVA when variances differ;
- pre-specified contrasts when only certain comparisons matter.

For a rank-based alternative, the **Kruskal–Wallis test** assesses whether group distributions differ in rank location. Significant results still require corrected post-hoc comparisons.

### Comparing paired or related measurements

Use a **paired t-test** for a numeric before/after or matched-pair outcome. The normality condition applies to the **within-pair differences**, not separately to the before and after values.

Use the **Wilcoxon signed-rank test** when a rank-based analysis of paired differences is appropriate. It assumes the distribution of differences is symmetric. If that assumption is implausible, a sign test is more robust but less powerful.

Be careful with uncontrolled before/after studies: a paired test measures change, but it does not prove the intervention caused the change. Seasonality, regression to the mean, and other concurrent events remain possible explanations.

### Comparing two proportions

Use a **two-proportion z-test** for independent binary outcomes when expected counts are sufficiently large. It is common in A/B tests of conversion or retention.

Report the result on a scale stakeholders understand:

- **absolute difference:** $p_T - p_C$;
- **relative lift:** $(p_T - p_C) / p_C$;
- **risk ratio:** $p_T / p_C$.

The absolute difference should usually be primary because it connects directly to incremental outcomes. “A 10% lift” can sound impressive while representing a move from 1.0% to only 1.1%.

For a small $2 \times 2$ table with sparse expected counts, use **Fisher's exact test**. For paired binary outcomes, use **McNemar's test**, which is missing from many simplified decision charts.

### Comparing proportions across several groups

Use a **chi-square test of homogeneity** to ask whether the distribution of a categorical outcome is the same across several populations or treatments.

The calculation is identical to a chi-square independence test; the distinction is in study design and interpretation. When expected cell counts are sparse, combine categories only with domain justification or use an exact or simulation-based method.

### Relationships between two numeric variables

Use **Pearson correlation** for the strength of a linear relationship. Inspect a scatterplot first: a correlation near zero can hide a strong curved relationship, and one outlier can dominate the coefficient.

Use **Spearman rank correlation** for a monotonic relationship or when ranks are more defensible than raw distances. Spearman correlation can detect a consistently increasing nonlinear relationship, but it will not capture every form of dependence.

Remember:

- correlation is symmetric; prediction is not;
- correlation does not control for confounding;
- a high correlation does not imply agreement;
- repeated observations from the same unit violate the usual independence calculation.

When the goal is prediction or adjustment, regression is normally more informative than a standalone correlation coefficient.

### Relationships between categorical variables

Use a **chi-square test of independence** to ask whether two categorical variables are associated within one population. Use Fisher's exact test for a sparse $2 \times 2$ table.

Report an effect size such as **Cramér's V** in addition to the p-value. With a very large sample, a negligible association can be highly significant.

### Comparing variances

Variance questions matter in manufacturing, service-level reliability, and risk—not only as assumption checks.

- The **F-test for equal variances** is sensitive to departures from normality.
- **Levene's test** is more robust.
- The **Brown–Forsythe test**, which centers groups at their medians, is more resistant to skew and outliers.

Do not automatically run a variance test and use its p-value to decide between pooled and Welch's t-test. This two-stage procedure adds instability. If equal variance is not guaranteed by design or strong knowledge, Welch's t-test is a sound default.

### Goodness of fit and distribution checks

A **chi-square goodness-of-fit test** compares observed category counts with a pre-specified expected distribution—for example, whether traffic follows a planned 50/30/20 channel allocation.

For numeric distributions:

- **Shapiro–Wilk** is a test of normality and is often used for small or moderate samples;
- **Anderson–Darling** gives more weight to tail differences;
- **Kolmogorov–Smirnov** compares an empirical distribution with a fully specified reference distribution, or compares two empirical distributions.

In practice, normality tests are frequently misused. With a huge sample, they detect harmless deviations; with a tiny sample, they may miss important ones. Use a histogram, Q–Q plot, sample size, outlier review, and knowledge of the measurement process together.

## Assumptions: check the ones that can invalidate the answer

The most consequential assumption is usually **independence**, not perfect normality.

Use this review order:

1. **Design:** Were assignment, sampling, and exclusions valid?
2. **Unit:** Is each independent unit represented correctly?
3. **Missingness:** Why is data missing, and does missingness differ by group?
4. **Measurement:** Is the metric defined and collected consistently?
5. **Shape and outliers:** Are skew, heavy tails, bounds, or influential points material?
6. **Model-specific assumptions:** Linearity, proportional hazards, equal variance, or expected cell counts.

Never remove an outlier only because it makes the result inconvenient. Determine whether it is a data error, a valid rare event, or evidence that the chosen model is inadequate. Report sensitivity analyses when a conclusion depends on a few observations.

## Statistical significance is not business significance

A p-value answers a narrow conditional question: assuming the null hypothesis and model are true, how surprising is a result at least this incompatible with the null?

It is not:

- the probability that the null hypothesis is true;
- the probability that the result happened by chance;
- the size or importance of the effect;
- proof that the measurement or experimental design was valid.

Every test result should be accompanied by:

- the estimated effect;
- a confidence interval;
- the sample size and unit of analysis;
- the method and important assumptions;
- a plain-language decision or next action.

For continuous outcomes, consider a raw mean difference and a standardized measure such as Hedges' $g$. For binary outcomes, use an absolute difference and often a risk ratio. For categorical association, consider Cramér's V. The raw effect on the business scale should usually come first.

## Power and sample size: plan before collecting data

**Statistical power** is the probability that a planned procedure detects an effect of a specified size when that effect exists. Before an experiment, agree on:

- significance level $\alpha$;
- desired power, often 80% or 90%;
- baseline rate or variance;
- minimum detectable effect;
- allocation ratio;
- expected attrition and noncompliance;
- clustering or repeated-measure structure.

Do not choose the minimum detectable effect merely to fit the sample you happen to have. Start with the smallest effect worth acting on, then determine whether the required sample and experiment duration are feasible.

Avoid stopping an ordinary fixed-horizon test whenever $p < 0.05$. Repeated peeking increases the false-positive rate. Use a pre-specified horizon or a valid sequential design.

## Multiple testing and subgroup analysis

At a 5% significance level, testing 20 independent null hypotheses produces at least one false positive about 64% of the time:

$$
1 - (1 - 0.05)^{20} \approx 0.64
$$

Use one of these strategies:

- declare one primary outcome and a small set of planned secondary outcomes;
- control the family-wise error rate with Holm's procedure when any false positive is costly;
- control the false discovery rate with Benjamini–Hochberg in broader discovery work;
- treat unplanned subgroup findings as exploratory and confirm them in new data.

To claim that an effect differs between segments, test the **interaction**. “Significant in segment A but not significant in segment B” does not itself prove the segment effects differ.

## Worked example: did onboarding improve activation?

Assume an A/B test produced:

| Group | Activated | Total users | Activation rate |
| --- | ---: | ---: | ---: |
| Control | 1,200 | 10,000 | 12.0% |
| Treatment | 1,320 | 10,000 | 13.2% |

The estimated absolute effect is **1.2 percentage points**. Relative lift is **10%**, but the absolute effect is the better anchor: about 120 additional activations per 10,000 eligible users.

```python
import numpy as np
from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep

successes = np.array([1320, 1200])
totals = np.array([10_000, 10_000])

z_stat, p_value = proportions_ztest(successes, totals)

# Treatment minus control, using a score-based confidence interval.
ci_low, ci_high = confint_proportions_2indep(
    count1=1320,
    nobs1=10_000,
    count2=1200,
    nobs2=10_000,
    method="score",
    compare="diff",
)

absolute_effect = successes[0] / totals[0] - successes[1] / totals[1]

print(f"Absolute effect: {absolute_effect:.3%}")
print(f"95% CI: [{ci_low:.3%}, {ci_high:.3%}]")
print(f"z = {z_stat:.2f}, p = {p_value:.4f}")
```

A strong conclusion would read like this:

> The redesigned onboarding increased seven-day activation from 12.0% to 13.2%, an absolute increase of 1.2 percentage points (95% CI reported from the score method). This corresponds to roughly 120 additional activated users per 10,000 eligible users. The experiment was randomized at the user level and analyzed on an intention-to-treat basis. If the interval remains above the pre-defined practical threshold and guardrail metrics are acceptable, we recommend rollout.

Notice the order: **effect, uncertainty, scale, design, decision**. The p-value is supporting evidence, not the headline.

## A reusable analysis workflow

Use the following sequence on real work.

### 1. Frame

- Write the population, intervention or exposure, comparison, outcome, and time window.
- Name the experimental and analysis units.
- Define the estimand and smallest effect worth acting on.

### 2. Design

- Determine whether the study is randomized or observational.
- Identify pairing, repeated measures, clusters, censoring, and exposure time.
- Pre-specify exclusions, primary metrics, segments, and stopping rules.

### 3. Inspect

- Validate row counts, uniqueness, ranges, missingness, and group balance.
- Plot the outcome by group and inspect temporal patterns.
- Investigate data quality issues before inferential modeling.

### 4. Select

- Choose a test whose estimand and assumptions match the question.
- Prefer robust defaults such as Welch's t-test when equal variance is doubtful.
- Use regression or specialized models when covariates or complex structure matter.

### 5. Estimate

- Calculate the effect on an interpretable scale.
- Report a confidence interval.
- Run the planned test and correction for multiple comparisons.

### 6. Stress-test

- Check sensitivity to influential observations and defensible alternative specifications.
- Examine missing-data and sample-ratio issues.
- Confirm that conclusions are not driven by post-hoc filtering.

### 7. Communicate

- Lead with the decision-relevant effect and uncertainty.
- State limitations and what the design supports: association or causation.
- Recommend an action, further data collection, or no change.

## Common mistakes I expect you to catch in review

| Mistake | Why it fails | Better approach |
| --- | --- | --- |
| Selecting a test from data type alone | Ignores pairing, clustering, assignment, and estimand | Start with design and unit of analysis |
| Testing events as independent users | Understates uncertainty | Aggregate or use clustered methods |
| Using a normality p-value as a gatekeeper | Overreacts at large $n$ and lacks power at small $n$ | Combine plots, robustness, and domain context |
| Calling Mann–Whitney a test of medians | Its general target is rank/distributional difference | State the estimand and assumptions precisely |
| Reporting only $p < 0.05$ | Hides magnitude and uncertainty | Report effect size and confidence interval |
| Running many tests and choosing the winner | Inflates false positives | Pre-specify and correct multiplicity |
| Claiming causation from correlation | Confounding and reverse causality remain | Use a causal design or qualify the claim |
| Comparing “significant” with “not significant” | Does not test whether effects differ | Test an interaction directly |
| Peeking and stopping early | Inflates false positives in fixed-horizon tests | Pre-specify duration or use sequential methods |
| Treating non-significance as proof of no effect | The test may be underpowered | Interpret the confidence interval and equivalence margin |

## When the decision tree is no longer enough

Move beyond basic tests when you have:

- covariates requiring precision adjustment or confounding control;
- hierarchical data such as users within stores;
- repeated observations over time;
- count outcomes with different exposure durations;
- censored time-to-event outcomes;
- noncompliance, interference, or attrition;
- many correlated metrics or adaptive experimentation;
- a need to establish equivalence or non-inferiority.

At that point, the right question is usually not “Which test?” but “Which statistical model represents this design and estimand?”

## Final checklist for a new team member

Before sending an analysis for review, confirm:

- [ ] The business decision and primary question are explicit.
- [ ] The population, unit of analysis, outcome, and estimand are defined.
- [ ] Independence, pairing, clustering, and repeated measures are handled.
- [ ] Data quality and missingness checks are documented.
- [ ] The chosen test matches the design and target quantity.
- [ ] Assumptions were evaluated with plots and context, not one mechanical test.
- [ ] Effect size and confidence interval are reported.
- [ ] Multiple testing and stopping rules are addressed.
- [ ] Causal language matches the study design.
- [ ] The conclusion states what should happen next.

## Closing perspective

Learning the names of statistical tests is useful. Learning to identify the question each test actually answers is what makes someone a reliable data scientist.

Use the decision chart to form an initial hypothesis about the method. Then slow down: reconstruct the data-generating process, protect the unit of analysis, define the estimand, quantify uncertainty, and connect the result to a decision. That discipline matters far more than memorizing a larger catalog of tests.
