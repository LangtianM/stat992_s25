# LIME and SHAP
Generating local explainable models.

["Why Should I Trust You?": Explaining the Predictions of Any Classifier](zotero://select/items/@ribeiroWhyShouldTrust2016)
2016; Association for Computing Machinery; Marco Tulio Ribeiro, Sameer Singh, Carlos Guestrin
**LIME:** Fit a local interpretable model $g$ around a certain observation.
- Smaple $x_{n}' \sim \pi_{x}$.
- Summarize $x_{n}' \to z_{n}'$.
- Solve
$$
\xi=\arg \min _{g \in G} L\left(f(z_{n};), g(z_{n}'), \pi_x\right)+\Omega(g).
$$
Example
- $\pi_{x}$ select sentences with high similarity.
- $x_{n}\to z_{n}$ transform to word counts.
- $g$: linear model.
- $\Omega$: $l^{1}$ regularization.
$$
\frac{1}{N}\sum_{i=1}^{n}(f(z_{n}) - g(z_{n}))^{2} + \lVert g \rVert _{1}
$$

**SP-LIME**: Explain representative observations to understand the whole model.
Submodular optimization: find representative observations.

[Algorithms to estimate Shapley value feature attributions](zotero://select/items/@chenAlgorithmsEstimateShapley2022)
2022; Hugh Chen, Ian C. Covert, Scott M. Lundberg, Su-In Lee.
**SHAP**: Credit asisgnment: takeout one employee to see how the profit decreases.
 - Employee $\to$ feature.
 - Team $\to$ subset of features.
 - Profit $\to$ Expected prediction.
$$
v_{x}(s) = \mathbb{E}_{p(x_{s^{c}}|x_{s})}[f(x_{s'}, x_{s^{c}})]
$$
Fix $x_{s}$, as current sample's values, average over all other features.
$\left.\phi_i(v)=\frac{1}{|D|!} \sum_{\pi \subseteq \Pi(D)}\left(v\left(\operatorname{Pre}^i \pi\right) \cup\{i\}\right)-v\left(\operatorname{Pr}^i(\pi)\right)\right)$.
> Sampling from conditional distribution could be very expensive

[L-Shapley and C-Shapley: Efficient Model Interpretation for Structured Data](zotero://select/items/@chenLShapleyCShapleyEfficient2018)
2018; Jianbo Chen, Le Song, Martin J. Wainwright, Michael I. Jordan
**Local Shapley:** only perturbs the neighboring features of a given feature
**Connected Shapley:** Only take acoount to "connected" features. (not only nearby)


[An unexpected unity among methods for interpreting model predictions](zotero://select/items/@lundbergUnexpectedUnityMethods2016)
2016; Scott Lundberg, Su-In Lee.
-  $x \in\mathbb{R}^{p}$: Original inputs.
 - $x'=h_{x}(x)$: transfored inputs, a binary vector of length $M$ representing whether a value is known or missing. 
 - $g(x')$ : simple approximation to the original model for an individual prediction.
Find an interpretable local model $\xi$ that minimizes the following objective function:
$$
\xi=\arg \min _{g\in\mathcal{G}} L(f, g, \pi_{x'}) + \Omega(g),
$$
where $\pi_{x'}$ is a sample weighting kernel, $\Omega$ penalizes the complexity of $g$. If $g$ is assumed to be linear. Then the loss function $L$, the sample weighting kernel $\pi_{x'}$, and the regularization term $\Omega$ are all uniquely determined under some assumptions from game theory.
- Efficiency: the model to correctly capture the original predicted value.
- Symmetry: if two features contribute equally to the model, then their effects must be the same.
- Monotonicity: if observing a feature increases $f$ more than $f^{\prime}$ in all situations, then that feature's effect should be larger for $f$ than for $f^{\prime}$.




