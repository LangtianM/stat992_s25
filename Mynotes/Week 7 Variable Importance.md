

[Efficient nonparametric statistical inference on population feature importance using Shapley values](zotero://select/items/@williamsonEfficientNonparametricStatistical2020)
2020; PMLR; Brian Williamson, Jean Feng

Define **predictiveness** as $V:\mathcal{F} \times \mathcal{M}\to \mathbb{R}$ where $\mathcal{F}$ is the model class (functions from $\mathcal{X}$ to $\mathcal{Y}$), $\mathcal{M}$ is the data distribution class. Examples for predictiveness include  $R^{2}$, AUC.

For the underlying data distribution $P_{0}$, predictive function class $\mathcal{F}$, and predictiveness measure $V$, define
$$
f_{0, s} = \arg \max_{f\in\mathcal{F_{s}}}V(f, P_{0}),
$$
where $\mathcal{F}_{s}$ is the function class that only use feature subset $s\subseteq \{1, 2, \dots, p \}$.
Then the **SPVIM** for variable $X_{j}$ is defined as
$$
\psi_{0,0, j}=\frac{1}{p} \sum_{s \subseteq N \backslash\{j\}} \frac{1}{\binom{p-1}{|s|}}\left[V\left(f_{0, s \cup\{j\}}, P_0\right)-V\left(f_{0, s}, P_0\right)\right]
$$
(Shapley value with $V$ being the metric)
**Remark:** The feature importance is not correlated with the model. It's inherent in data and its distribution.
**Remark:** Really hard to compute, need using subsample to estimate

Advantages of this method??
- Population level importance ? (But we still need to choose a function class to compute)
- Avaliability to do statistical inference? (Isn't it because it need to be estimated?)
---

[Lazy Estimation of Variable Importance for Large Neural Networks](zotero://select/items/@gaoLazyEstimationVariable2022)
2022; PMLR; Yue Gao, Abby Stevens, Garvesh Raskutti, Rebecca Willett
**Motivation:** retrain is too expensive for feature attribution
**Idea:** Using Linearization to approximate a new model and avoid retaining the entire model.
$$
\Delta \theta_j=\arg \min _\omega \frac{1}{n} \sum_{i=1}^n\left[Y_i-h_{\theta_f}\left(X_i^{(j)}\right)-\left.\omega^{\top} \nabla_\theta h_\theta\left(X_i^{(j)}\right)\right|_{\theta=\theta_f}\right]^2+\lambda\|\omega\|_2^2
$$
where
- $h_\theta$ ：the neural network function
- $X_i^{(j)}$ ：the sample with $j$th feature substituted by a fixed reference value
- $\lambda$ : regularization parameter
Then the new model is:
$$
h_{\theta_{f}+\Delta \theta_{j}}(X^{(j)})
$$
the importance of the variable is defined as the differences in predictiveness:
$$
\hat{V} I_j^{(L A Z Y)}=V\left(h_{\theta_f}, P_n\right)-V\left(h_{\theta_j+\Delta \theta_j}, P_{n_{,-j}}\right)
$$
Remark: you don't need to retrain the entire model, only need to figure out a the changes maed by moving a step to the retrained model.
![[attachments/Pasted image 20250306154433.png]]

---

[MDI+: A Flexible Random Forest-Based Feature Importance Framework](zotero://select/items/@agarwalMDIFlexibleRandom2023)
2023; Abhineet Agarwal, Ana M. Kenney, Yan Shuo Tan, Tiffany M. Tang, Bin Yu

