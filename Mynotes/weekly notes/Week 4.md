[Examples are not enough, learn to criticize! Criticism for Interpretability](zotero://select/items/@kimExamplesAreNot2016)
2016; Curran Associates, Inc.; Been Kim, Rajiv Khanna, Oluwasanmi O Koyejo
**Motivation**
- Example based explanations are not enough to capture complex data distributions
- Both Prototypes and Criticism (examples that highlight what is missing in the prototypes) should be used for model interpretability 
**Methods**
- **MMD-Critic:** Maximum Mean Discrepancy for prototype selection and criticism$$
\operatorname{MMD}(\mathcal{F}, P, Q)=\sup _{f \in \mathcal{F}}\left(\mathrm{E}_{X \sim P}[f(X)]-\mathrm{E}_{Y \sim Q}[f(Y)]\right).$$Let $\mathrm{X}$ be samples and $\mathrm{X_{S}}$ be a subset of samples. Select prototype indices $\mathrm{S}$ which minimize $\text{MMD}^{2}(\mathcal{F}, \mathrm{X}, \mathrm{X_{S}})$. 
- An efficient greedy method for the optimization problem is proposed.
- **Model Criticism:** Find data points not well explained by the prototypes by maximizing$L(\mathrm{C})=\sum_{l \in \mathrm{C}}\left|\frac{1}{n} \sum_{i \in[n]} k\left(x_i, x_l\right)-\frac{1}{m} \sum_{j \in \mathrm{~S}} k\left(x_j, x_l\right)\right|$, where $k$ is the kernel function.
- **Computationally hard**: always need a $n\times n$ matrix to store the kernels
 
[Counterfactual Explanations without Opening the Black Box: Automated Decisions and the GDPR](zotero://select/items/@wachterCounterfactualExplanationsOpening2018)
Find counterfactual $x'$:
$$\arg \min _{x^{\prime}} \max _\lambda \lambda\left(f_w\left(x^{\prime}\right)-y^{\prime}\right)^2+d\left(x_i, x^{\prime}\right)$$

[Generating Visual Explanations](zotero://select/items/@hendricksGeneratingVisualExplanations2016)
2016; Lisa Anne Hendricks, Zeynep Akata, Marcus Rohrbach, Jeff Donahue, Bernt Schiele, Trevor Darrell
Generate accompany text to explain why the model made such prediction.


