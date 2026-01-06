PDP Method:
$$
\text{PDP}_{d}(z) = \mathbb{E}[f(X_{d}, X_{-d})|X_{d}=z]
$$

> [Visualizing the Effects of Predictor Variables in Black Box Supervised Learning Models](zotero://select/items/@apleyVisualizingEffectsPredictor2020)
> 2020; Daniel W. Apley, Jingyu Zhu

ALE:
$$
f_{j, A L E}\left(x_j\right)=\int_{x_{\min j}}^{x_j} E\left[\left.\frac{\partial f\left(X_j, X_{-j}\right)}{\partial X_j} \right\rvert\, X_j=z_j\right] d z_j
$$


[Peeking Inside the Black Box: Visualizing Statistical Learning With Plots of Individual Conditional Expectation](zotero://select/items/@goldsteinPeekingBlackBox2015)
2015; ASA Website; Alex Goldstein, Adam Kapelner, Justin Bleich, Emil Pitkin
Focus on individual conditional expectation:
$$
f_{I C E}^{(i)}\left(X_j\right)=f\left(X_j, X_{-j}^{(i)}\right)
$$
Consider the exmaple model:
$$
\begin{aligned}
Y= & 0.2 X_1-5 X_2+10 X_2 \mathbb{1}_{X_3 \geq 0}+\mathcal{E}, \\
& \mathcal{E} \stackrel{\mathrm{iid}}{\sim} \mathcal{N}(0,1), \quad X_1, X_2, X_3 \stackrel{\text { iid }}{\sim} \mathrm{U}(-1,1) .
\end{aligned}
$$
![[attachments/Pasted image 20250227142036.png]]

[Visualizing Fit and Lack of Fit in Complex Regression Models with Predictor Effect Plots and Partial Residuals](zotero://select/items/@foxVisualizingFitLack2018)
2018; John Fox, Sanford Weisberg
