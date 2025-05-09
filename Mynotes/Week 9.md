[Week9Notes](file:///Users/malangtian/Documents/StudyMaterials/Courses/stat992_s25/notes/week9.pdf)

> [Distill-and-Compare: Auditing Black-Box Models Using Transparent Model Distillation](zotero://select/items/@tanDistillandCompareAuditingBlackBox2018)
> 2018; Association for Computing Machinery; Sarah Tan, Rich Caruana, Giles Hooker, Yin Lou

- **Distillation:** Treat the black-box model as a teacher and train a transparent mimic model (student) to reproduce its risk scores.
- **Comparison:** Train an outcome model using the same model class and features to predict the ground-truth outcome.
- **Compare:** Identify where these two models behave differently to reveal potential biases or flaws in the black-box model.
![[attachments/Pasted image 20250402171708.png]]

> [InterPLM: Discovering Interpretable Features in Protein Language Models via Sparse Autoencoders](zotero://select/items/@simonInterPLMDiscoveringInterpretable2025)
> 2025; Elana Simon, James Zou

![[attachments/Pasted image 20250402190324.png]]
- Train a sparse autoencoder in the embeddng space to get higher-dimensional sparse embeddings.
- There are only few dimensions are activated in the high-dimensional embeddings, which makes it easier to be connected with certain biology features.
- The model may "discover" some new biology patterns.


> [Adaptive wavelet distillation from neural networks through interpretations](zotero://select/items/@haAdaptiveWaveletDistillation)
> Wooseok Ha, Chandan Singh, Francois Lanusse, Srigokul Upadhyayula, Bin Yu

![[attachments/Pasted image 20250402212952.png]]
Afttribute to the wavelet transformed data

---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
Cell In[31], line 7
      3 exmaple_genres = ["Comedy", "Horror", "Thriller", "Drama"]
      5 reduced_CAV = RecommenderCAV(processor, no_feature_model)
----> 7 reduced_CAV.train_cav("Comedy")
      9 example_cavs = {}
     10 for genre in exmaple_genres:

File ~/Documents/StudyMaterials/Courses/stat992_s25/hw2/RecommenderCAV.py:36, in RecommenderCAV.train_cav(self, concept_genre)
     33 y_cav = labels[valid_idx]
     35 clf = LogisticRegression()
---> 36 clf.fit(X_cav, y_cav)
     37 self.cav = clf.coef_.flatten()
     38 self.concept_genre = concept_genre

File /opt/miniconda3/envs/rec_env/lib/python3.10/site-packages/sklearn/base.py:1389, in _fit_context.<locals>.decorator.<locals>.wrapper(estimator, *args, **kwargs)
   1382     estimator._validate_params()
   1384 with config_context(
   1385     skip_parameter_validation=(
   1386         prefer_skip_nested_validation or global_skip_validation
   1387     )
   1388 ):
-> 1389     return fit_method(estimator, *args, **kwargs)

File /opt/miniconda3/envs/rec_env/lib/python3.10/site-packages/sklearn/linear_model/_logistic.py:1301, in LogisticRegression.fit(self, X, y, sample_weight)
   1299 classes_ = self.classes_
   1300 if n_classes < 2:
-> 1301     raise ValueError(
   1302         "This solver needs samples of at least 2 classes"
   1303         " in the data, but the data contains only one"
   1304         " class: %r" % classes_[0]
   1305     )
   1307 if len(self.classes_) == 2:
   1308     n_classes = 1

ValueError: This solver needs samples of at least 2 classes in the data, but the data contains only one class: 0
  