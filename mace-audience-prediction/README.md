# Predicting Musical Preferences of a Live Concert Audience

**Team:** Michele Angiolillo, Lucia Leone, Matteo Palumbo, Chiara Pastorelli, Alessandro Querqui – MSc in Data Science, Sapienza University of Rome

**Question:** can the musical preferences of a heterogeneous concert audience be predicted from socio-demographic and behavioural variables collected in real time, and how much does feature engineering improve a classifier on this kind of noisy, self-reported data?

## What makes this project different: the data is ours

Instead of using a ready-made dataset, we collected the data ourselves. We attended the MACE concert in Milan (30 April 2026) and ran an on-site digital survey: spectators scanned a QR code and filled in a questionnaire before the show, covering socio-demographic traits, listening habits and, through 1–5 Likert scales, what they wanted from the night (energy/moshing, a psychedelic journey, emotion/singing). The final question – which guest artist they hoped to see – is the target.

This gives a primary dataset of **319 real respondents**, with all the messiness that comes with it. The target is the artist mapped onto **four musical macro-genres**: Classic Rap, Trap/New Scene, Indie/Pop, Alternative/Electronic.

## Approach

- **EDA:** demographic, geographic and behavioural profiling of the audience (strong 19–21 skew, female majority, concentration in Northern/Central Italy).
- **Feature engineering:** grouped free-text and high-cardinality answers into macro-categories (regions into macro-areas, artist descriptions into four adjectives + "Other").
- **Class balancing:** compared SMOTE against class weighting; SMOTE won on all three models. Applied **only inside the training folds** to avoid data leakage.
- **Models:** Logistic Regression, Gradient Boosting, Random Forest, evaluated with **stratified 5-fold cross-validation** (not just a single split, which had proved over-optimistic).
- **Hyperparameter tuning:** GridSearchCV inside a SMOTE + StandardScaler pipeline.

## Key result: less is more

The most interesting finding came from the feature importance analysis. Musical-genre variables alone carried about 49% of the predictive weight, while many response categories contributed almost nothing. We ran a **pooling experiment**: every category with importance below 1.5% was collapsed into a single "Other" class, shrinking the feature space from 36 to 20 columns.

Simplifying the model **improved** its cross-validated accuracy from **38.3% to 45.1%**, and made it more interpretable at the same time. Reducing noise beat keeping every variable.

## Honest reading of the numbers

Final weighted F1 is about 0.45, and accuracy around 45% on four classes. That is modest in absolute terms, but the baseline for four balanced classes is 25%, so the model is well above chance on a genuinely hard problem: predicting one of four musical tastes from a short self-reported survey. The per-class breakdown is informative in itself: Indie/Pop is predicted best (F1 0.60), while Alternative/Electronic (only 6 test samples) is the weakest, and the report's conclusion is that the priority is not a fancier model but **collecting more data** for the underrepresented classes.

The value of this project is less in the final accuracy than in the end-to-end process on self-collected data: designing the survey, handling imbalance without leakage, validating honestly, and showing that principled feature reduction improves both performance and interpretability.

## Files

- [`project_script.ipynb`](project_script.ipynb) – full analysis (Google Colab)
- [`report.pdf`](report.pdf) – detailed project report

## Tools

Python, pandas, NumPy, scikit-learn, imbalanced-learn (SMOTE), Matplotlib, Seaborn
