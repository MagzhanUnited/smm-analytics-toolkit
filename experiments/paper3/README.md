# Experiments of the third paper

Script behind the paper "Can Engagement with a Brand Post Be Predicted Before It Is Published? A Time-Aware
Evaluation on Open Facebook Data".

- **Data.** UCI "Facebook metrics" (S. Moro, P. Rita, B. Vala, *J. Bus. Res.* 69(9), 2016), licence CC BY 4.0:
  500 posts of 2014 from the Facebook page of a cosmetics brand. Download the archive from the UCI repository
  (dataset 368) and put `dataset_Facebook.csv` in `~/Desktop/projects/aitu/smm-ml/data/uci_facebook_metrics/`
  (or change `DATA` in the script). The data are not stored in this repository.
- **Packages.** Python 3.12 with numpy, pandas, scipy and scikit-learn; versions of the reported run are in the
  result file (`environment`).
- **Run.** `python -I -B experiments/paper3/engagement.py` (about 11 minutes on a laptop processor) writes
  `results/engagement.json`.

What the script does: it predicts the interactions, engaged users and reach of a post (log scale) from the
attributes known before publication, with three median baselines (all earlier posts, by post type, by post type
over the last three months) and three trained models (ridge regression, random forest, gradient boosting), under
three evaluations (forward in time month by month from July to December, one month left out, repeated random
10-fold cross-validation), with bootstrap intervals over test posts and over test months. It also measures
leakage (reach and impressions among the inputs), the contribution of each attribute, and the association of
paid promotion with the three metrics.
