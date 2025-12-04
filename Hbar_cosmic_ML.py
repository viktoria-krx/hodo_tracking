%load_ext autoreload
%autoreload 2

from read_file import *
from find_tracks import *
from fit_tracks import *
from fit_vertex import *
from draw_tracks import *
from get_event_features import *
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.cluster import DBSCAN, HDBSCAN
import cProfile, pstats
import time
from itertools import combinations
from mpl_toolkits.axes_grid1 import make_axes_locatable


from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve
import xgboost as xgb

# plt.rcParams.update({"figure.dpi": 72})

import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", message="R\^2 score is not well-defined")
plt.style.use("asacusa.mplstyle")

plot=True

root_path = "~/Documents/Hodoscope/cern_data/2025_Data/output_000636.root" # cosmics
root_path = "~/Documents/Hodoscope/cern_data/2025_Data/output_001825_6567.root" # cosmics

BGdf = build_hits_df_fast(root_path)
BGdf["Hbar_BG"] = "BG"
# hits_df["z_used"] = np.where(np.isnan(hits_df["z_reco"]), hits_df["z"], hits_df["z_reco"])

max_ev = BGdf["event"].max()

run_list = np.concatenate([np.arange(1705, 1721), np.arange(1724, 1730)])
Hbardf = build_hits_df_from_runs(run_list, version="cusp_run")
Hbardf["Hbar_BG"] = ["Hbar" if m == True else "BG" for m in Hbardf["mixGate"]]
Hbardf.loc[:, "event"] = Hbardf["event"] + max_ev

# hits_df = pd.concat([BGdf, Hbardf], ignore_index=True)
hits_df = pd.concat([BGdf[BGdf.event <= 5000], Hbardf], ignore_index=True)
# hits_df = Hbardf

hits_df["z_used"] = np.where(np.isnan(hits_df["z_reco"]), hits_df["z"], hits_df["z_reco"])
hits_df["dz_used"] = np.where(np.isnan(hits_df["dz_reco"]), hits_df["dz"], hits_df["dz_reco"])
hits_df["bgoToT"] = hits_df[(hits_df["detector"] == "bgo") & hits_df["LE"].notna() & hits_df["TE"].notna()& (hits_df["LE"] < hits_df["TE"])]["TE"] - hits_df[(hits_df["detector"] == "bgo") & hits_df["LE"].notna() & hits_df["TE"].notna() & (hits_df["LE"] < hits_df["TE"])]["LE"]


clustered_list = []
for event_id, ev in hits_df[hits_df.LE < 0].groupby("event"):
    ev["det_key"] = ev["detector"] + "_" + ev["channel"].astype(str)
    # select only hodo/tile hits for clustering
    ev_ht = ev[ev.detector.isin(["hodoO","hodoI","tileO","tileI"])].copy()
    if ev_ht.empty:
        ev["track_id"] = -1
        clustered_list.append(ev)
        continue

    labels = cluster_by_phi_uncertainty(ev_ht, base_eps=1, min_samples=2, theta_weight=0.2, coords="cylindrical")
    # labels = cluster_by_phi_hdbscan(ev_ht, min_samples=2, theta_weight=0, coords="cylindrical")
    ev_ht["track_id"] = labels

    # merge labels back into the full event (bgo keep -1)
    ev = ev.merge(ev_ht[["det_key","track_id"]], on="det_key", how="left")
    ev["track_id"] = ev["track_id"].fillna(-1).astype(int)

    clustered_list.append(ev)

clustered_hits = pd.concat(clustered_list, ignore_index=True)


lines_df = fit_lines_from_clusters_svd(clustered_hits, include_bgo=False, 
                                       use_xyz_errors=True, xyz_error_cols=["dx", "dy", "dz_used"], prefilter_ransac=False, ransac_thresh=15.0, weighted=False, weight_col="dz_used")

vertices_df = reconstruct_vertex_from_midpoints(clustered_hits, lines_df,
                                            bgo_radius=45.0, 
                                            max_dist_to_bgo=25.0)


# vertices_df = find_vertices_from_tracks(lines_df, eps=5.0)
lines_df = lines_df.merge(vertices_df, on="event", how="left")

clustered_hits = clustered_hits.merge(lines_df, on=["event", "track_id"], how="left")

plot_events(clustered_hits, lines_df, clustered_hits[clustered_hits.mixGate].event.unique()[:30])

event_features_df = compute_event_features_from_clustered_hits(
    clustered_hits,
    bgo_center=(0, 0, 0)
)

run_feats = set(["event", "cusp", "time", "mix", "Hbar"])
all_feats = [ele for ele in event_features_df.columns if ele not in run_feats]
ML_feats = [ele for ele in event_features_df.columns if ele not in set(["vertex", "trigger"])]

for feat in all_feats:
    plt.hist([event_features_df[event_features_df.Hbar][feat].values, event_features_df[~event_features_df.Hbar][feat].values], bins=100, stacked=True, label=["Mixing", "BG"])
    plt.legend()
    plt.xlabel(feat)
    plt.show()


# plot_events(clustered_hits, lines_df, event_features_df[event_features_df.n_tracks > 8].event)

border_event = BGdf.event.max()

bool_cols = ["Hbar", "vertex", "trigger"]
df_bool = event_features_df[bool_cols].astype(bool)

# Count all combinations
combo_counts = df_bool.value_counts().sort_index()

# Plot
combo_counts.plot(kind='bar', figsize=(10,6))
plt.xticks(rotation=45, ha='right')
plt.ylabel("Number of events")
plt.title("All combinations of main boolean features")
plt.show()





print(f"{event_features_df[event_features_df.Hbar].shape[0]} Hbar candidates vs {event_features_df[~event_features_df.Hbar].shape[0]} background events")
print(f"{event_features_df[event_features_df.vertex].shape[0]} events with a vertex")
print(f"{event_features_df[event_features_df.trigger].shape[0]} events where data is complete")
print(f"{event_features_df[event_features_df.vertex & event_features_df.trigger].shape[0]} events with a vertex and complete data")
print(f"{event_features_df[event_features_df.Hbar & event_features_df.trigger].shape[0]} Hbar candidate events with complete data")
print(f"{event_features_df[event_features_df.vertex & event_features_df.trigger & event_features_df.Hbar].shape[0]} Hbar candidate events with a vertex and complete data")





plt.hist(event_features_df[event_features_df.Hbar].bgoEdep, 100, range=(0, 150), histtype="step", label="Hbar Candidates", density=True)
plt.hist(event_features_df[~event_features_df.Hbar].bgoEdep, 100, range=(0, 150), histtype="step", label="Background", density=True)
plt.xlabel("Edep [MeV]")
plt.legend()
plt.yscale("log")
plt.show()

plt.hist(event_features_df[event_features_df.Hbar].bgoToTSum, 100, range=(0, 10000), histtype="step", label="Hbar Candidates", density=True)
plt.hist(event_features_df[~event_features_df.Hbar].bgoToTSum, 100, range=(0, 10000), histtype="step", label="Background", density=True)
plt.xlabel("Edep [ns]")
plt.legend()
plt.yscale("log")
plt.show()


y_bins = np.logspace(np.log10(10), np.log10(10000), 40)  # from 1 ns to 8000 ns
x_bins = np.linspace(0, 180, 18)
plt.hist2d(event_features_df[event_features_df.Hbar].mean_angle, event_features_df[event_features_df.Hbar].bgoToTSum, bins=(x_bins, y_bins))
plt.xlabel("Mean inter-track angle [°]")
plt.ylabel("BGO ToT sum [ns]")
plt.yscale("log")
plt.colorbar()
plt.title("Hbar Candidates")
# plt.colorbar(label="Min inter-track angle [°]")
plt.show()


y_bins = np.logspace(np.log10(10), np.log10(10000), 40)  # from 1 ns to 8000 ns
x_bins = np.linspace(0, 180, 18)
plt.hist2d(event_features_df[~event_features_df.Hbar].mean_angle, event_features_df[~event_features_df.Hbar].bgoToTSum, bins=(x_bins, y_bins))
plt.xlabel("Mean inter-track angle [°]")
plt.ylabel("BGO ToT sum [ns]")
plt.yscale("log")
plt.colorbar()
plt.title("Background")
# plt.colorbar(label="Min inter-track angle [°]")
plt.show()


y_bins = np.logspace(np.log10(10), np.log10(10000), 40)  # from 1 ns to 8000 ns
x_bins = np.linspace(0, 180, 18)
plt.hist2d(event_features_df.mean_angle, event_features_df.bgoToTSum, bins=(x_bins, y_bins))
plt.xlabel("Mean inter-track angle [°]")
plt.ylabel("BGO ToT sum [ns]")
plt.yscale("log")
plt.colorbar()
plt.title("Everything")
# plt.colorbar(label="Min inter-track angle [°]")
plt.show()


y_bins = np.logspace(np.log10(5), np.log10(200), 40)
plt.hist2d(event_features_df[event_features_df.Hbar].mean_angle, event_features_df[event_features_df.Hbar].bgoEdep, bins=(x_bins, y_bins))
plt.xlabel("Mean inter-track angle [°]")
plt.ylabel("BGO E dep [MeV]")
plt.yscale("log")
plt.colorbar()
plt.title("Hbar Candidates")
# plt.colorbar(label="Min inter-track angle [°]")
plt.show()

y_bins = np.logspace(np.log10(5), np.log10(200), 40)
plt.hist2d(event_features_df[~event_features_df.Hbar].mean_angle, event_features_df[~event_features_df.Hbar].bgoEdep, bins=(x_bins, y_bins))
plt.xlabel("Mean inter-track angle [°]")
plt.ylabel("BGO E dep [MeV]")
plt.yscale("log")
plt.colorbar()
plt.title("Background")
# plt.colorbar(label="Min inter-track angle [°]")
plt.show()

y_bins = np.logspace(np.log10(5), np.log10(200), 40)
plt.hist2d(event_features_df.mean_angle, event_features_df.bgoEdep, bins=(x_bins, y_bins))
plt.xlabel("Mean inter-track angle [°]")
plt.ylabel("BGO E dep [MeV]")
plt.yscale("log")
plt.colorbar()
plt.title("Everything")
# plt.colorbar(label="Min inter-track angle [°]")
plt.show()

plt.hist([event_features_df[event_features_df.Hbar].opp_track_dt, event_features_df[~event_features_df.Hbar].opp_track_dt], bins=np.arange(10, step=0.125), stacked=True, label=["Hbar", "Background"])
plt.xlabel("Opposite track dt [ns]")
plt.legend()
plt.show()

plt.hist([event_features_df[event_features_df.Hbar].opp_track_dt, event_features_df[~event_features_df.Hbar].opp_track_dt], bins=np.arange(50), stacked=True, label=["Hbar", "Background"])
plt.xlabel("Opposite track dt [ns]")
plt.legend()
plt.show()

plt.hist2d(event_features_df.n_tracks, event_features_df.bgoEdep, bins=(np.arange(1, 20), y_bins), norm="log")
plt.xlabel("n tracks")
plt.ylabel("BGO E dep [MeV]")
plt.yscale("log")
plt.colorbar()
plt.title("Everything")
plt.show()

plt.hist2d(event_features_df[event_features_df.Hbar].n_tracks, event_features_df[event_features_df.Hbar].bgoEdep, bins=(np.arange(1, 20), y_bins), norm="log")
plt.xlabel("n tracks")
plt.ylabel("BGO E dep [MeV]")
plt.yscale("log")
plt.colorbar()
plt.title("Hbar Candidates")
plt.show()

plt.hist2d(event_features_df[~event_features_df.Hbar].n_tracks, event_features_df[~event_features_df.Hbar].bgoEdep, bins=(np.arange(1, 20), y_bins), norm="log")
plt.xlabel("n tracks")
plt.ylabel("BGO E dep [MeV]")
plt.yscale("log")
plt.colorbar()
plt.title("Background")
plt.show()

plt.hist2d(event_features_df.n_tracks, event_features_df.mean_angle, bins=(np.arange(1, 20), np.linspace(0, 180, 19)), norm="log")
plt.xlabel("n tracks")
plt.ylabel("Mean inter-track angle [°]")
# plt.yscale("log")
plt.colorbar()
plt.title("Everything")
plt.show()

plt.hist2d(event_features_df[event_features_df.Hbar].n_tracks, event_features_df[event_features_df.Hbar].mean_angle, bins=(np.arange(1, 20), np.linspace(0, 180, 19)), norm="log")
plt.xlabel("n tracks")
plt.ylabel("Mean inter-track angle [°]")
# plt.yscale("log")
plt.colorbar()
plt.title("Hbar Candidates")
plt.show()

plt.hist2d(event_features_df[~event_features_df.Hbar].n_tracks, event_features_df[~event_features_df.Hbar].mean_angle, bins=(np.arange(1, 20), np.linspace(0, 180, 19)), norm="log")
plt.xlabel("n tracks")
plt.ylabel("Mean inter-track angle [°]")
# plt.yscale("log")
plt.colorbar()
plt.title("Background")
plt.show()


#### This is where the ML starts



cols = ['n_tracks', 'mean_angle', 'min_angle', 'max_angle', 'vertex_x', 'vertex_y', 'vertex_z', 'dist_to_bgo', 'dt', 'dt_max', 'dt_min', 'dt_mean', 'bgoToTSum', 'bgoEdep']#, 'opp_track_dt']

# Most models can't handle NaN values
evs = np.array(event_features_df[event_features_df.vertex]["event"])
X = np.array(event_features_df[event_features_df.vertex][cols])
y = np.array(event_features_df[event_features_df.vertex]["Hbar"])
# y = y * 0.9

evsnan = np.array(event_features_df["event"])
Xnan = np.array(event_features_df[cols])
ynan = np.array(event_features_df["Hbar"])
# ynan = ynan * 0.9

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

Xnan_train, Xnan_test, ynan_train, ynan_test = train_test_split(Xnan, ynan, test_size=0.2, random_state=0)

scnan = StandardScaler()

Xnan_train = scnan.fit_transform(Xnan_train)
Xnan_test = scnan.transform(Xnan_test)


def evaluate_model(y_test, y_pred, y_probs, model_name):
    """Utility to compute and print common metrics."""
    cm = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, y_probs)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n=== {model_name} ===")
    print("Confusion Matrix:\n", cm)
    print(f"ROC AUC: {auc:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")

    return {
        "model": model_name,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }


def plot_importance(importances, cols, title):
    plt.figure(figsize=(8, 4))
    plt.bar(cols, importances)
    plt.xticks(rotation=90)
    plt.ylabel("Feature importance")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# === 1. Logistic Regression ===
def logistic_regression(X_train, y_train, X_test, y_test, cols, plot=False, **kwargs):
    model = LogisticRegression(random_state=0, **kwargs)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]
    if plot:
        plot_importance(abs(model.coef_[0]), cols, "Logistic Regression Coefficients")
    return evaluate_model(y_test, y_pred, y_probs, "Logistic Regression")


# === 2. PCA + Logistic Regression ===
def pca_logistic_regression(X_train, y_train, X_test, y_test, cols, n_components=3, plot=False, **kwargs):
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    model = LogisticRegression(random_state=0, **kwargs)
    model.fit(X_train_pca, y_train)
    y_pred = model.predict(X_test_pca)
    y_probs = model.predict_proba(X_test_pca)[:, 1]
    if plot:
            
        plt.figure(figsize=(8, 4))
        for i in range(n_components):
            plt.plot(cols, abs(pca.components_[i]), label=f"PC{i+1}")
        plt.legend()
        plt.xticks(rotation=90)
        plt.ylabel("Feature contribution")
        plt.title("PCA Component Loadings")
        plt.tight_layout()
        plt.show()

    return evaluate_model(y_test, y_pred, y_probs, "PCA + Logistic Regression")


# === 3. Linear Regression (for binary task) ===
def linear_regression(X_train, y_train, X_test, y_test, cols, plot=False, **kwargs):
    model = LinearRegression(**kwargs)
    model.fit(X_train, y_train)
    y_probs = model.predict(X_test)
    y_pred = (y_probs > 0.5).astype(int)
    if plot:
        plot_importance(abs(model.coef_), cols, "Linear Regression Coefficients")
    return evaluate_model(y_test, y_pred, y_probs, "Linear Regression")


# === 4. Decision Tree ===
def decision_tree(X_train, y_train, X_test, y_test, cols, plot=False, **kwargs):
    model = DecisionTreeClassifier(random_state=0, **kwargs)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]
    if plot:
        plot_importance(model.feature_importances_, cols, "Decision Tree Feature Importance")
    return evaluate_model(y_test, y_pred, y_probs, "Decision Tree")


# === 5. Random Forest ===
def random_forest(X_train, y_train, X_test, y_test, cols, plot=False, **kwargs):
    model = RandomForestClassifier(random_state=0, **kwargs)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]
    if plot:
        plot_importance(model.feature_importances_, cols, "Random Forest Feature Importance")
    return evaluate_model(y_test, y_pred, y_probs, "Random Forest")


# === 6. XGBoost ===
def xgboost_model(X_train, y_train, X_test, y_test, cols, plot=False, **kwargs):
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="auc", random_state=0, **kwargs)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]
    if plot:
        plot_importance(model.feature_importances_, cols, "XGBoost Feature Importance")
    return evaluate_model(y_test, y_pred, y_probs, "XGBoost")


models = ["Logistic Regression", "PCA + Logistic Regression", "Linear Regression", "Decision Tree", "Random Forest", "XGBoost", "XGBoost_NaN"]

results = []

results.append(logistic_regression(X_train, y_train, X_test, y_test, cols))
results.append(pca_logistic_regression(X_train, y_train, X_test, y_test, cols))
results.append(linear_regression(X_train, y_train, X_test, y_test, cols))
results.append(decision_tree(X_train, y_train, X_test, y_test, cols))
results.append(random_forest(X_train, y_train, X_test, y_test, cols))
results.append(xgboost_model(X_train, y_train, X_test, y_test, cols))
results.append(xgboost_model(Xnan_train, ynan_train, Xnan_test, ynan_test, cols))


results_df = pd.DataFrame(results)
print(results_df)

results_df.plot(
    x="model", 
    y=["AUC", "Precision", "Recall", "F1"],
    kind="bar",
    figsize=(10,6),
    title="Model Comparison pre Optimization"
)
plt.xticks(rotation=15)
plt.tight_layout()
plt.legend(loc="lower center", ncols=4)
plt.show()

#### Hyperparameter Optimization

### Logistic Regression
lr_model = LogisticRegression(random_state=0)

lr_model.fit(X_train, y_train)

y_pred = lr_model.predict(X_test)
y_probs = lr_model.predict_proba(X_test)[:, 1]

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_probs))

# param_grid = {
#     'penalty':['l1','l2','elasticnet'],
#     'C' : np.logspace(-4,4,20),
#     'solver': ['lbfgs','newton-cg','liblinear','sag','saga'],
#     'max_iter'  : [100,1000,2500,5000]
# }

param_grid = {
    'penalty':['l2'],
    'C' : [1e-12], #np.logspace(-12, -8, 5),
    'solver': ['liblinear'],
    'max_iter'  : [2]
}

grid_search_lr = GridSearchCV(
    estimator=lr_model,
    param_grid=param_grid,
    scoring='recall',
    cv=5,
    n_jobs=-1,
    verbose=0
)

grid_search_lr.fit(X_train, y_train)

print("Search params:", param_grid)
print("Best params:", grid_search_lr.best_params_)
print("Best Recall (CV):", grid_search_lr.best_score_)

best_params_lr = grid_search_lr.best_params_
best_lr = LogisticRegression(
    random_state=0,
    **best_params_lr
)
best_lr.fit(X_train, y_train)

y_pred = best_lr.predict(X_test)
y_probs = best_lr.predict_proba(X_test)[:, 1]

# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# print("Recall:", recall_score(y_test, y_pred))
# print("Precision:", precision_score(y_test, y_pred))
# print("F1:", f1_score(y_test, y_pred))
# print("AUC:", roc_auc_score(y_test, y_probs))

evaluate_model(y_test, y_pred, y_probs, "Optimized Logistic Regression")
plot_importance(abs(best_lr.coef_[0]), cols, "Optimized Logistic Regression Coefficients")

### PCA + Logistic Regression


### Linear Regression

# linr_model = LinearRegression()

# linr_model.fit(X_train, y_train)

# y_probs = linr_model.predict(X_test)
# y_pred = (y_probs > 0.5).astype(int)
# # y_pred = linr_model.predict(X_test)
# # y_probs = linr_model.predict_proba(X_test)[:, 1]

# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))    
# print("Recall:", recall_score(y_test, y_pred))
# print("Precision:", precision_score(y_test, y_pred))
# print("F1:", f1_score(y_test, y_pred))
# print("AUC:", roc_auc_score(y_test, y_probs))


# param_grid = {
#     'copy_X': [True,False], 
#     'fit_intercept': [True,False], 
#     # 'alpha': [0.1, 1.0, 10.0, 100.0],
#     'positive': [True,False]
#     }

# grid_search_linr = GridSearchCV(
#     estimator=linr_model,
#     param_grid=param_grid,
#     scoring='roc_auc',
#     cv=5,
#     n_jobs=-1,
#     verbose=2
# )

# grid_search_linr.fit(X_train, y_train)

# print("Search params:", param_grid)
# print("Best params:", grid_search_linr.best_params_)
# print("Best Recall (CV):", grid_search_linr.best_score_)

# best_params_linr = grid_search_linr.best_params_
# best_linr = LinearRegression(
#     **best_params_linr
# )
# best_linr.fit(X_train, y_train)

# y_probs = best_linr.predict(X_test)
# y_pred = (y_probs > 0.5).astype(int)
# # y_pred = best_linr.predict(X_test)
# # y_probs = best_linr.predict_proba(X_test)[:, 1]

# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# print("Recall:", recall_score(y_test, y_pred))
# print("Precision:", precision_score(y_test, y_pred))
# print("F1:", f1_score(y_test, y_pred))
# print("AUC:", roc_auc_score(y_test, y_probs))

### Decision Tree

dt_model = DecisionTreeClassifier(random_state=0)

dt_model.fit(X_train, y_train)

y_pred = dt_model.predict(X_test)
y_probs = dt_model.predict_proba(X_test)[:, 1]

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_probs))

# param_grid = {
#     'max_depth': [5, 10, 20, 30],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }

param_grid = {
    'max_depth': [10],
    'min_samples_split': [5],
    'min_samples_leaf': [3]
}


grid_search_dt = GridSearchCV(
    estimator=dt_model,
    param_grid=param_grid,
    scoring='recall',
    cv=5,
    n_jobs=-1,
    verbose=0
)

grid_search_dt.fit(X_train, y_train)

print("Search params:", param_grid)
print("Best params:", grid_search_dt.best_params_)
print("Best Recall (CV):", grid_search_dt.best_score_)


best_params_dt = grid_search_dt.best_params_
best_dt = DecisionTreeClassifier(
    random_state=0,
    **best_params_dt
)
best_dt.fit(X_train, y_train)


y_pred = best_dt.predict(X_test)
y_probs = best_dt.predict_proba(X_test)[:, 1]

evaluate_model(y_test, y_pred, y_probs, "Optimized Decision Tree")
plot_importance(abs(best_dt.feature_importances_), cols, "Optimized Decision Tree Coefficients")



### Random Forest

rf_model = RandomForestClassifier(random_state=0)

rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
y_probs = rf_model.predict_proba(X_test)[:, 1]

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_probs))



param_grid = {
    'n_estimators': [50],
    'max_depth': [14],
    'min_samples_leaf': [2],
    'bootstrap': [False],
    'min_samples_split': [5]
}

grid_search_rf = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    scoring='recall',
    cv=5,
    n_jobs=-1,
    verbose=0
)

grid_search_rf.fit(X_train, y_train)

print("Search params:", param_grid)
print("Best params:", grid_search_rf.best_params_)
print("Best Recall (CV):", grid_search_rf.best_score_)


best_params_rf = grid_search_rf.best_params_
best_rf = RandomForestClassifier(
    random_state=0,
    **best_params_rf
)
best_rf.fit(X_train, y_train)


y_pred = best_rf.predict(X_test)
y_probs = best_rf.predict_proba(X_test)[:, 1]

evaluate_model(y_test, y_pred, y_probs, "Optimized Random Forest")
plot_importance(abs(best_rf.feature_importances_), cols, "Optimized Random Forest Coefficients")



### XGBoost

xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='auc', random_state=0)

# param_grid = {
#     'n_estimators': [250, 300, 350],
#     'max_depth': [2, 3, 4],
#     'learning_rate': [0.1],
#     'subsample': [0.6, 0.7, 0.8],
#     'colsample_bytree': [0.5]
# }

param_grid = {
    'n_estimators': [300],
    'max_depth': [4],
    'learning_rate': [0.1],
    'subsample': [0.8],
    'colsample_bytree': [0.5]
}

grid_search_xgb = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='recall',
    cv=5,
    n_jobs=-1,
    verbose=0
)

grid_search_xgb.fit(X_train, y_train)

print("Search params:", param_grid)
print("Best params:", grid_search_xgb.best_params_)
print("Best Recall (CV):", grid_search_xgb.best_score_)

best_params_xgb = grid_search_xgb.best_params_
best_xgb = xgb.XGBClassifier(
    # use_label_encoder=False,
    eval_metric='auc',
    random_state=0,
    **best_params_xgb
)
best_xgb.fit(X_train, y_train)

y_pred = best_xgb.predict(X_test)
y_probs = best_xgb.predict_proba(X_test)[:, 1]

evaluate_model(y_test, y_pred, y_probs, "Optimized XGBoost")
plot_importance(abs(best_xgb.feature_importances_), cols, "Optimized XGBoost Coefficients")



### XGBoost with NaNs

xgb_nan_model = xgb.XGBClassifier(random_state=0)

# param_grid = {
#     'n_estimators': [100, 200, 250, 300],
#     'max_depth': [2, 3, 4],
#     'learning_rate': [0.1],
#     'subsample': [0.4, 0.5, 0.6],
#     'colsample_bytree': [0.5]
# }

param_grid = {
    'n_estimators': [250],
    'max_depth': [4],
    'learning_rate': [0.1],
    'subsample': [0.6],
    'colsample_bytree': [0.5]
}

grid_search_xgb_nan = GridSearchCV(
    estimator=xgb_nan_model,
    param_grid=param_grid,
    scoring='recall',
    cv=5,
    n_jobs=-1,
    verbose=0
)

grid_search_xgb_nan.fit(Xnan_train, ynan_train)

print("Search params:", param_grid)
print("Best params:", grid_search_xgb_nan.best_params_)
print("Best Recall (CV):", grid_search_xgb_nan.best_score_)

best_params_xgb_nan = grid_search_xgb_nan.best_params_
best_xgb_nan = xgb.XGBClassifier(
    # use_label_encoder=False,
    eval_metric='auc',
    random_state=0,
    **best_params_xgb_nan
)
best_xgb_nan.fit(Xnan_train, ynan_train)

y_pred = best_xgb_nan.predict(Xnan_test)
y_probs = best_xgb_nan.predict_proba(Xnan_test)[:, 1]

evaluate_model(ynan_test, y_pred, y_probs, "Optimized XGBoost with NaN")
plot_importance(abs(best_xgb_nan.feature_importances_), cols, "Optimized XGBoost with NaN Coefficients")




results_opt = []

results_opt.append(logistic_regression(X_train, y_train, X_test, y_test, cols, **best_params_lr))
results_opt.append(pca_logistic_regression(X_train, y_train, X_test, y_test, cols))
results_opt.append(linear_regression(X_train, y_train, X_test, y_test, cols))
results_opt.append(decision_tree(X_train, y_train, X_test, y_test, cols, **best_params_dt))
results_opt.append(random_forest(X_train, y_train, X_test, y_test, cols, **best_params_rf))
results_opt.append(xgboost_model(X_train, y_train, X_test, y_test, cols, **best_params_xgb))
results_opt.append(xgboost_model(Xnan_train, ynan_train, Xnan_test, ynan_test, cols, **best_params_xgb_nan))

xgboost_model(Xnan_train, ynan_train, Xnan_test, ynan_test, cols, **best_params_xgb_nan, plot=True)

results_opt_df = pd.DataFrame(results_opt)
print(results_opt_df)

results_opt_df.plot(
    x="model", 
    y=["AUC", "Precision", "Recall", "F1"],
    kind="bar",
    figsize=(10,6),
    title="Model Comparison Optimized"
)
plt.xticks(rotation=15)
plt.tight_layout()
plt.legend(loc="lower center", ncols=4)
plt.show()

def proba_model(model, X, y):
    probs = cross_val_predict(
        model,
        X,
        y,
        cv=5,
        method="predict_proba"
    )[:,1]
    return probs

probs_lr = proba_model(best_lr, X, y)
probs_dt = proba_model(best_dt, X, y)
probs_rf = proba_model(best_rf, X, y)
probs_xgb = proba_model(best_xgb, X, y)
probs_xgb_nan = proba_model(best_xgb_nan, Xnan, ynan)


evs_notna = np.array(event_features_df[event_features_df.vertex_x.notna()]["event"])
evs_notna = pd.DataFrame(evs_notna, columns=["event"])

evs_notna["prob_Hbar_LR"] = probs_lr
evs_notna["prob_Hbar_DT"] = probs_dt
evs_notna["prob_Hbar_RF"] = probs_rf
evs_notna["prob_Hbar_XGB"] = probs_xgb

evs_all = np.array(event_features_df["event"])
evs_all = pd.DataFrame(evs_all, columns=["event"])
evs_all["prob_Hbar_XGB_nan"] = probs_xgb_nan

event_features_df = pd.merge(event_features_df, evs_notna, on="event", how="left")
event_features_df = pd.merge(event_features_df, evs_all, on="event", how="left")


for mod in ["prob_Hbar_LR", "prob_Hbar_DT", "prob_Hbar_RF", "prob_Hbar_XGB", "prob_Hbar_XGB_nan"]:

    plt.hist2d(event_features_df[mod], event_features_df.Hbar, bins=(10, 2), range=((0, 1), (0,1)), norm="log")
    plt.annotate(f"prob > 0.5: {len(event_features_df[(event_features_df.Hbar == 1) & (event_features_df[mod] > 0.5)])}", xy=(0.75, 0.67), xycoords="axes fraction", fontsize=8, color="white", ha="center")
    plt.annotate(f"prob < 0.5: {len(event_features_df[(event_features_df.Hbar == 1) & (event_features_df[mod] < 0.5)])}", xy=(0.25, 0.67), xycoords="axes fraction", fontsize=8, color="white", ha="center")
    plt.annotate(f"prob > 0.5: {len(event_features_df[(event_features_df.Hbar == 0) & (event_features_df[mod] > 0.5)])}", xy=(0.75, 0.33), xycoords="axes fraction", fontsize=8, color="white", ha="center")
    plt.annotate(f"prob < 0.5: {len(event_features_df[(event_features_df.Hbar == 0) & (event_features_df[mod] < 0.5)])}", xy=(0.25, 0.33), xycoords="axes fraction", fontsize=8, color="white", ha="center")
    plt.ylabel("Hbar")
    plt.xlabel(mod)
    plt.colorbar()
    plt.show()



def probability_band_confusion_matrix(y_true, y_proba, bands=None, model_name="Model"):
    y_true = np.asarray(y_true).astype(int)

    if bands is None:
        bands = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    # Replace NaN with -1 so it falls into the first band (0–0.2)
    y_proba_fixed = np.where(np.isnan(y_proba), -1.0, y_proba)

    # Assign band index
    band_idx = np.digitize(y_proba_fixed, bands, right=False) - 1
    band_idx = np.clip(band_idx, 0, len(bands) - 2)

    # Prepare table
    band_labels = [f"{bands[i]}–{bands[i+1]}" for i in range(len(bands)-1)]
    df = pd.DataFrame(0, index=["True 0", "True 1"], columns=band_labels)

    # Count
    for t, b in zip(y_true, band_idx):
        df.loc[f"True {t}", df.columns[b]] += 1

    # Plot
    plt.figure(figsize=(10,3))
    plt.imshow(df, norm="log")
    plt.title(f"Probability-Band Confusion Matrix: {model_name}")
    plt.xticks(range(len(df.columns)), df.columns)
    plt.yticks(range(2), df.index)
    plt.colorbar(label="Count")
    plt.grid(False)
    # Annotate
    for i in range(2):
        for j in range(len(df.columns)):
            plt.text(j, i, df.iloc[i, j], ha='center', va='center')

    plt.show()
    return df

for mod in ["prob_Hbar_DT", "prob_Hbar_RF", 
            "prob_Hbar_XGB", "prob_Hbar_XGB_nan"]:

    pb_cm = probability_band_confusion_matrix(
        y_true=event_features_df.Hbar.values,
        y_proba=event_features_df[mod].values,
        model_name=mod
    )


mask_vertex = event_features_df.vertex_x.notna().values
y_true = event_features_df["Hbar"].values

def filled_predictions(df, proba_column, mask_vertex):
    """Return predictions for ALL events.
       For events where model is not applicable → predict class 0."""
    
    preds = np.zeros(len(df))  # default prediction for NaN events
    preds[mask_vertex] = df.loc[mask_vertex, proba_column].values >= 0.5
    return preds.astype(int)

pred_DT  = filled_predictions(event_features_df, "prob_Hbar_DT",  mask_vertex)
pred_RF  = filled_predictions(event_features_df, "prob_Hbar_RF",  mask_vertex)
pred_XGB = filled_predictions(event_features_df, "prob_Hbar_XGB", mask_vertex)

# XGB_nan predicts everywhere, so no filling needed:
pred_XGB_nan = (event_features_df["prob_Hbar_XGB_nan"].values >= 0.5).astype(int)


for name, preds in [
    ("DT", pred_DT),
    ("RF", pred_RF),
    ("XGB", pred_XGB),
    ("XGB_nan", pred_XGB_nan),
]:
    print(name)
    print("  Precision:", precision_score(y_true, preds))
    print("  Recall:",    recall_score(y_true, preds))
    print()

models = {
    "DT": "prob_Hbar_DT",
    "RF": "prob_Hbar_RF",
    "XGB": "prob_Hbar_XGB",
    "XGB_nan": "prob_Hbar_XGB_nan"
}


mask_vertex = event_features_df.vertex_x.notna().values
y_true = event_features_df["Hbar"].values
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
for name, col in models.items():
    proba = event_features_df[col].values.copy()
    
    # For models that cannot produce a prediction when vertex is missing
    if name != "XGB_nan":
        # Assign conservative default proba = 0 where the model is not applicable
        proba[~mask_vertex] = 0.0
    
    # Compute ROC
    fpr, tpr, _ = roc_curve(y_true, proba)
    auc_score = auc(fpr, tpr)

    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f_scores = (2 * precision * recall) / (precision + recall)


    # Find the threshold with the maximal F-score
    max_f_score_idx = np.argmax(f_scores)
    max_f_score_threshold = thresholds[max_f_score_idx]

    ax[0].plot(fpr, tpr, ".", label=f"{name} (AUC = {auc_score:.3f})")
    ax[1].plot(recall, precision,".", label=f"{name} (F-score = {f_scores[max_f_score_idx]:.3f})")
    ax[1].scatter(recall[max_f_score_idx], precision[max_f_score_idx], marker="x", color="black", zorder=5)#, label=f"{name} (F-score = {f_scores[max_f_score_idx]:.3f})")

# Plot formatting
ax[0].plot([0, 1], [0, 1], 'k--', linewidth=1)
ax[1].plot([0, 1], [0.5, 0.5], 'k--', linewidth=1)
ax[0].set_xlabel("False Positive Rate")
ax[0].set_ylabel("True Positive Rate")
ax[0].set_title("ROC Curves for All Models")
ax[0].legend(fontsize=8)
ax[1].set_xlabel("Recall")
ax[1].set_ylabel("Precision")
ax[1].set_title("Precision-Recall Curves for All Models")
ax[1].legend(fontsize=8)
plt.tight_layout()
plt.suptitle("All Runs", fontsize=16, y=0.98)
fig.subplots_adjust(top=0.85)
plt.show()

mask_vertex = event_features_df[event_features_df.event > border_event].vertex_x.notna().values
y_true = event_features_df[event_features_df.event > border_event]["Hbar"].values
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
for name, col in models.items():
    proba = event_features_df[event_features_df.event > border_event][col].values.copy()
    
    # For models that cannot produce a prediction when vertex is missing
    if name != "XGB_nan":
        # Assign conservative default proba = 0 where the model is not applicable
        proba[~mask_vertex] = 0.0
    
    # Compute ROC
    fpr, tpr, _ = roc_curve(y_true, proba)
    auc_score = auc(fpr, tpr)

    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f_scores = (2 * precision * recall) / (precision + recall)


    # Find the threshold with the maximal F-score
    max_f_score_idx = np.argmax(f_scores)
    max_f_score_threshold = thresholds[max_f_score_idx]

    ax[0].plot(fpr, tpr, ".", label=f"{name} (AUC = {auc_score:.3f})")
    ax[1].plot(recall, precision,".", label=f"{name} (F-score = {f_scores[max_f_score_idx]:.3f})")
    ax[1].scatter(recall[max_f_score_idx], precision[max_f_score_idx], marker="x", color="black", zorder=5)#, label=f"{name} (F-score = {f_scores[max_f_score_idx]:.3f})")

# Plot formatting
ax[0].plot([0, 1], [0, 1], 'k--', linewidth=1)
ax[1].plot([0, 1], [0.5, 0.5], 'k--', linewidth=1)
ax[0].set_xlabel("False Positive Rate")
ax[0].set_ylabel("True Positive Rate")
ax[0].set_title("ROC Curves for All Models")
ax[0].legend(fontsize=8)
ax[1].set_xlabel("Recall")
ax[1].set_ylabel("Precision")
ax[1].set_title("Precision-Recall Curves for All Models")
ax[1].legend(fontsize=8)
plt.tight_layout()
plt.suptitle("Hbar Runs", fontsize=16, y=0.98)
fig.subplots_adjust(top=0.85)
plt.show()














# xgb_n_est = []
# for i in np.arange(1, 250, 1):
#     xgb_n_est.append(xgboost_model(X_train, y_train, X_test, y_test, cols, n_estimators=int(i), learning_rate=0.1, max_depth=5))

# xgb_n_est_df = pd.DataFrame(xgb_n_est)
# xgb_n_est_df["sum"] = xgb_n_est_df.AUC+xgb_n_est_df.Recall+xgb_n_est_df.Precision+xgb_n_est_df.F1
# xgb_n_est_df["n_estimators"] = np.arange(1, 250, 1)
# xgb_n_est_df.plot(
#     x="n_estimators", 
#     y=["AUC", "Recall", "Precision"]#, "Precision", "Recall", "F1"],  
# )

# plt.plot(xgb_n_est_df.n_estimators, xgb_n_est_df.AUC+xgb_n_est_df.Recall+xgb_n_est_df.Precision+xgb_n_est_df.F1)
# plt.show()

# #np.where(np.max(xgb_n_est_df["sum"]) == xgb_n_est_df["sum"])[0][0]

# evsnan = np.array(event_features_df["event"])
# Xnan = np.array(event_features_df[cols])
# ynan = np.array(event_features_df["Hbar"])

# Xnan_train, Xnan_test, ynan_train, ynan_test = train_test_split(Xnan, ynan, test_size=0.2, random_state=0)

# sc = StandardScaler()

# Xnan_train = sc.fit_transform(Xnan_train)
# Xnan_test = sc.transform(Xnan_test)


# xgb_se_model = xgb.XGBClassifier(
#     use_label_encoder=False,
#     eval_metric='auc',
#     random_state=0,
#     n_estimators=50,
#     learning_rate=0.1,
#     max_depth=5,
#     missing=np.nan
# )

# xgb_model.fit(Xnan_train, ynan_train)


# ynan_pred = xgb_model.predict(Xnan_test)
# ynan_probs = xgb_model.predict_proba(Xnan_test)[:, 1]

# print("Confusion Matrix:\n", confusion_matrix(ynan_test, ynan_pred))
# print("Recall:", recall_score(ynan_test, ynan_pred))
# print("Precision:", precision_score(ynan_test, ynan_pred))
# print("F1:", f1_score(ynan_test, ynan_pred))
# print("AUC:", roc_auc_score(ynan_test, ynan_probs))

# probs = cross_val_predict(
#     xgb_model,
#     Xnan,
#     ynan,
#     cv=5,
#     method="predict_proba"
# )[:,1]

# evs = pd.DataFrame(evsnan, columns=["event"])
# evs["prob_Hbar_XGBnan"] = probs

# event_features_df = pd.merge(event_features_df, evs, on="event", how="left")






# rf_model = RandomForestClassifier(random_state=0)

# rf_model.fit(X_train, y_train)

# y_pred = rf_model.predict(X_test)
# y_probs = rf_model.predict_proba(X_test)[:, 1]

# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# print("Recall:", recall_score(y_test, y_pred))
# print("Precision:", precision_score(y_test, y_pred))
# print("F1:", f1_score(y_test, y_pred))
# print("AUC:", roc_auc_score(y_test, y_probs))



# param_grid = {
#     'n_estimators': [ 50],
#     'max_depth': [9, 10, 11, 12],
#     'min_samples_leaf': [2],
#     'bootstrap': [False],
#     'min_samples_split': [5]
#     # 'learning_rate': [0.01, 0.1, 0.2],
#     # 'subsample': [0.7, 0.9, 1.0]
#     # 'colsample_bytree': [0.5, 0.7, 0.9]
# }

# grid_search_rf = GridSearchCV(
#     estimator=rf_model,
#     param_grid=param_grid,
#     scoring='recall',
#     cv=5,
#     n_jobs=-1,
#     verbose=2
# )

# grid_search_rf.fit(X_train, y_train)


# print("Best params:", grid_search_rf.best_params_)
# print("Best Recall (CV):", grid_search_rf.best_score_)


# best_params_rf = grid_search_rf.best_params_
# best_rf = RandomForestClassifier(
#     random_state=0,
#     **best_params_rf
# )
# best_rf.fit(X_train, y_train)


# y_pred = best_rf.predict(X_test)
# y_probs = best_rf.predict_proba(X_test)[:, 1]

# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# print("Recall:", recall_score(y_test, y_pred))
# print("Precision:", precision_score(y_test, y_pred))
# print("F1:", f1_score(y_test, y_pred))
# print("AUC:", roc_auc_score(y_test, y_probs))


# rf_n_est = []
# for i in np.arange(1, 500, 20):
#     rf_n_est.append(random_forest(X_train, y_train, X_test, y_test, cols, n_estimators=int(i)))

# rf_n_est_df = pd.DataFrame(rf_n_est)
# rf_n_est_df["n_estimators"] = np.arange(1, 500, 20)
# rf_n_est_df.plot(
#     x="n_estimators", 
#     y=["AUC", "Recall", "Precision"]#, "Precision", "Recall", "F1"],  
# )


# print("Train AUC:", roc_auc_score(y_train, rf_model.predict_proba(X_train)[:,1]))
# print("Test AUC:", roc_auc_score(y_test, rf_model.predict_proba(X_test)[:,1]))



# probs = cross_val_predict(
#     rf_model,
#     X,
#     y,
#     cv=5,
#     method="predict_proba"
# )[:,1]

# evs = np.array(event_features_df[event_features_df.vertex_x.notna()]["event"])
# evs = pd.DataFrame(evs, columns=["event"])
# evs["prob_Hbar_RF"] = probs

# # event_features_df.drop(["prob_Hbar"], axis=1, inplace=True)

# event_features_df = pd.merge(event_features_df, evs, on="event", how="left")

# # plt.hist2d(event_features_df.prob_Hbar, event_features_df.Hbar, bins=(20, 2), range=((0, 1), (0, 1)))
# # plt.xlabel("Probability of Hbar")
# # plt.ylabel("Hbar")
# # plt.show()

for th in np.arange(0.1, 1, 0.1):
    y_bins = np.logspace(np.log10(5), np.log10(400), 40)
    plt.hist2d(event_features_df[event_features_df.prob_Hbar_RF > th].mean_angle, event_features_df[event_features_df.prob_Hbar_RF > th].bgoEdep, bins=(x_bins, y_bins))
    plt.xlabel("Mean inter-track angle [°]")
    plt.ylabel("BGO E dep [MeV]")
    plt.yscale("log")
    plt.colorbar()
    plt.title(f"Hbar Candidates (RF prob. > {th})")
    # plt.colorbar(label="Min inter-track angle [°]")
    plt.show()

    y_bins = np.logspace(np.log10(5), np.log10(400), 40)
    plt.hist2d(event_features_df[event_features_df.prob_Hbar_RF < th].mean_angle, event_features_df[event_features_df.prob_Hbar_RF < th].bgoEdep, bins=(x_bins, y_bins))
    plt.xlabel("Mean inter-track angle [°]")
    plt.ylabel("BGO E dep [MeV]")
    plt.yscale("log")
    plt.colorbar()
    plt.title(f"Background (RF prob. < {th})")
    # plt.colorbar(label="Min inter-track angle [°]")
    plt.show()



for th in np.arange(0.1, 1, 0.1):
    y_bins = np.logspace(np.log10(5), np.log10(400), 40)
    plt.hist2d(event_features_df[event_features_df.prob_Hbar_XGB > th].mean_angle, event_features_df[event_features_df.prob_Hbar_XGB > th].bgoEdep, bins=(x_bins, y_bins))
    plt.xlabel("Mean inter-track angle [°]")
    plt.ylabel("BGO E dep [MeV]")
    plt.yscale("log")
    plt.colorbar()
    plt.title(f"Hbar Candidates (XGB prob. > {th})")
    # plt.colorbar(label="Min inter-track angle [°]")
    plt.show()

    y_bins = np.logspace(np.log10(5), np.log10(400), 40)
    plt.hist2d(event_features_df[event_features_df.prob_Hbar_XGB < th].mean_angle, event_features_df[event_features_df.prob_Hbar_XGB < th].bgoEdep, bins=(x_bins, y_bins))
    plt.xlabel("Mean inter-track angle [°]")
    plt.ylabel("BGO E dep [MeV]")
    plt.yscale("log")
    plt.colorbar()
    plt.title(f"Background (XGB prob. < {th})")
    # plt.colorbar(label="Min inter-track angle [°]")
    plt.show()



plt.hist([event_features_df.time[(event_features_df.event > border_event) & (event_features_df.Hbar)]*1e-9, event_features_df.time[(event_features_df.event > border_event) & (~event_features_df.Hbar)]*1e-9], bins=35, stacked=True, label=["Mixing", "BG"])
plt.hist(event_features_df.time[(event_features_df.event > border_event) & (event_features_df.prob_Hbar_RF > 0.5)]*1e-9, bins=35, histtype="step", label="Hbar RF prob > 0.5", lw=2)
plt.legend()
plt.xlabel("Time in s")
# plt.yscale("log")
plt.show()

plt.hist([event_features_df.time[(event_features_df.event > border_event) & (event_features_df.Hbar)]*1e-9, event_features_df.time[(event_features_df.event > border_event) & (~event_features_df.Hbar)]*1e-9], bins=35, stacked=True, label=["Mixing", "BG"])
plt.hist(event_features_df.time[(event_features_df.event > border_event) & (event_features_df.prob_Hbar_XGB > 0.5)]*1e-9, bins=35, histtype="step", label="Hbar XGB prob > 0.5", lw=2)
plt.legend()
plt.xlabel("Time in s")
# plt.yscale("log")
plt.show()

plt.hist([event_features_df.time[(event_features_df.event > border_event) & (event_features_df.Hbar)]*1e-9, event_features_df.time[(event_features_df.event > border_event) & (~event_features_df.Hbar)]*1e-9], bins=35, stacked=True, label=["Mixing", "BG"])
plt.hist(event_features_df.time[(event_features_df.event > border_event) & (event_features_df.prob_Hbar_XGB_nan > 0.5)]*1e-9, bins=35, histtype="step", label="Hbar XGB NaN prob > 0.5", lw=2)
plt.legend()
plt.xlabel("Time in s")
# plt.yscale("log")
plt.show()

triggercondition = clustered_hits["event"].isin(
    np.intersect1d(
        clustered_hits.loc[clustered_hits.detector == "bgo", "event"].unique(),
        clustered_hits.groupby("event")
        .filter(lambda g: 
            ({"hodoO", "hodoI"} <= set(g.loc[
                g.detector.isin(["hodoO", "hodoI"]) &
                g["LE_Us"].notna() &
                g["LE_Ds"].notna(),
                "detector"
            ]))
        )["event"].unique()
    )
)


full_events = clustered_hits[triggercondition].groupby("event").count()


plt.hist(event_features_df.time[event_features_df.cusp == 6453]*1e-9, bins=35, range=(0, 350))
plt.show()

plt.hist(event_features_df.time[(event_features_df.cusp == 6453) & (event_features_df.prob_Hbar_XGBnan > 0.5)]*1e-9, bins=35, range=(0, 350),histtype="step", label="XGB")
plt.hist(event_features_df.time[(event_features_df.cusp == 6453) & (event_features_df.prob_Hbar_RF > 0.5)]*1e-9, bins=35, range=(0, 350), histtype="step", label="RF")
plt.legend()
plt.xlabel("Time in s")
plt.ylabel("Hbar prob. > 0.5")
plt.show()


plt.hist([event_features_df[(event_features_df.event.isin(full_events) & (event_features_df.event > border_event))].time*1e-9, event_features_df[(~event_features_df.event.isin(full_events) & (event_features_df.event > border_event))].time*1e-9], stacked=True)
plt.show()

plt.hist2d(event_features_df.event.isin(full_events), event_features_df.vertex_x.notna(), bins=(2, 2))
plt.colorbar()
plt.xlabel("Is full event?")
plt.ylabel("Has vertex?")
plt.show()

plt.hist2d(event_features_df.event.isin(full_events), event_features_df.n_tracks, bins=(2, 10))
plt.colorbar()
plt.xlabel("Is full event?")
plt.ylabel("Has N tracks?")
plt.show()
           
plt.hist([event_features_df.n_tracks[event_features_df.vertex_x.notna()], event_features_df.n_tracks[~event_features_df.vertex_x.notna()]], bins=15, range=(0, 15), stacked=True, label=["Has vertex", "No vertex"])
plt.xlabel("N tracks")
plt.legend()
plt.show()

ev_arr = np.zeros((event_features_df[event_features_df.Hbar].cusp.nunique(), 5))
for i, run in enumerate(event_features_df[event_features_df.Hbar].cusp.unique()):
    ev_arr[i, 0] = run
    ev_arr[i, 1] = len(event_features_df[(event_features_df.cusp == run)])
    ev_arr[i, 2] = len(event_features_df[(event_features_df.cusp == run) & (event_features_df.vertex) & event_features_df.trigger])
    ev_arr[i, 3] = len(event_features_df[(event_features_df.cusp == run) & (event_features_df.vertex)])
    # ev_arr[i, 4] = len(event_features_df[(event_features_df.cusp == run) & (event_features_df.prob_Hbar_RF > 0.5)])
    ev_arr[i, 4] = len(event_features_df[(event_features_df.cusp == run) & (event_features_df.prob_Hbar_XGB_nan > 0.5)])

plt.plot(ev_arr[1:,0], ev_arr[1:,1], "o", label="Total Events")
plt.plot(ev_arr[1:,0], ev_arr[1:,2], "o", label="Complete Events")
plt.plot(ev_arr[1:,0], ev_arr[1:,3], "o", label="Fittable Events")
plt.plot(ev_arr[1:,0], ev_arr[1:,4], "o", label="XGB Hbar Events")
plt.legend(ncols=2)
plt.xlabel("Run Number")
plt.ylabel("Number of Events")
plt.show()


fig, ax = plt.subplots()
ax.set_aspect("equal")
hist, xbins, ybins, im = ax.hist2d(event_features_df[event_features_df.event > border_event].prob_Hbar_XGB_nan > 0.5, event_features_df[event_features_df.event > border_event].Hbar & event_features_df[event_features_df.event > border_event].trigger, bins=(2,2), range=((0, 1), (0, 1)))
for i in [0, 1]:
    for j in [0,1]:
        ax.text(xbins[j]+0.25,ybins[i]+0.25, int(hist.T[i,j]), 
                color="w", ha="center", va="center", fontweight="bold")
plt.xlabel("ML prediction > 0.5")
plt.ylabel("Trigger Condition + Mixing")
plt.xticks([0.25, 0.75], ["False", "True"])
plt.yticks([0.25, 0.75], ["False", "True"])
plt.title("Hbar Runs")
plt.show()


fig, ax = plt.subplots()
ax.set_aspect("equal")
hist, xbins, ybins, im = ax.hist2d(event_features_df[event_features_df.event < border_event].prob_Hbar_XGB_nan > 0.5, event_features_df[event_features_df.event < border_event].Hbar & event_features_df[event_features_df.event < border_event].trigger, bins=(2,2), range=((0, 1), (0, 1)))
for i in [0, 1]:
    for j in [0,1]:
        ax.text(xbins[j]+0.25,ybins[i]+0.25, int(hist.T[i,j]), 
                color="w", ha="center", va="center", fontweight="bold")
plt.xlabel("ML prediction > 0.5")
plt.ylabel("Trigger Condition + Mixing")
plt.xticks([0.25, 0.75], ["False", "True"])
plt.yticks([0.25, 0.75], ["False", "True"])
plt.title("Cosmic Runs")
plt.show()



# plt.hist([event_features_df[event_features_df.Hbar].vertex_x, event_features_df[~event_features_df.Hbar].vertex_x], bins=50, stacked=False, label=["Hbar", "Background"], density=True, histtype="step")
# plt.yscale("log")

# plt.hist([event_features_df[event_features_df.Hbar].vertex_y, event_features_df[~event_features_df.Hbar].vertex_y], bins=50, stacked=False, label=["Hbar", "Background"], density=True, histtype="step")
# plt.yscale("log")

# plt.hist([event_features_df[event_features_df.Hbar].vertex_z, event_features_df[~event_features_df.Hbar].vertex_z], bins=50, stacked=False, label=["Hbar", "Background"], density=True, histtype="step")
# plt.yscale("log")

# ########### Checking Background ###########

# BGdf_sort = BGdf.sort_values("fpgaTimeTag")

# plt.plot(BGdf_sort.fpgaTimeTag, BGdf_sort.event)

# plt.hist(BGdf_sort.groupby("event").fpgaTimeTag.first().diff()*1e-9, bins=100)
# # plt.yscale("log")    
# plt.show()

# plt.hist(BGdf_sort.groupby("event").fpgaTimeTag.first().diff()*1e-9, bins=100, range=(0,1))
# plt.yscale("log")
# plt.show()

# from scipy.signal.windows import exponential

# def fit_function(M, center, tau):
#     return exponential(int(M), int(center), tau, sym=False)



# fit_exponential(BGdf_sort.groupby("event").fpgaTimeTag.first().diff()*1e-9, BGdf_sort.groupby("event").fpgaTimeTag.first().diff()*1e-9)

# h = plt.hist(BGdf_sort.groupby("event").fpgaTimeTag.first().diff()*1e-9, bins=np.arange(81))

# plt.plot(h[1][1:], h[0]/np.max(h[0]), "o")
# plt.plot(fit_function(80, 0, 10.012))

# popt, pcov = curve_fit(fit_function, 80, h[0]/np.max(h[0]), p0=[0, 9])

# plt.plot(h[1][1:], h[0]/np.max(h[0]), "o")
# plt.plot(fit_function(80, *popt))

# lam = 1 / popt[1]