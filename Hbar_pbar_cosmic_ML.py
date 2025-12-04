%load_ext autoreload
%autoreload 2

from read_file import *
from find_tracks import *
from fit_tracks import *
from fit_vertex import *
from draw_tracks import *
from get_event_features import *
from ML_functions import *
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
border_event_pbar = BGdf.event.max()

run_list = np.arange(2078, 2081)
pbardf = build_hits_df_from_runs(run_list, version="cusp_run")
#pbardf = pbardf[pbardf.fpgaTimeTag > 225e9]
pbardf["Hbar_BG"] = ["pbar" if m == True else "BG" for m in pbardf["mixGate"]]
pbardf.loc[:, "event"] = pbardf["event"] + max_ev

max_ev = pbardf["event"].max()
border_event = pbardf.event.max()

run_list = np.concatenate([np.arange(1705, 1721), np.arange(1724, 1730)])
Hbardf = build_hits_df_from_runs(run_list, version="cusp_run")
Hbardf["Hbar_BG"] = ["Hbar" if m == True else "BG" for m in Hbardf["mixGate"]]
Hbardf.loc[:, "event"] = Hbardf["event"] + max_ev


hits_df = pd.concat([BGdf, pbardf, Hbardf], ignore_index=True)

# hits_df = pd.concat([BGdf, Hbardf], ignore_index=True)
# hits_df = pd.concat([BGdf[BGdf.event <= 5000], Hbardf], ignore_index=True)
# hits_df = Hbardf

hits_df["z_used"] = np.where(np.isnan(hits_df["z_reco"]), hits_df["z"], hits_df["z_reco"])
hits_df["dz_used"] = np.where(np.isnan(hits_df["z_reco"]), hits_df["dz"], hits_df["dz_reco"])
hits_df["bgoToT"] = hits_df[(hits_df["detector"] == "bgo") & hits_df["LE"].notna() & hits_df["TE"].notna()& (hits_df["LE"] < hits_df["TE"])]["TE"] - hits_df[(hits_df["detector"] == "bgo") & hits_df["LE"].notna() & hits_df["TE"].notna() & (hits_df["LE"] < hits_df["TE"])]["LE"]


params = {
    "eps": [0.8],                            
    "zweight_same": [1], 
    "zweight_diff": [0],  
    "weight_power": [0.5], #[0.25, 0.5, 0.75],
    "weight_power_z": [0.85], #[0.7, 0.8, 0.85, 0.9], #[0.75, 0.8, 0.9, 1],
    "dist_bgo": [300],
    "vertex_cluster": [True],
    "vertex_eps": [200], #, 300, 400], #[50, 100, 150, 200, 250, 300, 350, 400],
    "vertex_alpha": [100] #[1, 10, 50, 100]
}

eps = params["eps"][0]
z_w_same = params["zweight_same"][0]
z_w_diff = params["zweight_diff"][0]
w_pow = params["weight_power"][0]
w_pow_z = params["weight_power_z"][0]
dist_bgo = params["dist_bgo"][0]
vertex_cluster = params["vertex_cluster"][0]
vertex_eps = params["vertex_eps"][0]
vertex_alpha = params["vertex_alpha"][0]

clustered_list = []
for event_id, ev in hits_df[hits_df.LE < 0].groupby("event"):
    ev["det_key"] = ev["detector"] + "_" + ev["channel"].astype(str)

    # select only hodo/tile hits for clustering
    ev_ht = ev[ev.detector.isin(["hodoO","hodoI","tileO","tileI"])].copy()
    if ev_ht.empty:
        ev["track_id"] = -1
        clustered_list.append(ev)
        continue

    labels = cluster_by_phi_layer_uncertainty(ev_ht,
            base_eps=eps,
            min_samples=2,
            z_weight_same=z_w_same,
            z_weight_diff=z_w_diff,
            sigma_floor_deg=0.5,
            coords="cylindrical")
    
    # labels = cluster_by_phi_uncertainty(ev_ht, base_eps=eps, min_samples=2, theta_weight=th_w, coords="cylindrical")
    ev_ht["track_id"] = labels

    # merge labels back into the full event (bgo keep -1)
    ev = ev.merge(ev_ht[["det_key","track_id"]], on="det_key", how="left")
    ev["track_id"] = ev["track_id"].fillna(-1).astype(int)

    clustered_list.append(ev)

print("clustering done")

clustered_hits = pd.concat(clustered_list, ignore_index=True)

_lines_df = fit_lines_from_clusters_svd(clustered_hits, include_bgo=False, 
                                    use_xyz_errors=True, xyz_error_cols=["dx", "dy", "dz_used"], weight_power=w_pow, weight_power_z=w_pow_z, prefilter_ransac=False, ransac_thresh=15.0, weighted=False, weight_col="dz_used")

print("line fitting done")

vertices_df = reconstruct_vertex_from_midpoints(clustered_hits, _lines_df,
                                        bgo_radius=45.0, 
                                        max_dist_to_bgo=dist_bgo, 
                                        cluster_mids=vertex_cluster, cluster_eps=vertex_eps, cluster_alpha=vertex_alpha)#25.0)

print("vertex reconstruction done")


# vertices_df = find_vertices_from_tracks(lines_df, eps=5.0)
lines_df = _lines_df.merge(vertices_df, on="event", how="left")

clustered_hits = clustered_hits.merge(lines_df, on=["event", "track_id"], how="left")

plot_events(clustered_hits, lines_df, clustered_hits[clustered_hits.Hbar_BG == "pbar"].event.unique()[:30])

event_features_df = compute_event_features_from_clustered_hits(
    clustered_hits,
    bgo_center=(0, 0, 0)
)

print("event features done")

run_feats = set(["event", "cusp", "time", "mix", "Hbar"])
all_feats = [ele for ele in event_features_df.columns if ele not in run_feats]
ML_feats = [ele for ele in event_features_df.columns if ele not in set(["vertex", "trigger", "Annihilation", "event", "cusp", "time", "mix", "Hbar"])]

for feat in all_feats:
    # plt.hist([event_features_df[event_features_df.Hbar][feat].values, event_features_df[~event_features_df.Hbar][feat].values], bins=100, stacked=True, label=["Mixing", "BG"])
    # plt.legend()
    # plt.xlabel(feat)
    # plt.show()

    plt.hist([event_features_df[event_features_df.Annihilation == 0][feat].values, event_features_df[event_features_df.Annihilation == 0.9][feat].values, event_features_df[event_features_df.Annihilation == 1][feat].values], bins=100, stacked=False, label=["BG", "Hbar", "pbar"], histtype="step", density=True)
    plt.legend()
    plt.xlabel(feat)
    plt.show()


    plt.hist([event_features_df[event_features_df.Annihilation > 0.5][feat].values, event_features_df[~(event_features_df.Annihilation > 0.5)][feat].values], bins=100, stacked=True, label=["Mixing", "BG"])
    plt.legend()
    plt.xlabel(feat)
    plt.show()




# training data sets
evs = np.array(event_features_df["event"])
X = np.array(event_features_df[ML_feats])
y_reg = np.array(event_features_df["Annihilation"])
y_cla = np.array(event_features_df["Annihilation"] > 0.5).astype(int)


# Splitting the dataset
X_train, X_test, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=0.2, random_state=0)

X_train, X_test, y_cla_train, y_cla_test = train_test_split(X, y_cla, test_size=0.2, random_state=0)


sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


model_reg = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=250,
    learning_rate=0.1,
    subsample=0.6,
    colsample_bytree=0.5,
    max_depth=4,
)
model_reg.fit(X_train, y_reg_train)

model_cla = xgb.XGBClassifier(n_estimators=250,
    learning_rate=0.1,
    subsample=0.6,
    colsample_bytree=0.5,
    max_depth=4,
    )
model_cla.fit(X_train, y_cla_train)

plot_importance(model_reg.feature_importances_, ML_feats, "XGBoost Regressor Feature Importance")

plot_importance(model_cla.feature_importances_, ML_feats, "XGBoost Classifier Feature Importance")

y_reg_pred = (model_reg.predict(X_test) > 0.5).astype(int)
y_reg_probs = model_reg.predict(X_test)

y_cla_pred = model_cla.predict(X_test)
y_cla_probs = model_cla.predict_proba(X_test)[:, 1]

evaluate_model((y_reg_test > 0.5).astype(int), y_reg_pred, y_reg_probs, "XGBoost Regressor")

evaluate_model(y_cla_test, y_cla_pred, y_cla_probs, "XGBoost Classifier")

y_reg_pred = cross_val_predict(
    model_reg,
    X,
    y_reg,
    cv=5,            # 5-fold cross-validation
    n_jobs=-1,       # use all CPU cores
    method="predict" # default, so optional
)


y_cla_pred = cross_val_predict(
    model_cla,
    X,
    y_cla,
    cv=5,            # 5-fold cross-validation
    n_jobs=-1,       # use all CPU cores
    method="predict_proba" # default, so optional
)[:,1]

plt.hist([y_reg_pred, y_cla_pred], bins=100, stacked=False, label=["XGBoost Regressor", "XGBoost Classifier"], histtype="step", lw=2)
plt.legend()
plt.xlabel("Probability of Annihilation")
plt.show()


prob_df = pd.DataFrame(np.array([evs, y_reg_pred, y_cla_pred]).T, columns=["event","prob_Annihilation_reg", "prob_Annihilation_cla"])


event_features_df = pd.merge(event_features_df, prob_df, on="event", how="left")

def plot_roc_prc(y_true, proba, mask=None, model_name="Model", title="All Runs"):

    if mask is not None:
        y_true = y_true[mask]
        proba = proba[mask]
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    # Compute ROC
    fpr, tpr, _ = roc_curve(y_true, proba)
    auc_score = auc(fpr, tpr)

    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f_scores = (2 * precision * recall) / (precision + recall)

    # Find the threshold with the maximal F-score
    max_f_score_idx = np.argmax(f_scores)
    max_f_score_threshold = thresholds[max_f_score_idx]

    ax[0].plot(fpr, tpr, ".", label=f"XGB (AUC = {auc_score:.3f})")
    ax[1].plot(recall, precision,".", label=f"XGB (F-score = {f_scores[max_f_score_idx]:.3f})")
    ax[1].scatter(recall[max_f_score_idx], precision[max_f_score_idx], marker="x", color="black", zorder=5)#, label=f"{name} (F-score = {f_scores[max_f_score_idx]:.3f})")

    # Plot formatting
    ax[0].plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax[1].plot([0, 1], [0.5, 0.5], 'k--', linewidth=1)
    ax[0].set_xlabel("False Positive Rate")
    ax[0].set_ylabel("True Positive Rate")
    ax[0].set_title(f"ROC Curves for {model_name}")
    ax[0].legend(fontsize=8)
    ax[1].set_xlabel("Recall")
    ax[1].set_ylabel("Precision")
    ax[1].set_title(f"Precision-Recall Curves for {model_name}")
    ax[1].legend(fontsize=8)
    plt.tight_layout()
    plt.suptitle(f"{model_name}: {title}", fontsize=16, y=0.98)
    fig.subplots_adjust(top=0.85)
    plt.show()


plot_roc_prc(event_features_df["Annihilation"].values > 0.5, event_features_df["prob_Annihilation_reg"].values.copy(), mask=None, model_name="XGB Regressor", title="All Runs")

plot_roc_prc(event_features_df["Annihilation"].values > 0.5, event_features_df["prob_Annihilation_reg"].values.copy(), mask=event_features_df.event > border_event_pbar, model_name="XGB Regressor", title="Pbar/Mixing Runs")

plot_roc_prc(event_features_df["Annihilation"].values > 0.5, event_features_df["prob_Annihilation_reg"].values.copy(), mask=event_features_df.event > border_event, model_name="XGB Regressor", title="Mixing Runs")



plot_roc_prc(event_features_df["Annihilation"].values > 0.5, event_features_df["prob_Annihilation_cla"].values.copy(), mask=None, model_name="XGB Classifier", title="All Runs")

plot_roc_prc(event_features_df["Annihilation"].values > 0.5, event_features_df["prob_Annihilation_cla"].values.copy(), mask=event_features_df.event > border_event_pbar, model_name="XGB Classifier", title="Pbar/Mixing Runs")

plot_roc_prc(event_features_df["Annihilation"].values > 0.5, event_features_df["prob_Annihilation_cla"].values.copy(), mask=event_features_df.event > border_event, model_name="XGB Classifier", title="Mixing Runs")



plot_events(clustered_hits, lines_df, event_features_df.event[(event_features_df.prob_Annihilation_cla > 0.8) & (event_features_df.prob_Annihilation_reg > 0.8)].values[:30])



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


for mod in ["prob_Annihilation_reg", "prob_Annihilation_cla"]:
    pb_cm = probability_band_confusion_matrix(
        y_true=event_features_df["Annihilation"].values > 0.5,
        y_proba=event_features_df[mod].values,
        model_name=mod
    )


for thresh in np.arange(0.1, 1.0, 0.1):

    plt.hist([event_features_df[(event_features_df.prob_Annihilation_cla > thresh) & (event_features_df.event > border_event)].time*1e-9, event_features_df[(event_features_df.prob_Annihilation_cla <= thresh) & (event_features_df.event > border_event)].time*1e-9], bins=50, stacked=False, label=[f"Annihilation prob > {thresh:.1f}", f"Annihilation prob <= {thresh:.1f}"], density=False, histtype="step")
    plt.legend()
    plt.title("Mixing Runs")
    plt.show()

for thresh in np.arange(0.1, 1.0, 0.1):

    plt.hist([event_features_df[(event_features_df.prob_Annihilation_cla > thresh) & (border_event < event_features_df.event) & (event_features_df.event > border_event_pbar)].time*1e-9, event_features_df[(event_features_df.prob_Annihilation_cla <= thresh) & (border_event < event_features_df.event) & (event_features_df.event > border_event_pbar)].time*1e-9], bins=50, stacked=False, label=[f"Annihilation prob > {thresh:.1f}", f"Annihilation prob <= {thresh:.1f}"], density=False, histtype="step")
    plt.legend()
    plt.title("pbar Extraction")
    plt.show()






