
# coding: utf-8

# Copyright (C) Egon Kidmose 2015-2017
# 
# This file is part of lstm-rnn-correlation.
# 
# lstm-rnn-correlation is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
# 
# lstm-rnn-correlation is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public
# License along with lstm-rnn-correlation. If not, see
# <http://www.gnu.org/licenses/>.

# # Alert correlation with Laten Semantic Analysis
# 
# Implemented as an example of an alternative to lstm-rnn-tied-weights.
# Aim is to be have input and output of the same format as lstm-rnn-tied-weights, in order to easy data preparation and calculation of metrics.

# In[ ]:


import pickle
import time
import os
import sys
import logging
import datetime
import socket
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

import pandas as pd
import scipy as sp
import numpy as np
import h5py

import matplotlib
try: # If X is not available select backend not requiring X
    os.environ['DISPLAY']
except KeyError:
    matplotlib.use('Agg')
try: # If ipython, do inline
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass
import matplotlib.pyplot as plt


# In[ ]:


logger = logging.getLogger('lsa')
logger.handlers = []
logger.setLevel(logging.DEBUG)

OUTPUT = 'output'
runid = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
if os.environ.get('SLURM_ARRAY_TASK_ID', False):
    runid += '-slurm-{}_{}'.format(
        os.environ['SLURM_ARRAY_JOB_ID'],
        os.environ['SLURM_ARRAY_TASK_ID']
    )
elif os.environ.get('SLURM_JOB_ID', False):
    runid += '-slurm-' + os.environ['SLURM_JOB_ID']
else:
     runid += '-' + socket.gethostname()
out_dir = OUTPUT + '/' + runid
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
out_prefix = out_dir + '/' + runid + '-'

# info log file
infofh = logging.FileHandler(out_prefix + 'info.log')
infofh.setLevel(logging.INFO)
infofh.setFormatter(logging.Formatter(
        fmt='%(message)s',
))
logger.addHandler(infofh)

# verbose log file
vfh = logging.FileHandler(out_prefix + 'verbose.log')
vfh.setLevel(logging.DEBUG)
vfh.setFormatter(logging.Formatter(
        fmt='%(asctime)s - PID:%(process)d - %(levelname)s - %(message)s',
))
logger.addHandler(vfh)

# Console logger
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter())
logger.addHandler(ch)

logger.info('Output, including logs, are going to: {}'.format(out_dir))


# In[ ]:


# parse env vars
env = dict()

# Data control
env['CSVIN'] = os.environ.get('CSVIN', None)
if env['CSVIN'] is None:
    logger.critical("Cannot run without a CSV input")
    sys.exit(-1)
env['RAND_SEED'] = int(os.environ.get('RAND_SEED', time.time())) # Current unix time if not specified
env['VAL_CUT'] = int(os.environ.get('VAL_CUT', -1))
if not (0 <= env['VAL_CUT']) and (env['VAL_CUT'] <= 9):
    logger.critical("Invalid cross validation cut: {}".format(env['VAL_CUT']))
    sys.exit(-1)

logger.info("Starting.")
logger.info("env: " + str(env))
for k in sorted(env.keys()):
    logger.info('env[\'{}\']: {}'.format(k,env[k]))

logger.debug('Saving env')
with open(out_prefix + 'env.json', 'w') as f:
    json.dump(env, f)


# In[ ]:


seed = env['RAND_SEED']
def rndseed():
    global seed
    seed += 1
    return seed

def shuffle(pairs):
    np.random.seed(rndseed())
    return pairs.reindex(np.random.permutation(pairs.index))

class Timer(object):
    def __init__(self, name='', log=logger.info):
        self.name = name
        self.log = log
        
    def __enter__(self):
        self.log('Timer(%s) started' % (self.name, ))
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.dur = datetime.timedelta(seconds=self.end - self.start)
        self.log('Timer(%s):\t%s' % (self.name, self.dur))


# ## Load data

# In[ ]:


# load data
logger.info('Loading data')
data = pd.read_csv(env['CSVIN'])
data.loc[data.incident == 'benign','incident'] = '-1' # encode benign with -1
data.incident = data.incident.astype(int) # parse strings'

idx_train = data.cut != env['VAL_CUT']
idx_val = data.cut == env['VAL_CUT']

logger.info(data.shape)
logger.info(data.groupby(data.incident).incident.count())


# ## Build LSA pipeline

# In[ ]:


# vectoriser
vectoriser = TfidfVectorizer(
    max_df=0.5, # ignore tokens occuring in more than # of alerts
    max_features=10000, # only # most common tokens
    min_df=2, # Only if token is in more than two alerts
    analyzer='char', # Do N-gram analysis
    ngram_range=(1, 3)
)

vectoriser.fit(data[idx_train].alert)

logger.info(
    "Features in tf-idf vector: %d" % 
    vectoriser.transform(data[idx_train].alert).get_shape()[1]
)

# SVD
SVD_COMPONENTS = 100
svd = TruncatedSVD(SVD_COMPONENTS)

# lsa
lsa = make_pipeline(
    vectoriser,
    svd, 
    Normalizer(copy=False)
)

# Fit pipeline
_ = lsa.fit(data[idx_train].alert)

logger.info(
    "Variance explained by SVD using %d components: %.3f %%" %
    (SVD_COMPONENTS, svd.explained_variance_ratio_.sum() * 100. )
)


# ## Tranform data

# In[ ]:


# Tranform data
data = data.join(
    pd.DataFrame(
        lsa.transform(data.alert), 
        index=data.index,
    )
)


# In[ ]:


logger.info(data.shape)


# # Clustering

# In[ ]:


def precompute_distance_matrix(X):
    """
    precomputing takes 20 sec/500 samples on js3, OMP_NUM_THREADS=16
    precomputing takes 9 min/2632 samples on js3, OMP_NUM_THREADS=16
    """
    return sp.spatial.distance.cdist(X, X, metric='cosine')


# In[ ]:


logger.info('Getting alerts for clustering')
clust_alerts = {
    'train': (data[idx_train][list(range(SVD_COMPONENTS))].values, data[idx_train].incident.values),
    'validation': (data[idx_val][list(range(SVD_COMPONENTS))].values, data[idx_val].incident.values),
}

logger.info("Precomputing clustering alert distances")

X = dict()
X_dist_precomp = dict()
y = dict()
for (cut, v) in clust_alerts.items():
    alerts_matrix, incidents_vector = v
    X[cut] = alerts_matrix
    X_dist_precomp[cut] = precompute_distance_matrix(X[cut])
    y[cut] = incidents_vector


# ## Clustering

# In[ ]:


import sklearn
from sklearn.cluster import DBSCAN
from sklearn import metrics

def cluster(eps, min_samples, X_dist_precomp):
    return DBSCAN(
        eps=eps, min_samples=min_samples, metric='precomputed'
    ).fit(X_dist_precomp)

def build_cluster_to_incident_mapper(y, y_pred):
    # Assign label to clusters according which incident has the largest part of its alert in the given cluster
    # weight to handle class skew
    weights = {l: 1/cnt for (l, cnt) in zip(*np.unique(y, return_counts=True))}
    allocs = zip(y, y_pred)

    from collections import Counter
    c = Counter(map(tuple, allocs))

    mapper = dict()
    for _, (incident, cluster) in sorted([(c[k]*weights[k[0]], k) for k in c.keys()]):
        mapper[cluster] = incident

    mapper[-1] = -1 # Don't rely on what DBSCAN deems as noise
    return mapper

def dbscan_predict(dbscan_model, X_new, metric=sp.spatial.distance.cosine):
    # Result is noise by default
    y_new = np.ones(shape=len(X_new), dtype=int)*-1 

    # Iterate all input samples for a label
    for j, x_new in enumerate(X_new):
        for i, x_core in enumerate(X['train'][dbscan_model.core_sample_indices_]): 
            if  metric(x_new, x_core) < dbscan_model.eps:
                # Assign label of x_core to x_new
                y_new[j] = dbscan_model.labels_[dbscan_model.core_sample_indices_[i]]
                break
    return y_new


# In[ ]:


logger.info("Iterating clustering algorithm parameters")
epss = np.array([0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1])
min_sampless = np.array([1, 3, 10, 30])

def ParamSpaceMatrix(dtype=None):
    return np.zeros(shape=(len(epss), len(min_sampless)), dtype=dtype)

cl_model = ParamSpaceMatrix(dtype=object)
mapper = ParamSpaceMatrix(dtype=object)

cuts = ['train', 'validation']

def ParamSpaceMatrices(dtype=None):
    return {cut : ParamSpaceMatrix(dtype=dtype) for cut in cuts}


# In[ ]:


# Cluster and build mapper
for i, eps in enumerate(epss):
    for j, min_samples in enumerate(min_sampless):
        logger.info("Clustering, eps={}, min_samples={}".format(eps, min_samples))
        # Cluster
        cl_model[i,j] = cluster(eps, min_samples, X_dist_precomp['train'])
        # get cluster assignments
        y_pred = cl_model[i,j].labels_ 
        # Build classifier - get mapper used for classification
        mapper[i,j] = build_cluster_to_incident_mapper(y['train'], y_pred)


# In[ ]:


# predict 
y_pred = ParamSpaceMatrices(dtype=object)
y_pred_inc = ParamSpaceMatrices(dtype=object)

for cut in cuts:
    for i, eps in enumerate(epss):
        for j, min_samples in enumerate(min_sampless):
            if cut == 'train':
                # pred is abused to hold clustering results
                y_pred[cut][i,j] = cl_model[i,j].labels_ # cluster assignment
                y_pred_inc[cut][i,j] = y[cut] # true incident label
            elif cut == 'validation':
                logger.info('Predicting for (eps, min_samples)=({:1.0e},{:>2d})'.format(eps, min_samples))
                y_pred[cut][i,j] = dbscan_predict(cl_model[i][j], X[cut])
                y_pred_inc[cut][i,j] = np.array([mapper[i,j][el] for el in y_pred[cut][i,j]]) # predict incident
            else:
                raise NotImplementedError('Unexpected value for cut:{}'.format(cut))
            


# In[ ]:


def false_alert_rate_outliers_score(y, y_pred):
    idx_outliers = y_pred == -1
    return (y[idx_outliers] == -1).mean()

def false_alert_rate_clusters_score(y, y_pred):
    idx_clusters = y_pred != -1
    return (y[idx_clusters] == -1).mean()

def arf_score(y, y_pred):
    return len(y) / len(set(y_pred))

def narf_score(y, y_pred):
    return (len(y) / len(set(y_pred)) - 1) / (len(y) - 1)

def imr_score(y, y_pred):
    df_clustering = pd.DataFrame({
        'cluster': y_pred,
        'inc_true': y,
    })
    cluster_sizes = pd.DataFrame({'cluster_size': df_clustering[df_clustering.cluster != -1].groupby('cluster').size()})
    df_clustering = pd.merge(df_clustering, cluster_sizes.reset_index(), on='cluster', how='outer')
    # asuming one alert is picked at random from each cluster;
    # probability that given alert is picked to represent the cluster it belongs to
    df_clustering['alert_pick_prob'] = 1/df_clustering.cluster_size 
    # probability distribution of what incident a cluster will be asumed to represent
    df_prob = pd.DataFrame(df_clustering.groupby(['cluster', 'inc_true']).sum().alert_pick_prob.rename('inc_hit'))
    # probability that a given incident will not come out of a cluster
    df_prob['inc_miss'] = 1 - df_prob.inc_hit.fillna(0)
    assert (df_prob[df_prob.inc_miss < 0].inc_miss.abs() < 1e-12).all(), "Error larger than 1e-12, still just imprecission?"
    df_prob = df_prob.abs()
    # ... of any cluster
    inc_miss_prob = df_prob.reset_index().groupby('inc_true').inc_miss.prod().rename('inc_miss_prob')
    inc_miss_prob = inc_miss_prob[inc_miss_prob.index != -1] # Don't care about missing the noise pseudo-incident
    return inc_miss_prob.sum() / df_clustering[df_clustering.inc_true != -1].inc_true.unique().shape[0]

def cm_inc_clust(y, y_pred):
    cm_inc_clust = pd.DataFrame(
        metrics.confusion_matrix(y, y_pred),
        index=sorted(set.union(set(y), set(y_pred))),
        columns=sorted(set.union(set(y), set(y_pred))),
    )
    # drop dummy row for non-existing incident IDs
    assert (cm_inc_clust.drop(list(set(y)), axis=0) == 0).values.all(), "Non-empty row for invalid incident id"
    cm_inc_clust = cm_inc_clust.loc[sorted(list(set(y)))]

    # drop dummy collumns for non-existing cluster IDs
    assert (cm_inc_clust.drop(list(set(y_pred)), axis=1) == 0).values.all(), "Non-empty collumn for invalid cluster id"
    cm_inc_clust = cm_inc_clust[sorted(list(set(y_pred)))]

    cm_inc_clust.rename(index={-1: 'benign'}, inplace=True)
    cm_inc_clust.rename(columns={-1: 'noise'}, inplace=True)
    return cm_inc_clust

def cm_inc_inc(y, y_pred_inc):
    cm_inc_inc = pd.DataFrame(
        metrics.confusion_matrix(y, y_pred_inc),
        index=sorted(set.union(set(y), set(y_pred_inc))),
        columns=sorted(set.union(set(y), set(y_pred_inc))),
    )

    cm_inc_inc.rename(index={-1: 'benign'}, inplace=True)
    cm_inc_inc.rename(columns={-1: 'benign'}, inplace=True)
    return cm_inc_inc

def per_class_metrics(y, y_pred_inc):
    d = dict(
        zip(
            ['precission', 'recall', 'f1', 'support'],
            metrics.precision_recall_fscore_support(y, y_pred_inc)
        )
    )
    cmat = metrics.confusion_matrix(y, y_pred_inc)
    d['accuracy'] = cmat.diagonal()/cmat.sum(axis=1)
    d['labels'] = sklearn.utils.multiclass.unique_labels(y, y_pred_inc)
    return d


# In[ ]:


# calculating metrics
m = {
    # clustering
    'n_clusters': ParamSpaceMatrices(dtype=int),
    'homogenity': ParamSpaceMatrices(),
    'outliers': ParamSpaceMatrices(dtype=int),
    # classification, general
    'accuracy': ParamSpaceMatrices(),
    'precision': ParamSpaceMatrices(),
    'recall': ParamSpaceMatrices(),
    'f1': ParamSpaceMatrices(),
    'per_class': ParamSpaceMatrices(dtype=object),
    # correlating and filtering metrics
    'arf': ParamSpaceMatrices(),
    'narf': ParamSpaceMatrices(),
    'imr': ParamSpaceMatrices(),
    'faro': ParamSpaceMatrices(),
    'farc': ParamSpaceMatrices(),
    # confusion matrices
    'cm_inc_clust': ParamSpaceMatrices(dtype=object),
    'cm_inc_inc': ParamSpaceMatrices(dtype=object),
}

for cut in cuts:
    for i, eps in enumerate(epss):
        for j, min_samples in enumerate(min_sampless):
            # clustering metrics
            # Number of clusters in labels, ignoring noise if present.
            m['n_clusters'][cut][i,j] = len(set(y_pred[cut][i,j])) - (1 if -1 in y_pred[cut][i,j] else 0)
            m['outliers'][cut][i,j] = sum(y_pred[cut][i,j] == -1)
            m['homogenity'][cut][i,j] = metrics.homogeneity_score(y[cut], y_pred[cut][i,j])
            # classification metrics
            m['accuracy'][cut][i,j] = metrics.accuracy_score(y[cut], y_pred_inc[cut][i,j])
            m['precision'][cut][i,j] = metrics.precision_score(y[cut], y_pred_inc[cut][i,j], average='weighted')
            m['recall'][cut][i,j] = metrics.recall_score(y[cut], y_pred_inc[cut][i,j], average='weighted')
            m['f1'][cut][i,j] = metrics.f1_score(y[cut], y_pred_inc[cut][i,j], average='weighted')
            m['per_class'][cut][i,j] = per_class_metrics(y[cut], y_pred_inc[cut][i,j])
            # correlating and filtering
            m['arf'][cut][i,j] = arf_score(y[cut], y_pred[cut][i,j])
            m['narf'][cut][i,j] = narf_score(y[cut], y_pred[cut][i,j])
            m['imr'][cut][i,j] = imr_score(y[cut], y_pred[cut][i,j])
            m['faro'][cut][i,j] = false_alert_rate_outliers_score(y[cut], y_pred[cut][i,j])
            m['farc'][cut][i,j] = false_alert_rate_clusters_score(y[cut], y_pred[cut][i,j])
            # confusion matrices
            m['cm_inc_clust'][cut][i,j] = cm_inc_clust(y[cut], y_pred[cut][i,j])
            m['cm_inc_inc'][cut][i,j] = cm_inc_inc(y[cut], y_pred_inc[cut][i,j])
            
            logger.info(
                "Performance on {} cut with (eps, min_samples)=({:1.0e},{:>2d}): n_clusters={:>3d}, homogenity={:1.3f}, f1={:1.3f}, noise={:>3d}".format(
                    cut, eps, min_samples, 
                    m['n_clusters'][cut][i,j],
                    m['homogenity'][cut][i,j],
                    m['f1'][cut][i,j],
                    m['outliers'][cut][i,j],
                )
            )
        


# In[ ]:


# save metrics to hdf5
import h5py

with h5py.File(out_prefix + 'metrics.h5', 'w'): # truncate
    pass

def traverse_metrics(m, hdf5file, path=''):
    logger.debug('Saving %s' % (path))
    if isinstance(m, dict):
        for k in m:
            traverse_metrics(m[k], hdf5file, path='%s/%s' % (path, k,))
        return
    elif isinstance(m, np.ndarray) and m.dtype == np.dtype('O'):
        for x in range(0, m.shape[0]):
            for y in range(0, m.shape[1]):
                traverse_metrics(m[x,y], hdf5file, path='%s/index_%d_%d' % (path, x, y))
        return
    elif isinstance(m, pd.DataFrame):
        m.to_hdf(hdf5file, path)
    else:
        with h5py.File(hdf5file, 'a') as hf:
            hf.create_dataset(path,  data=m)
    logger.debug('Saved %s' % (path, ))

traverse_metrics(m, out_prefix + 'metrics.h5')


# ## Plotting

# In[ ]:


def param_plot_prepare(
    title,
):
    fig, ax = plt.subplots(figsize=(10,6))
    ax.set_title(title)

    ax.set_yscale('log')
    ax.set_ylabel('min_samples')
    ax.set_yticklabels(min_sampless)
    ax.set_yticks(min_sampless)
    ax.set_ylim(min_sampless[0]/3, min_sampless[-1]*3)

    ax.set_xscale('log')
    ax.set_xlabel('eps')
    ax.set_xticklabels(epss)
    ax.set_xticks(epss)
    ax.set_xlim(epss[0]/3,epss[-1]*3)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.0e'))


def param_plot_scatter(
    data,
    xcoords,
    ycoords,

):
    # scaling
    data = np.copy(data)
    data = data-data.min() # Range to start at zero
    data = data/data.max() # Range to end at one

    coords = np.array([
            (i, j)
            for i in range(data.shape[0])
            for j in range(data.shape[1])
    ])

    plt.scatter(
        xcoords[coords[:,0]],
        ycoords[coords[:,1]],
        data[coords[:,0], coords[:,1]]*1000,
        c='white',
        alpha=0.5
    )


def param_plot_annotate(
    data,
    xcoords,
    ycoords,
    fmt='{}',
):
    coords = np.array([
            (i, j)
            for i in range(data.shape[0])
            for j in range(data.shape[1])
    ])
        
    for x, y, label in zip(
        xcoords[coords[:,0]],
        ycoords[coords[:,1]],
        data[coords[:,0], coords[:,1]],
    ):
        plt.annotate(
            fmt.format(label),
            xy = (x, y), xytext = (0, 0),
            textcoords = 'offset points', ha = 'center', va = 'center',
        )


def param_plot_save(filename):
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')


# In[ ]:


for label, values, fmt in [
    ('Cluster count', m['n_clusters'], '{}'),
    ('Cluster homogenity', m['homogenity'], '{:.2f}'),
    ('Outliers', m['outliers'], '{}'),
]:
    for cut in cuts:
        param_plot_prepare('{} by DBSCAN parameters ({} alerts)'.format(label, cut))
        param_plot_scatter(values[cut], epss, min_sampless)
        param_plot_annotate(values[cut], epss, min_sampless, fmt=fmt)
        
        param_plot_save(
            '{}{}_{}.pdf'.format(
                out_prefix,
                label.replace('(','').replace(')','').replace(" ", "_").lower(),
                cut
            )
        )


# In[ ]:


for label, values, fmt in [
    ('Incident prediction accuracy', m['accuracy'], '{:.2f}'),
    ('Incident prediction precision', m['precision'],  '{:.2f}'),
    ('Incident prediction recall', m['recall'], '{:.2f}'),
    ('Incident prediction F1 score', m['f1'], '{:.2f}'),
]:
    for cut in ['validation']: # training evaluated on true labels for clustering is of little use
        param_plot_prepare('{} by DBSCAN parameters ({} alerts)'.format(label, cut))
        param_plot_scatter(values[cut], epss, min_sampless)
        param_plot_annotate(values[cut], epss, min_sampless, fmt=fmt)
        
        param_plot_save(
            '{}{}_{}.pdf'.format(
                out_prefix,
                label.replace('(','').replace(')','').replace(" ", "_").lower(),
                cut
            )
        )


# In[ ]:


for label, values, fmt in [
    ('Alert Reduction Factor', m['arf'], '{:.2f}'),
    ('Normalised Alert Reduction Factor', m['narf'], '{:.2e}'),
    ('Incident Miss Rate', m['imr'], '{:.2e}'),
    ('False alerts rate among outliers', m['faro'], '{:.2f}'),
]:
    for cut in cuts:
        param_plot_prepare('{} by DBSCAN parameters ({} alerts)'.format(label, cut))
        param_plot_scatter(values[cut], epss, min_sampless)
        param_plot_annotate(values[cut], epss, min_sampless, fmt=fmt)
        
        param_plot_save(
            '{}{}_{}.pdf'.format(
                out_prefix,
                label.replace('(','').replace(')','').replace(" ", "_").lower(),
                cut
            )
        )


# In[ ]:


logger.info('One fold of cross validation completed')
sys.exit(0)

