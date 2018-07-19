# lstm-rnn-correlation #

An application of Long Short-Term Memory (LSTM) Recurrent Neural
Networks (RNNs) for learning a **mapping function**, mapping strings
into a small, fixed-size **abstract feature space**. Existing
clustering algorithms can then be applied to cluster samples. The key
contribution is that the mapping function is learned from labelled
text samples, such that no feature engineering or domain expertise is
required.

We have applied this method to Intrusion Detection (IDS) alerts,
demonstrating that application of the method can reduce the manual
workload in a Security Operations Center (SOC) roughly one order of
magitude, with hardly any increased risk over overlooking an incident
[^kidmose2017].

## Reuse ##

We strongly encourage reproduction of our results, modifications to
the method and the like, hence we have release this, our own
prototype, under LGPL3.

If reusing this for scientific work, or in anything else that is
published, we kindly request that a reference is made to our paper
presenting the method and the above application [^kidmose2017].  We
will also be very happy to know about such work, so a notification via
email or a PR with a reference will also be appriciated.

## Installation and use ##

We have developped this on python 2.7 (Attempting to be 3.0
compatible, but testing is required) on Debian based Linux
distributions (Linux mint 17, Linux Mint 18, Ubuntu 14.04 Server,
etc). We encourage such an environment for your first installation.
Furthermore we have used and encourage the use of `virtualenv`, for
isolation and the ability to easily roll back.

    virtualenv --system-site-packages -p python2.7 env

Dependencies are installed with:

    pip install --upgrade setuptools pip # Old version of pip causes errors
    pip install -r requirements.txt

Jupyter, is used for development and small test
runs:

    jupyter notebook

For running the full loads, notebooks (`*.ipynb`) are converted to
python scripts (`*.py`):

    jupyter nbconvert --to=script *.ipynb

For ideas on how to run the resulting python script on suitable
platforms, we refer to our scripts for the same:

 * `setup-for-cuda-on-aws.sh`: Dependency installation on Amazon Web
   Services (AWS) instances. Go for GPU.
 * `slurm-job.sl`: Used for starting jobs on the Abacus 2.0 compute
   cluster[^abacus], managed through `slurm`.

## Workflow ##

We have used the following workflow:

 1. Prepare data sets. Refer to folders
    `preprocessing/{mcfp_recorded_merge}/` and the below description
    of data sets.
 2. Train, validate and test the method. Refer to
    `lstm-rnn-tied-weights.ipynb`
 3. Compute and format results for presentation. Refer to
    `results.ipynb`.

## Data sets ##

The data sets used in our work with this approach are the following:

### Malware Capture Facility merged with private benign traffic ###

 1. Retrieve data. Refer to `preprocessing/mcfp_recorded_merge/data_cfg.py` for
    public data on bot malware activity (Note: We downloaded `pcap`s
    and applied the Snort IDS as per [^kidmose2017]).
 2. Record private traffic, validate absence of malicious activity,
    apply the Snort IDS, as per [^kidmose2017].
 3. Clean up data and merge into a single set. Refer to
    `preprocessing/mcfp_recorded_merge/preprocessing.ipynb`.

### CIC-IDS-2017 ###

Homepage for the data set:
http://www.unb.ca/cic/datasets/ids-2017.html.  Private
links/credentials to the data have been provided to us uppon request
by e-mail.

**IMPORTANT - Data inconsistency:** We observed a noteable
inconsistency between the data description on the homepage and the
labels of the flows: `Friday-WorkingHours-Morning.pcap_ISCX.csv`
contains flows labelled as botnet and timestamped from ~10:00 a.m. to
~1:00 p.m., while the description states the botnet was active from
10:02 a.m. – 11:02 a.m.. See `clean-flows.ipynb` for details

 1. Run pcaps through snort, concatenate the log files.
 2. Label alerts according to incident descriptions
    (`preprocessing/cic-ids-2017/incident_descriptions.csv`).
    * A description consists of an label (incident), the attacker and
      victim IP, and a time interval.
    * Each alert gets the label from the description where it falls
      within the time interval and at least one IP in the alert (src
      or dst) matches one IP in the description, not considering the
      routers internal and external IPs (This is to avoid effects of
      NAT).
    * All alerts not matching a description defaults to benign.
 3. Stratify such that no label occurs more than 200 times, and drop
    *benign* to match the total size of the MCFP dataset.


## References ##

[^kidmose2017]: Egon Kidmose, Matija Stevanovic, Søren Brandbyge and
  Jens M. Pedersen, Automating the discovery of correlated and false
  intrusion alerts by using neural networks.

[^abacus]: [https://abacus.deic.dk/](https://abacus.deic.dk/)
