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

Dependencies are installed with:

    pip install -r requirements.txt

Ipython notebook, or jupyter, is used for development and small test
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

 1. Retrieve data. Refer to `data_cfg.py` for public data on bot
    malware activity (Note: We downloaded `pcap`s and applied the
    Snort IDS as per [^kidmose2017]).
 2. Clean up data and merge into a single set. Refer to
    `preprocessing.ipynb`.
 3. Train, validate and test the method. Refer to
    `lstm-rnn-tied-weights.ipynb`
 4. Compute and format results for presentation. Refer to
    `results.ipynb`.

## References

[^kidmose2017]: Egon Kidmose, Matija Stevanovic, Søren Brandbyge and
  Jens M. Pedersen, Automating the discovery of correlated and false
  intrusion alerts by using neural networks.

[^abacus]: [https://abacus.deic.dk/](https://abacus.deic.dk/)
