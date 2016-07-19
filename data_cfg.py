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

import pandas as pd
import numpy as np

data = pd.DataFrame([
#'data/mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-132-1/2015-09-09_win3.pcap.out', #	1
#'data/mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-114-2/2015-04-22_capture-win2.pcap.out', #	1
#'data/mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-22/2013-11-06_capture-win8.pcap.out', #	1
#'data/mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-121-1/2015-04-22_capture-win5.pcap.out', #	1
#'data/mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-15/2013-09-28_capture-win19.pcap.out', #	2
#'data/mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-129-1/2015-06-30_capture-win20.pcap.out', #	3
#'data/mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-45/botnet-capture-20110815-rbot-dos-icmp.pcap.out', #	3
#'data/mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-143-1/2015-10-23_win6.pcap.out', #	3
#'data/mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-59/2014-03-12_capture-win15.pcap.out', #	4
#'data/mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-45/botnet-capture-20110815-rbot-dos-icmp-more-bandwith.pcap.out', #	4
#'data/mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-123-1/2015-04-22_capture-win8.pcap.out', #	4
#'data/mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-14/2013-10-18_capture-win15.pcap.out', #	5
#'data/mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-45/botnet-capture-20110815-rbot-dos.pcap.out', #	5
#'data/mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-118-1/2015-04-20_capture-win5.pcap.out', #	6
#'data/mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-92/192.168.3.104-eldorado2-1.pcap.out', #	6
#'data/mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-48/botnet-capture-20110816-sogou.pcap.out', #	10
#'data/mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-60/2014-03-12_win20.pcap.out', #	12
#'data/mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-11/capture-win19.pcap.out', #	13
#'data/mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-102/capture-win2.pcap.out', #	13
#'data/mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-90/192.168.3.104-unvirus.pcap.out', #	18
#'data/mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-134-1/2015-10-11_win3.pcap.out', #	20
#'data/mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-26/2013-10-30_capture-win10.pcap.out', #	21
#'data/mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-114-1/2015-04-09_capture-win2.pcap.out', #	22
#'data/mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-73/2014-05-16_capture-win15.pcap.out', #	28
#'data/mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-24/2013-11-06_capture-win18.pcap.out', #	29
#'data/mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-142-1/2015-10-23_win7.pcap.out', #	85
#'data/mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-47/botnet-capture-20110816-donbot.pcap.out', #	88
#'data/mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-65/2014-04-07_capture-win11.pcap.out', #	90
('data/mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-46/botnet-capture-20110815-fast-flux.pcap.out', '147.32.84.165'), #	100
('data/mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-113-1/2015-03-12_capture-win6.pcap.out', '10.0.2.106'), #	184
('data/mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-2/2013-08-20_capture-win2.pcap.out', '10.0.2.16'), #	317
('data/mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-116-1/2012-05-25-capture-1.pcap.out', '192.168.0.9'), #	328
('data/mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-89-1/2014-09-15_capture-win2.pcap.out', '10.0.2.102'), #	390
('data/mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-36/capture-win2.pcap.out', '10.0.2.102'), #	395
('data/mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-49/botnet-capture-20110816-qvod.pcap.out', '147.32.84.165'), #	444
#'data/mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-128-1/2015-06-07_capture-win12.pcap.out', #	611
#'data/mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-140-1/2015-10-23_win11.pcap.out', #	839
#'data/mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-42/botnet-capture-20110810-neris.pcap.out', #	865
#'data/mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-55/capture-win13.pcap.out', #	954
#'data/mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-140-2/2015-10-27_capture-win11.pcap.out', #	1354
#'data/mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-141-1/2015-23-10_win10.pcap.out', #	1548
#'data/mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-69/2014-04-07_capture-win17.pcap.out', #	1704
#'data/mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-43/botnet-capture-20110811-neris.pcap.out', #	1785
#'data/mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-54/botnet-capture-20110815-fast-flux-2.pcap.out', #	2015
#'data/mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-100/2014-12-20_capture-win5.pcap.out', #	2685
#'data/mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-35-1/2014-01-31_capture-win7.pcap.out', #	3199
#'data/mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-149-2/2015-12-09_capture-win4.pcap.out', #	3817
#'data/mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-127-1/2015-06-07_capture-win8.pcap.out', #	4900
#'data/mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-44/botnet-capture-20110812-rbot.pcap.out', #	5338
#'data/mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-149-1/2015-12-09_capture-win4.pcap.out', #	5896
#'data/mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-126-1/2015-06-07_capture-win7.pcap.out', #	6992
#'data/mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-125-1/2015-06-07_capture-win5.pcap.out', #	7461
#'data/mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-110-4/2015-04-22_capture-win9.pcap.out', #	17501
#'data/mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-150-1/2015-12-05_capture-win3.pcap.out', #	18854
#'data/mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-3/2013-08-20_capture-win15.pcap.out', #	38279
#'data/mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-78-1/2014-05-30_capture-win8.pcap.out', #	84863
])
data.columns = ['filename', 'victim_ip']
data['incident'] = map(str, np.arange(data.shape[0])+1)

data.loc[data.shape[0]] =[
    'data/own-recordings/filteredalerts.log',
    'benign', # Victim IP - None in benign
    'benign', # Benign, So 'None' (None, empty object, is dropped when joining)
]
