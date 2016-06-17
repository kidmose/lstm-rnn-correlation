#!/usr/bin/env python
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
"""Tests for data processing functions. """

import unittest
import numpy as np
from pprint import pprint
import re

import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
import lstm_rnn_tied_weights

from operator import itemgetter

class Test(unittest.TestCase):

    def setUp(self):
        pass

    def test_split(self):
        a = list(range(10))
        res = list(lstm_rnn_tied_weights.split(a, [60,20,20]))

        print("res="+str(res))

        lens = map(len, res)
        self.assertSequenceEqual(lens, [6, 2, 2])

        from operator import add
        items = sorted(reduce(add, res))
        self.assertSequenceEqual(items, list(range(10)))

    def test_cross_join(self):
        n = 3
        alerts = [(i, 'alert'+str(i)) for i in range(n)]
        pairs = list(lstm_rnn_tied_weights.cross_join(alerts))
        for p in pairs:
            print p

        self.assertEqual(len(pairs), n**2, msg="Unexpected number of pairs")

        hashable_pairs = [
            (str(p[0]), str(p[1]), str(p[2]), str(p[3]), p[4], p[5], p[6])
            for p in pairs]
        self.assertEqual(len(set(hashable_pairs)), n**2, msg="Unexpected number of unique pairs")

        self.assertEqual(sum(map(itemgetter(4), pairs)), n, msg="Unexpected number correlated pairs")

    def test_cross_join_offset(self):
        n = 3
        offset = 4
        alerts = [(i, 'alert'+str(i)) for i in range(n)]

        pairs = list(lstm_rnn_tied_weights.cross_join(alerts))
        print('Originale pairs:')
        for p in pairs:
            print p
        hashable_pairs = [
            (str(p[0]), str(p[1]), str(p[2]), str(p[3]), p[4], p[5], p[6])
            for p in pairs]

        pairs_os = list(lstm_rnn_tied_weights.cross_join(alerts, offset=offset))
        print('Offset pairs:')
        for p in pairs_os:
            print p
        hashable_pairs_os = [
            (str(p[0]), str(p[1]), str(p[2]), str(p[3]), p[4], p[5], p[6])
            for p in pairs_os]

        self.assertSequenceEqual(hashable_pairs_os, hashable_pairs[offset:], msg="Unexpected pairs")

    def test_uniquify_victim(self):
        # build test data (incidents)
        IP = lambda i: "{}.{}.{}.{}".format(i, i, i, i)
        ALERT = lambda iid, aid: "alert {aid} from incident {iid}, ips: {ip} -> 1.2.3.4".format(aid=aid, iid=iid, ip=IP(iid))
        ALERT_LIST = lambda iid: [ALERT(iid, aid) for aid in range(1,4)]
        incidents = [(iid, ALERT_LIST(iid)) for iid in range(1,6)]
        print("Test incidents:")
        pprint(incidents)

        # replace multiple IPs
        oldips = [IP(i) for i in range(1,6)]
        not_replaced = ["1.2.3.4"]
        new_incidents = lstm_rnn_tied_weights.uniquify_victim(incidents, oldips)
        print("Test incidents, with {} replaced:".format(oldips))
        pprint(new_incidents)
        for ip in oldips:
            self.assertFalse(
                ip in str(new_incidents),
                msg="{} was not replaced".format(ip)
            )
        for ip in not_replaced:
            self.assertTrue(
                ip in str(new_incidents),
                msg="{} was mistakenly replaced".format(ip)
            )
        ips = set(re.findall(lstm_rnn_tied_weights.PATTERN_IP, str(new_incidents)))
        num_ips = len(ips)
        self.assertEqual(
            num_ips,
            5+1,
            msg="Unexpected number of unique IPs in incidents: {}".format(ips),
        )
        for incident, alert_list in incidents:
            ips = set(re.findall(lstm_rnn_tied_weights.PATTERN_IP, str(alert_list)))
            num_ips = len(ips)
            self.assertEqual(
                num_ips,
                1+1,
                msg="Unexpected number of unique IPs in incident {}: {}".format(incident, ips)
            )

