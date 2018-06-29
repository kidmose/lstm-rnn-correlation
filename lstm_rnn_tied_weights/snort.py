import pandas as pd
import re
import datetime


"""
Deals with snort alerts.
"""


IPV4 = '(?:[0-9]{1,3}(?:\.[0-9]{1,3}){3})'
IPV6 = '(?:[0-9a-f]|:){1,4}(?::(?:[0-9a-f]{0,4})*){1,7}'
IP = '(?:{}|{})'.format(IPV4, IPV6)
IP_PORT = '('+IP+')(?::([^ ]+))?'

SNORT_REGEX = re.compile('^(.*)  \[\*\*] \[([^]]*)] (.*) \[Priority: ([0-9])] {([^}]*)} ('+IP+')(?::([^ ]+))? -> ('+IP+')(?::([^ ]+))?\n')
SNORT_TS_FMT = '%m/%d/%y-%H:%M:%S.%f'
SNORT_TS_FMT_NO_YR = '%m/%d-%H:%M:%S.%f'

def strptime(string):
    ts = None
    try:
        ts = datetime.datetime.strptime(string, SNORT_TS_FMT)
    except:
        pass
    try:
        ts = datetime.datetime.strptime(string, SNORT_TS_FMT_NO_YR)
    except:
        pass
    if ts is None:
        raise Exception('Failed to parse {}: {}'.format(type(string), string))
    return ts

def strftime(ts):
    return ts.strftime(SNORT_TS_FMT)

def parse_line(line):
    tupl = re.match(SNORT_REGEX, line).groups()
    tupl = tuple([strptime(tupl[0])]) + tupl[1:]
    return tupl


def build_line(tupl):
    tupl = tuple(tupl) # pandas.core.series.Series doesn't add like tuple 
    tupl = tuple([strftime(tupl[0])]) + tupl[1:]
    return "{}  [**] [{}] {} [Priority: {}] {{{}}} {}:{} -> {}:{}\n".format(*tupl)

test_code =\
"""
for fn in data['filename']:
    for l in open(fn).readlines():
        p = parse_line(l)
        b = build_line(p)
        assert l == b or l == (b[:5]+b[8:]), "Mismatch in result"
"""

test_row = pd.Series((
        pd.Timestamp('2016-05-04 09:38:44.365433'),
        '129:12:1',
        'Consecutive TCP small segments exceeding threshold [**] [Classification: Potentially Bad Traffic]',
        '2',
        'TCP',
        '10.149.34.24',
        '445',
        '10.130.8.48',
        '60741',
))
desired_output = '05/04/16-09:38:44.365433  [**] [129:12:1] '+\
    'Consecutive TCP small segments exceeding threshold [**] [Classification: Potentially Bad Traffic] '+\
    '[Priority: 2] {TCP} 10.149.34.24:445 -> 10.130.8.48:60741\n'

assert build_line(test_row) == desired_output

difficult_ips = [
    '94.63.149.152',
    '147.32.84.165',
]
for ip in difficult_ips:
    assert re.match('('+IP+')', ip).group()

difficult_ips_port = [
    '94.63.149.152:80',
    '147.32.84.165:1040',
]
for ip in difficult_ips_port:
    res = re.match(IP_PORT, ip).groups()
    assert res is not None
    assert len(res) == 2

difficult_lines = [
    '08/15-15:53:48.900440  [**] [120:3:1] (http_inspect) NO CONTENT-LENGTH OR TRANSFER-ENCODING IN HTTP RESPONSE [**] [Classification: Unknown Traffic] [Priority: 3] {TCP} 94.63.149.152:80 -> 147.32.84.165:1040\n',
]
for l in difficult_lines:
    res = parse_line(l)
    assert len(res) == 9
