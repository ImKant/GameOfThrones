#_*_ encoding=utf-8 _*_

import tushare as ts
import numpy as np
data1 = ts.get_h_data('002230')
print data1.head(100)
''''
ktype
默认为D日线数据
D=日k线 W=周 M=月
5=5分钟 15=15分钟
30=30分钟 60=60分钟

'''
data2 = ts.get_k_data('002230', ktype='W',autype='qfq')
print data2.tail(100)