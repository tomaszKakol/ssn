
# import time
from datetime import datetime
import matplotlib.pyplot as plt
import statistics
import pandas as pd
import numpy as np

# desired_width=320
# pd.set_option('display.width', desired_width)
# np.set_printoptions(linewidth=desired_width)
# pd.set_option('display.max_columns', 14)

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)


def Extract(lst, idx):
    return [item[idx] for item in lst]

file = open('mini_lotto.txt', 'r')
NUMBER = 5
linesfile = file.readlines()

loterry_no_column_number = 0
data_column_number = 1
results_column_number = 2

lottery_no_data = []
date_data = []
results_data = []

for x in linesfile:
    lottery_no_data.append(int(x.split()[loterry_no_column_number].split('.')[0]))
    date_data.append(x.split()[data_column_number])
    results_data.append(x.split()[results_column_number])

file.close()

print('lottery_no_data: ', lottery_no_data)
print('date_data: ', date_data)
print('results_data: ', results_data)


split_results_data = [int(sub_e) for e in results_data for sub_e in e.split(',')]
list_results_data = []
list_successive_difference = []
temp_list = []
for idx in range(len(split_results_data)):
    temp_list.append(split_results_data[idx])
    # print(idx)
    if (idx + 1) % NUMBER == 0:
        list_results_data.append(temp_list)
        list_successive_difference.append([temp_list[i + 1] - temp_list[i] for i in range(len(temp_list)-1)])
        temp_list = []


print('list_results_data: ', list_results_data)
print('list_successive_difference: ', list_successive_difference)

timestamp_data = []
for idx in range(len(date_data)):
    # print(date_data[idx])

    day, month, year = date_data[idx].split('.')
    day = int(day)
    month = int(month)
    year = int(year)
    # print(day, month, year)

    if year > 1971:
        timestamp = int(datetime(year=year, month=month, day=day).timestamp())
        # print(timestamp)
        timestamp_data.append(timestamp)

print('timestamp_data: ', timestamp_data)

# print('len date_data: ', len(date_data))
# print('len results_data: ', len(results_data))
# print('len list_results_data: ', len(list_results_data))
# print('len timestamp_data: ', len(timestamp_data))

# print('split_results_data: ', split_results_data)
# print('len split_results_data: ', len(split_results_data))

# count_numbers = Counter(split_results_data)

count_numbers = [[x, split_results_data.count(x)] for x in set(split_results_data)]

print('count_numbers: ', count_numbers)

count_numbers_x = Extract(count_numbers, 0)
count_numbers_y = Extract(count_numbers, 1)
print('count_numbers_x: ', count_numbers_x)
print('count_numbers_y: ', count_numbers_y)


plt.figure(1)
# plt.plot(count_numbers_x, count_numbers_y, 'ro')
plt.bar(count_numbers_x, count_numbers_y, align='center', alpha=0.5)
plt.xlabel('numerek')
plt.ylabel('ilość wystąpień')
plt.title('Wykres łącznej ilości wystąpień danych numerków we wszystkich losowaniach')
# plt.show()

_stdev = []
_variance = []
_mean = []
_harmonic_mean = []
_median = []

_stdev_s_diff = []
_variance_s_diff = []
_mean_s_diff = []
_harmonic_mean_s_diff = []
_median_s_diff = []

for idx in range(len(list_results_data)):
    _stdev.append(statistics.stdev(list_results_data[idx]))
    _variance.append(statistics.variance(list_results_data[idx]))
    _mean.append(statistics.mean(list_results_data[idx]))
    _harmonic_mean.append(statistics.harmonic_mean(list_results_data[idx]))
    _median.append(statistics.median(list_results_data[idx]))

    _stdev_s_diff.append(statistics.stdev(list_successive_difference[idx]))
    _variance_s_diff.append(statistics.variance(list_successive_difference[idx]))
    _mean_s_diff.append(statistics.mean(list_successive_difference[idx]))
    _harmonic_mean_s_diff.append(statistics.harmonic_mean(list_successive_difference[idx]))
    _median_s_diff.append(statistics.median(list_successive_difference[idx]))

'''
print('_stdev: ', _stdev)
print('_variance: ', _variance)
print('_mean: ', _mean)
print('_harmonic_mean: ', _harmonic_mean)
print('_median: ', _median)


plt.figure(2)
# red dashes, blue squares and green triangles
# plt.plot(lottery_no_data, _stdev, 'r')
# plt.plot(lottery_no_data, _stdev, 'r--', lottery_no_data, _mean, 'bs', lottery_no_data, _median, 'g^')
plt.subplot(511)
plt.title('Wykres statystyk z kolejnych losowań (results_data)')
plt.plot(lottery_no_data, _stdev, 'r')
plt.ylabel('stdev')
plt.subplot(512)
plt.plot(lottery_no_data, _variance, 'k')
plt.ylabel('variance')
plt.subplot(513)
plt.plot(lottery_no_data, _mean, 'b')
plt.ylabel('mean')
plt.subplot(514)
plt.plot(lottery_no_data, _median, 'g')
plt.ylabel('median')
plt.subplot(515)
plt.plot(lottery_no_data, _harmonic_mean, 'y')
plt.ylabel('harmonic mean')
plt.xlabel('Próba losowania')

plt.figure(3)
plt.subplot(511)
plt.title('Wykres statystyk z kolejnych losowań (successive_difference)')
plt.plot(lottery_no_data, _stdev_s_diff, 'r')
plt.ylabel('stdev')
plt.subplot(512)
plt.plot(lottery_no_data, _variance_s_diff, 'k')
plt.ylabel('variance')
plt.subplot(513)
plt.plot(lottery_no_data, _mean_s_diff, 'b')
plt.ylabel('mean')
plt.subplot(514)
plt.plot(lottery_no_data, _median_s_diff, 'g')
plt.ylabel('median')
plt.subplot(515)
plt.plot(lottery_no_data, _harmonic_mean_s_diff, 'y')
plt.ylabel('harmonic mean')
plt.xlabel('Próba losowania')
plt.show()
'''

df = pd.DataFrame({"_stdev": _stdev,
                   "_variance": _variance,
                   "_mean": _mean,
                   "_harmonic_mean": _harmonic_mean,
                   "_median": _median,
                   "_stdev_s_diff": _stdev_s_diff,
                   "_variance_s_diff": _variance_s_diff,
                   "_mean_s_diff": _mean_s_diff,
                   "_harmonic_mean_s_diff": _harmonic_mean_s_diff,
                   "_median_s_diff": _median_s_diff,
                   "timestamp_data": timestamp_data,
                   "lottery_no_data": lottery_no_data})


# # To find the covariance
# print(df.cov())
# print('--')
# print('--')
# print('pearson')
# print(df.corr(method='pearson'))
# print('--')
# print('--')
# print('kendall')
# print(df.corr(method='kendall'))
# print('--')
# print('--')
# print('spearman')
# print(df.corr(method='spearman'))
# print('callable')
# print(df.corr(method=callable))
_cov = str(df.cov())
_pearson = str(df.corr(method='pearson'))
_kendall = str(df.corr(method='kendall'))
_spearman = str(df.corr(method='spearman'))

f = open("CovAndCorr.txt", "w+")
f.write('KOWARIANCJA:')
f.write(_cov)
f.write('\n\n\nKORELACJA PEARSON:\n')
f.write(_pearson)
f.write('\n\n\nKORELACJA KENDDALL:\n')
f.write(_kendall)
f.write('\n\n\nKORELACJA SPEARMAN:\n')
f.write(_spearman)

f.close()

print(_pearson)
_pearson.to_csv('CovAndCorr.txt', sep='\t', index=True)









