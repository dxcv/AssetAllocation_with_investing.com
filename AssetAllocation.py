import os
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
from dateutil import parser
import math
import copy

folder_dir = './CSV/'

csv_file_list = os.listdir(folder_dir)
print(csv_file_list)

datas = pd.DataFrame()
for csv_file in csv_file_list:
    if csv_file[-3:] != 'csv':
        continue
    print(csv_file[:-20])
    df = pd.read_csv('%s%s' % (folder_dir, csv_file))
    df['Name'] = csv_file[:-20]
    df['Date2'] = df['Date'].apply(lambda x: parser.parse(x))
    datas = pd.concat([datas, df], ignore_index=True)
    #print(df)
print(datas)


reference_list = datas.resample('D', on='Date2', convention="end")
reference_datas = datas.loc[datas['Date2'].isin(list(reference_list.indices))]
pivoted_reference_datas = reference_datas.pivot(index='Date2', columns='Name', values='Price')
#print(pivoted_reference_datas)

sample_list = datas.resample('M', on='Date2', convention="end")
sample_datas = datas.loc[datas['Date2'].isin(list(sample_list.indices))]
pivoted_sample_datas = sample_datas.pivot(index='Date2', columns='Name', values='Price')
print(pivoted_sample_datas)


for column_nm in pivoted_sample_datas.columns:
    for row_nm in pivoted_sample_datas.index:

        if isinstance(pivoted_sample_datas[column_nm][row_nm], str):
            pivoted_sample_datas[column_nm][row_nm] = float(pivoted_sample_datas[column_nm][row_nm].replace(',',''))

        #print(column_nm, "\t", row_nm, "\t", type(pivoted_sample_datas[column_nm][row_nm]))
        if math.isnan(pivoted_sample_datas[column_nm][row_nm]) == True:
            # ref_row_nm = copy.copy(row_nm)
            ref_row_nm = str(row_nm)[:10]

            # 해당일에 데이터가 없는 경우 가장 최근 값을 대신 사용함
            for loop_cnt in range(10):
                try:
                    float_value = float(pivoted_reference_datas[column_nm][ref_row_nm].replace(',', '')) if isinstance(pivoted_reference_datas[column_nm][ref_row_nm], str) else pivoted_reference_datas[column_nm][ref_row_nm]
                    if math.isnan(float_value) == True:
                        # print("No Data", str(ref_row_nm))
                        ref_row_nm = str(datetime.strptime(ref_row_nm, '%Y-%m-%d').date() - timedelta(days=1))
                    else:
                        pivoted_sample_datas[column_nm][row_nm] = float_value
                except KeyError:
                    # print("KeyError", str(ref_row_nm))
                    ref_row_nm = str(datetime.strptime(ref_row_nm, '%Y-%m-%d').date() - timedelta(days=1))

        # 이후 연산작업을 위해 decimal을 float 형태로 변경
        if math.isnan(pivoted_sample_datas[column_nm][row_nm]) == False:
            pivoted_sample_datas[column_nm][row_nm] = float(pivoted_sample_datas[column_nm][row_nm])

# 유효기간을 벗어난 데이터 삭제
row_list = copy.deepcopy(pivoted_sample_datas.index)
for row_nm in row_list:
    for column_nm in pivoted_sample_datas.columns:
        if math.isnan(pivoted_sample_datas[column_nm][row_nm]) == True:
            pivoted_sample_datas.drop(index=row_nm, inplace=True)
            break

print(pivoted_sample_datas)




