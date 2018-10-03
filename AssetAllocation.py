import os
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import date
from datetime import timedelta
from dateutil import parser
import xlsxwriter
import openpyxl
import math
import copy
from scipy.optimize import minimize



def SaveExcelFiles(file='test.xlsx', obj_dict=None):

    # 만약 해당 파일이 존재하지 않는 경우 생성
    workbook = xlsxwriter.Workbook(file)
    workbook.close()

    # Create a Pandas Excel writer using Openpyxl as the engine.
    writer = pd.ExcelWriter(file, engine='openpyxl')
    # 주의: 파일이 암호화 걸리면 workbook load시 에러 발생
    writer.book = openpyxl.load_workbook(file)
    # Pandas의 DataFrame 클래스를 그대로 이용해서 엑셀 생성 가능
    for obj_nm in obj_dict:
        obj_dict[obj_nm].to_excel(writer, sheet_name=obj_nm)
    writer.save()

    return True

# List up CSV files from default folder
folder_dir = './CSV/'
csv_file_list = os.listdir(folder_dir)
#print(csv_file_list)

# Read data from CSV files and Make data frame
datas = pd.DataFrame()
for csv_file in csv_file_list:
    # Pass, if not CSV file
    if csv_file[-3:] != 'csv':
        continue

    df = pd.read_csv('%s%s' % (folder_dir, csv_file)) # Read CSV file
    df['Name'] = csv_file[:-20] # Set name of data
    df['Date2'] = df['Date'].apply(lambda x: parser.parse(x)) # Change date format
    datas = pd.concat([datas, df], ignore_index=True) # Merge new data to previous data
    #print(df)
#print(datas)


reference_list = datas.resample('D', on='Date2', convention="end")
reference_datas = datas.loc[datas['Date2'].isin(list(reference_list.indices))]
pivoted_reference_datas = reference_datas.pivot(index='Date2', columns='Name', values='Price')
#print(pivoted_reference_datas)

sample_list = datas.resample('M', on='Date2', convention="end")
sample_datas = datas.loc[datas['Date2'].isin(list(sample_list.indices))]
pivoted_sample_datas = sample_datas.pivot(index='Date2', columns='Name', values='Price')
#print(pivoted_sample_datas)



pivoted_sample_datas.index = [date(index.year, index.month, index.day) for index in pivoted_sample_datas.index]
pivoted_inserted_datas = copy.deepcopy(pivoted_sample_datas)
for num, index in enumerate(pivoted_sample_datas.index):
    year = index.year + 1 if index.month > 10 else index.year
    if index.month == 11:
        month = 1
    elif index.month == 12:
        month = 2
    else:
        month = index.month + 2

    next_last_date = date(year, month, 1) + timedelta(days=-1)

    if len(pivoted_sample_datas.index) > num + 1:
        #print(num, len(pivoted_sample_datas.index), index, next_last_date, pivoted_sample_datas.index[num+1] == next_last_date)
        if pivoted_sample_datas.index[num+1] != next_last_date:
            pivoted_inserted_datas = pd.concat([pivoted_inserted_datas, pd.DataFrame(index=[next_last_date], columns=pivoted_inserted_datas.columns)])
pivoted_inserted_datas = pivoted_inserted_datas.sort_index(ascending=1)


pivoted_filled_datas = copy.deepcopy(pivoted_inserted_datas)
for column_nm in pivoted_filled_datas.columns:
    for row_nm in pivoted_filled_datas.index:

        if isinstance(pivoted_filled_datas[column_nm][row_nm], str):
            pivoted_filled_datas[column_nm][row_nm] = float(pivoted_filled_datas[column_nm][row_nm].replace(',',''))

        #print(column_nm, "\t", row_nm, "\t", type(pivoted_sample_datas[column_nm][row_nm]))
        if math.isnan(pivoted_filled_datas[column_nm][row_nm]) == True:
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
                        pivoted_filled_datas[column_nm][row_nm] = float_value
                except KeyError:
                    # print("KeyError", str(ref_row_nm))
                    ref_row_nm = str(datetime.strptime(ref_row_nm, '%Y-%m-%d').date() - timedelta(days=1))

        # 이후 연산작업을 위해 decimal을 float 형태로 변경
        if math.isnan(pivoted_filled_datas[column_nm][row_nm]) == False:
            pivoted_filled_datas[column_nm][row_nm] = float(pivoted_filled_datas[column_nm][row_nm])


pivoted_profit_data = pivoted_filled_datas.rolling(window=2).apply(lambda x: x[1] / x[0] - 1)


# 유효기간을 벗어난 데이터 삭제
pivoted_droped_data = copy.deepcopy(pivoted_profit_data)
row_list = copy.deepcopy(pivoted_droped_data.index)
for row_nm in row_list:
    for column_nm in pivoted_droped_data.columns:
        if math.isnan(pivoted_droped_data[column_nm][row_nm]) == True:
            pivoted_droped_data.drop(index=row_nm, inplace=True)
            break


if 0:
    SaveExcelFiles(file='pivoted_data.xlsx', obj_dict={'pivoted_reference_datas': pivoted_reference_datas
        , 'pivoted_sample_datas': pivoted_sample_datas, 'pivoted_inserted_datas': pivoted_inserted_datas
        , 'pivoted_filled_datas': pivoted_filled_datas, 'pivoted_profit_data': pivoted_profit_data
        , 'pivoted_droped_data': pivoted_droped_data})




def RP_TargetVol(rets, Target, lb, ub):
    rets.index = pd.to_datetime(rets.index)
    covmat = pd.DataFrame.cov(rets)

    def annualize_scale(rets):

        med = np.median(np.diff(rets.index.values))
        seconds = int(med.astype('timedelta64[s]').item().total_seconds())
        if seconds < 60:
            freq = 'second'.format(seconds)
        elif seconds < 3600:
            freq = 'minute'.format(seconds // 60)
        elif seconds < 86400:
            freq = 'hour'.format(seconds // 3600)
        elif seconds < 604800:
            freq = 'day'.format(seconds // 86400)
        elif seconds < 2678400:
            freq = 'week'.format(seconds // 604800)
        elif seconds < 7948800:
            freq = 'month'.format(seconds // 2678400)
        else:
            freq = 'quarter'.format(seconds // 7948800)

        def switch1(x):
            return {
                'day': 252,
                'week': 52,
                'month': 12,
                'quarter': 4,
            }.get(x)

        return switch1(freq)

    # --- Risk Budget Portfolio Objective Function ---#

    def RiskParity_objective(x):

        variance = x.T @ covmat @ x
        sigma = variance ** 0.5
        mrc = 1 / sigma * (covmat @ x)
        rc = x * mrc
        #a = np.reshape(rc, (len(rc), 1))
        a = np.reshape(rc.values, (len(rc), 1))
        risk_diffs = a - a.T
        sum_risk_diffs_squared = np.sum(np.square(np.ravel(risk_diffs)))
        return (sum_risk_diffs_squared)

    # --- Constraints ---#

    def TargetVol_const_lower(x):

        variance = x.T @ covmat @ x
        sigma = variance ** 0.5
        sigma_scale = sigma * np.sqrt(annualize_scale(rets))

        vol_diffs = sigma_scale - (Target * 0.95)
        return (vol_diffs)

    def TargetVol_const_upper(x):

        variance = x.T @ covmat @ x
        sigma = variance ** 0.5
        sigma_scale = sigma * np.sqrt(annualize_scale(rets))

        vol_diffs = (Target * 1.05) - sigma_scale
        return (vol_diffs)

        # --- Calculate Portfolio ---#

    x0 = np.repeat(1 / covmat.shape[1], covmat.shape[1])
    #print(x0)
    lbound = np.repeat(lb, covmat.shape[1])
    ubound = np.repeat(ub, covmat.shape[1])
    bnds = tuple(zip(lbound, ubound))
    constraints = ({'type': 'ineq', 'fun': TargetVol_const_lower},
                   {'type': 'ineq', 'fun': TargetVol_const_upper})
    options = {'ftol': 1e-20, 'maxiter': 5000, 'disp': True}

    result = minimize(fun=RiskParity_objective,
                      x0=x0,
                      method='SLSQP',
                      constraints=constraints,
                      options=options,
                      bounds=bnds)
    print(result)
    return (result.x)


period_term = 12
for prd_idx, a in enumerate(pivoted_droped_data.index):
    if prd_idx + period_term > len(pivoted_droped_data):
        print('break', prd_idx + period_term, len(pivoted_droped_data))
        break

    result = RP_TargetVol(pivoted_droped_data[prd_idx:prd_idx + period_term], Target=0.20, lb=0.10, ub=0.30)
    #print(result)

    total_weight = 0
    result_dict = {}
    for rst_idx, weight in enumerate(result):
        total_weight += weight
        #print(pivoted_droped_data.columns[idx], weight)
        result_dict[pivoted_droped_data.columns[rst_idx]] = float(weight * 100)
    #print('현금', 1.0 - total_weight)
    result_dict['현금'] = 1.0 - total_weight
    print(prd_idx, pivoted_droped_data.index[prd_idx + period_term - 1], result_dict)

