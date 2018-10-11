import pandas as pd
import numpy as np
from datetime import datetime
from datetime import date
from datetime import timedelta

import math
import copy
from scipy.optimize import minimize
import File_Manager as fm



# List up CSV files from default folder
base_folder = './CSV/'
ex_list = ('FTSE China 50 Total Return', 'iBovespa Futures'
  , 'MSCI Brazil 25-50 Net Return', 'MSCI International EAFE Net'
  , 'MVIS Global Junior Gold Miners TR Net', 'Nifty 50 Futures', 'MVIS Russia TR Net'
  , 'US 10 Year T-Note Futures', 'US 30 Year T-Bond Futures')
datas = fm.ReadCSVFiles(base_folder, ex_list)

# Sampling 데이터가 휴일인 경우 가장 최근 영업일 데이터를 찾기 위해 사용
reference_list = datas.resample('D', on='Date2', convention="end")
reference_datas = datas.loc[datas['Date2'].isin(list(reference_list.indices))]
pivoted_reference_datas = reference_datas.pivot(index='Date2', columns='Name', values='Price')
#print(pivoted_reference_datas)

# Sampling 데이터 생성
sample_list = datas.resample('M', on='Date2', convention="end")
sample_datas = datas.loc[datas['Date2'].isin(list(sample_list.indices))]
pivoted_sample_datas = sample_datas.pivot(index='Date2', columns='Name', values='Price')
#print(pivoted_sample_datas)


# Index의 타입을 Timestamp에서 Date로 변경
pivoted_reference_datas.index = [date(index.year, index.month, index.day) for index in pivoted_reference_datas.index]
pivoted_sample_datas.index = [date(index.year, index.month, index.day) for index in pivoted_sample_datas.index]
# Sampling 데이터가 휴일인 경우 가장 최근 영업일 데이터로 채움
pivoted_inserted_datas = copy.deepcopy(pivoted_sample_datas)
for num, index in enumerate(pivoted_sample_datas.index):
    # 기본로직(Month 단위): After next month의 1일 전일

    # 10월 이후의 경우 After next month는 year가 넘어간다.
    year = index.year + 1 if index.month > 10 else index.year
    if index.month == 11:
        month = 1
    elif index.month == 12:
        month = 2
    else:
        month = index.month + 2

    # After next month의 1일 전으로 월별 말일을 찾음.
    next_last_date = date(year, month, 1) + timedelta(days=-1)

    # 마지말까지 확인전인 경우
    #if num + 1 < len(pivoted_sample_datas.index):
    if pivoted_sample_datas.index[-1] < pivoted_reference_datas.index[-1]:
        #print(num, len(pivoted_sample_datas.index), index, next_last_date, pivoted_sample_datas.index[num+1] == next_last_date)
        #print(next_last_date)
        # 다음 Sampling 데이터가 휴일이어서 데이터가 없는 경우 or 다음 Sampling 데이터와 다음달 말일이 다른 경우
        if next_last_date > pivoted_sample_datas.index[-1] or pivoted_sample_datas.index[num+1] != next_last_date:
            pivoted_inserted_datas = pd.concat([pivoted_inserted_datas, pd.DataFrame(index=[next_last_date], columns=pivoted_inserted_datas.columns)])
# 새로움 Sampling 데이터는 끝에 추가되기 때문에 날짜로 Sorting
pivoted_inserted_datas = pivoted_inserted_datas.sort_index(ascending=1)


pivoted_filled_datas = copy.deepcopy(pivoted_inserted_datas)
for column_nm in pivoted_filled_datas.columns:
    for row_nm in pivoted_filled_datas.index:

        # 값이 포맷이 string인 경우 float으로 변경
        if isinstance(pivoted_filled_datas[column_nm][row_nm], str):
            pivoted_filled_datas[column_nm][row_nm] = float(pivoted_filled_datas[column_nm][row_nm].replace(',',''))

        #print(column_nm, "\t", row_nm, "\t", pivoted_sample_datas[column_nm][row_nm], "\t", pivoted_filled_datas[column_nm][row_nm])
        if column_nm == 'China A50 Futures' and str(row_nm) == '2014-01-31':
            print(1)
        if math.isnan(pivoted_filled_datas[column_nm][row_nm]) == True:
            # ref_row_nm = copy.copy(row_nm)
            #ref_row_nm = str(row_nm)[:10]
            ref_row_nm = row_nm

            # 해당일에 데이터가 없는 경우 가장 최근 값을 대신 사용함
            for loop_cnt in range(10):
                try:
                    float_value = float(pivoted_reference_datas[column_nm][ref_row_nm].replace(',', '')) if isinstance(pivoted_reference_datas[column_nm][ref_row_nm], str) else pivoted_reference_datas[column_nm][ref_row_nm]
                    if math.isnan(float_value) == True:
                        # print("No Data", str(ref_row_nm))
                        #ref_row_nm = str(datetime.strptime(ref_row_nm, '%Y-%m-%d').date() - timedelta(days=1))
                        ref_row_nm = ref_row_nm - timedelta(days=1)
                    else:
                        pivoted_filled_datas[column_nm][row_nm] = float_value
                        break
                except KeyError:
                    # print("KeyError", str(ref_row_nm))
                    #ref_row_nm = str(datetime.strptime(ref_row_nm, '%Y-%m-%d').date() - timedelta(days=1))
                    ref_row_nm = ref_row_nm - timedelta(days=1)

        # 이후 연산작업을 위해 decimal을 float 형태로 변경
        if math.isnan(pivoted_filled_datas[column_nm][row_nm]) == False:
            pivoted_filled_datas[column_nm][row_nm] = float(pivoted_filled_datas[column_nm][row_nm])


# 지수값을 수익률로 변경
pivoted_profit_data = pivoted_filled_datas.rolling(window=2).apply(lambda x: x[1] / x[0] - 1)


# 유효기간을 벗어난 데이터 삭제
pivoted_droped_data = copy.deepcopy(pivoted_profit_data)
row_list = copy.deepcopy(pivoted_droped_data.index)
for row_nm in row_list:
    for column_nm in pivoted_droped_data.columns:
        # 수익률 생성시 문제있는 셀은 nan값
        if math.isnan(pivoted_droped_data[column_nm][row_nm]) == True:
            pivoted_droped_data.drop(index=row_nm, inplace=True)
            pivoted_filled_datas.drop(index=row_nm, inplace=True)
            break



def ObjectiveVol(rets, objective_type, target, lb, ub):
    rets.index = pd.to_datetime(rets.index)
    covmat = pd.DataFrame.cov(rets)
    var_list = pd.DataFrame.var(rets)

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

    def EqualRiskContribution_objective(x):

        variance = x.T @ covmat @ x
        sigma = variance ** 0.5
        mrc = 1 / sigma * (covmat @ x)
        rc = x * mrc
        #a = np.reshape(rc, (len(rc), 1))
        a = np.reshape(rc.values, (len(rc), 1))
        risk_diffs = a - a.T
        sum_risk_diffs_squared = np.sum(np.square(np.ravel(risk_diffs)))

        return (sum_risk_diffs_squared)

    def MinVariance_objective(x):

        variance = x.T @ covmat @ x
        sigma = variance ** 0.5

        return (sigma)

    def MostDiversifiedPortfolio_objective(x):

        portfolio_variance = x.T @ covmat @ x
        portfolio_sigma = portfolio_variance ** 0.5
        weighted_sum_sigma = x @ (var_list ** 0.5)

        return (portfolio_sigma / weighted_sum_sigma)

    # --- Constraints ---#

    def TargetVol_const_lower(x):

        variance = x.T @ covmat @ x
        sigma = variance ** 0.5
        sigma_scale = sigma * np.sqrt(annualize_scale(rets))

        vol_diffs = sigma_scale - (target * 0.95)

        return (vol_diffs)

    def TargetVol_const_upper(x):

        variance = x.T @ covmat @ x
        sigma = variance ** 0.5
        sigma_scale = sigma * np.sqrt(annualize_scale(rets))

        vol_diffs = (target * 1.05) - sigma_scale

        return (vol_diffs)

    def TotalWgt_const(x):

        return x.sum() - 1

    # --- Calculate Portfolio ---#

    x0 = np.repeat(1 / covmat.shape[1], covmat.shape[1])
    #print(x0)
    lbound = np.repeat(lb, covmat.shape[1])
    ubound = np.repeat(ub, covmat.shape[1])
    bnds = tuple(zip(lbound, ubound))
    constraints = ({'type': 'ineq', 'fun': TargetVol_const_lower},
                   {'type': 'ineq', 'fun': TargetVol_const_upper},
                   {'type': 'eq', 'fun': TotalWgt_const})
    options = {'ftol': 1e-20, 'maxiter': 5000, 'disp': False}

    obejctive_func = EqualRiskContribution_objective
    if objective_type == 1:
        obejctive_func = EqualRiskContribution_objective
    elif objective_type == 2:
        obejctive_func = MinVariance_objective
    elif objective_type == 3:
        obejctive_func = MostDiversifiedPortfolio_objective

    result = minimize(fun=obejctive_func,
                      x0=x0,
                      method='SLSQP',
                      constraints=constraints,
                      options=options,
                      bounds=bnds)
    #print(result)
    return (result.fun, result.x)


for objective_type in range(1, 4):
    total_profit = 1
    period_term = 12
    output = {}
    for prd_idx, index in enumerate(pivoted_droped_data.index):

        date = pivoted_droped_data.index[prd_idx + period_term - 1]

        #
        if prd_idx + period_term >= len(pivoted_droped_data):
            print('break', prd_idx + period_term, len(pivoted_droped_data))
            break

        # lb는 자산별 최소비율(%), ub는 자산별 최대비율(%)
        output[date] = {}
        rst_value, rst_weights = ObjectiveVol(pivoted_droped_data[prd_idx:prd_idx + period_term], objective_type, target=0.08, lb=0.0, ub=0.20)

        total_weight = 0
        rst_dict = {"Value": rst_value}
        for rst_idx, weight in enumerate(rst_weights):
            total_weight += weight * 100
            rst_dict[pivoted_droped_data.columns[rst_idx]] = int(weight * 100)
        rst_dict['현금'] = 100 - total_weight

        profit = 0
        if prd_idx + period_term < len(pivoted_droped_data):
            for col_idx, column in enumerate(pivoted_droped_data.columns):
                profit += rst_weights[col_idx] * pivoted_droped_data[column][prd_idx + period_term]
                output[date][col_idx] = rst_weights[col_idx]
            output[date]['Profit'] = profit
        total_profit *= profit + 1
        print(prd_idx, date, profit, total_profit - 1)
    result = pd.DataFrame.from_dict(output).transpose()

    if 1:
        fm.SaveExcelFiles(file='pivoted_data_%s.xlsx' % (objective_type), obj_dict={'pivoted_reference_datas': pivoted_reference_datas
            , 'pivoted_sample_datas': pivoted_sample_datas, 'pivoted_inserted_datas': pivoted_inserted_datas
            , 'pivoted_filled_datas': pivoted_filled_datas, 'pivoted_profit_data': pivoted_profit_data
            , 'pivoted_droped_data': pivoted_droped_data, 'Result': result})