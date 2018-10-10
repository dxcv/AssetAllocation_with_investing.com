import xlsxwriter
import openpyxl
import pandas as pd
import os
from dateutil import parser


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


def ReadCSVFiles(base_folder, ex_list):
    csv_files = os.listdir(base_folder)
    # print(csv_files)

    # Read data from CSV files and Make data frame
    datas = pd.DataFrame()
    for csv_file in csv_files:
        # Pass, if not CSV file
        if csv_file[-3:] != 'csv':
            continue

        # 예외 상품 제외
        if csv_file[:-20] in ex_list:
            print('Pass: ', csv_file[:-20])
            continue

        df = pd.read_csv('%s%s' % (base_folder, csv_file))  # Read CSV file
        df['Name'] = csv_file[:-20]  # Set name of data
        df['Date2'] = df['Date'].apply(lambda x: parser.parse(x))  # Change date format
        datas = pd.concat([datas, df], ignore_index=True)  # Merge new data to previous data
        # print(df)

    # print(datas)
    return datas

