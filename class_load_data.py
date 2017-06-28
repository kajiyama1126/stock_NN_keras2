import copy
import numpy as np
import datetime
import os
import pandas as pd
from scipy.stats import zscore


class load_data(object):
    def __init__(self,data_place,days,start,end):
        self.place = data_place
        self.days = days
        self.start = start
        self.end = end

    def read_stock_data(self):
        days = self.days
        i= 0
        basename = ['start', 'high', 'low', 'last']
        column_name = copy.copy(basename)
        for i in range(days - 1):
            for j in basename:
                column_name.append(j + str(i + 1))

        teacher_name = ['teacher_only_up_down']

        place = self.place
        directory = os.listdir(place)

        stock_data = np.empty((0, 4 * days), float)
        teacher_data = np.empty((0, 1), int)

        for name in directory:
            i += 1
            if i<self.start:
                pass
            else:
                print(name)
                os.chdir(place)
                data = pd.read_csv(name, encoding='cp932', index_col=0)
                x,y = self.read_data_oneday(data,column_name,teacher_name,normalization_option=True)
                # index_name = data.index
                # stock_data_onedata = np.empty((0, 4 * days), float)
                # teacher_data_onedata = np.empty((0, 1), int)
                # for name1 in index_name:
                #     stock_data_normalization = zscore(data.ix[name1, column_name])
                #     stock_data_tmp = np.array([stock_data_normalization])
                #     teacher_data_tmp_tmp = data.ix[name1, teacher_name]
                #     if teacher_data_tmp_tmp[0] > 0:
                #         tmp = int(1)
                #     else:
                #         tmp = int(0)
                #
                #     teacher_data_tmp = np.array([[tmp]])
                #     # print(stock_data_tmp)
                #     stock_data_onedata = np.append(stock_data_onedata, stock_data_tmp, axis=0)
                #     teacher_data_onedata = np.append(teacher_data_onedata, teacher_data_tmp, axis=0)

                # if name == '2010-06-30column_renamefor15day_reformconnect.csv':
                #     break

                #
                stock_data = np.append(stock_data, x, axis=0)
                teacher_data = np.append(teacher_data, y, axis=0)
                if i>= self.end:
                    break
                # print(stock_data)
        today = datetime.datetime.today()
        todayname = str(today.strftime("%Y%m%d_%H%M%S"))
        np.savetxt('x_train_test_norm' + todayname +'.csv', stock_data, delimiter=',')
        np.savetxt('y_train_test_norm' + todayname +'.csv', teacher_data, delimiter=',')

        return stock_data, teacher_data

    def read_data_oneday(self,data,column_name,teacher_name,normalization_option):

        days = self.days
        index_name = data.index
        stock_data_onedata = np.empty((0, 4 * days ), float)
        teacher_data_onedata = np.empty((0, 1), int)
        for name1 in index_name:
            onedata = data.ix[name1, column_name]
            if normalization_option:
                onedata = zscore(onedata)
            stock_data_tmp = np.array([onedata])
            teacher_data_tmp_tmp = data.ix[name1, teacher_name]
            if teacher_data_tmp_tmp[0] > 0:
                tmp = int(1)
            else:
                tmp = int(0)

            teacher_data_tmp = np.array([[tmp]])
            # print(stock_data_tmp)
            stock_data_onedata = np.append(stock_data_onedata, stock_data_tmp, axis=0)
            teacher_data_onedata = np.append(teacher_data_onedata, teacher_data_tmp, axis=0)

        return stock_data_onedata,teacher_data_onedata

    def load_stock_data(self,x_data,y_data):
        place = self.place
        os.chdir(place)
        stock_data = np.loadtxt(x_data, delimiter=',')
        teacher_data = np.loadtxt(y_data, delimiter=',')

        return stock_data, teacher_data

class load_data_add_dekidaka(load_data):
    def __init__(self,data_place,days,start,end):
        super(load_data_add_dekidaka,self).__init__(data_place,days,start,end)
        number_of_sale = ['number of sale']
        money_of_sale = ['money of sale']
        self.number_of = copy.copy(number_of_sale)
        self.money_of = copy.copy(money_of_sale)
        for i in range(days-1):
            self.number_of.append(number_of_sale[0] + str(i+1))
            self.money_of.append(money_of_sale[0] + str(i+1))

    def read_stock_data(self):
        days = self.days
        i= 0
        basename = ['start', 'high', 'low', 'last']
        column_name = copy.copy(basename)
        for i in range(days - 1):
            for j in basename:
                column_name.append(j + str(i + 1))

        teacher_name = ['teacher_only_up_down']

        place = self.place
        directory = os.listdir(place)

        stock_data = np.empty((0, 5 * days), float)
        teacher_data = np.empty((0, 1), int)

        for name in directory:
            i += 1
            if i<self.start:
                pass
            else:
                print(name)
                os.chdir(place)
                data = pd.read_csv(name, encoding='cp932', index_col=0)
                x,y = self.read_data_oneday(data,column_name,teacher_name,normalization_option=True)
                # index_name = data.index
                # stock_data_onedata = np.empty((0, 4 * days), float)
                # teacher_data_onedata = np.empty((0, 1), int)
                # for name1 in index_name:
                #     stock_data_normalization = zscore(data.ix[name1, column_name])
                #     stock_data_tmp = np.array([stock_data_normalization])
                #     teacher_data_tmp_tmp = data.ix[name1, teacher_name]
                #     if teacher_data_tmp_tmp[0] > 0:
                #         tmp = int(1)
                #     else:
                #         tmp = int(0)
                #
                #     teacher_data_tmp = np.array([[tmp]])
                #     # print(stock_data_tmp)
                #     stock_data_onedata = np.append(stock_data_onedata, stock_data_tmp, axis=0)
                #     teacher_data_onedata = np.append(teacher_data_onedata, teacher_data_tmp, axis=0)

                # if name == '2010-06-30column_renamefor15day_reformconnect.csv':
                #     break
                if i>= self.end:
                    break
                #
                stock_data = np.append(stock_data, x, axis=0)
                teacher_data = np.append(teacher_data, y, axis=0)
                # print(stock_data)
        today = datetime.datetime.today()
        todayname = str(today.strftime("%Y%m%d_%H%M%S"))
        np.savetxt('x_train_test_norm' + todayname +'.csv', stock_data, delimiter=',')
        np.savetxt('y_train_test_norm' + todayname +'.csv', teacher_data, delimiter=',')

        return stock_data, teacher_data

    def read_data_oneday(self,data,column_name,teacher_name,normalization_option):

        days = self.days

        index_name = data.index
        stock_data_onedata = np.empty((0, 5 * days ), float)
        teacher_data_onedata = np.empty((0, 1), int)
        for name1 in index_name:
            onedata = data.ix[name1, column_name]
            for i in range(self.days):
                onedata.append(data.ix[name1, self.money_of[i]]/data.ix[name1,self.number_of[i]])
            if normalization_option:
                onedata = zscore(onedata)
            stock_data_tmp = np.array([onedata])
            teacher_data_tmp_tmp = data.ix[name1, teacher_name]
            if teacher_data_tmp_tmp[0] > 0:
                tmp = int(1)
            else:
                tmp = int(0)

            teacher_data_tmp = np.array([[tmp]])
            # print(stock_data_tmp)
            stock_data_onedata = np.append(stock_data_onedata, stock_data_tmp, axis=0)
            teacher_data_onedata = np.append(teacher_data_onedata, teacher_data_tmp, axis=0)

        return stock_data_onedata,teacher_data_onedata

class load_date_separate(load_data):
    def __init__(self,date_place,days,start,end,code):
        super(load_date_separate,self).__init__(date_place,days,start,end)
        self.code = code

    def read_data_oneday(self,data,column_name,teacher_name,normalization_option):

        days = self.days
        # index_name = data.index
        stock_data_onedata = np.empty((0, 4 * days ), float)
        teacher_data_onedata = np.empty((0, 1), int)

        for stock_code in self.code:
            onedata = data.ix[stock_code, column_name]

            if normalization_option:
                onedata = zscore(onedata)
            stock_data_tmp = np.array([onedata])
            teacher_data_tmp_tmp = data.ix[stock_code, teacher_name]
            if teacher_data_tmp_tmp[0] > 0:
                tmp = int(1)
            else:
                tmp = int(0)

            teacher_data_tmp = np.array([[tmp]])

            stock_data_onedata = np.append(stock_data_onedata, stock_data_tmp, axis=0)
            teacher_data_onedata = np.append(teacher_data_onedata, teacher_data_tmp, axis=0)

        return stock_data_onedata, teacher_data_onedata

    def read_stock_data(self):
        days = self.days
        i= 0
        basename = ['start', 'high', 'low', 'last']
        column_name = copy.copy(basename)
        for i in range(days - 1):
            for j in basename:
                column_name.append(j + str(i + 1))

        teacher_name = ['teacher_only_up_down']

        place = self.place
        directory = os.listdir(place)

        stock_data = np.empty((0, 4 * days), float)
        teacher_data = np.empty((0, 1), int)

        for name in directory:
            i += 1
            if i<self.start:
                pass
            else:
                print(name)
                os.chdir(place)
                data = pd.read_csv(name, encoding='cp932', index_col=0)
                x,y = self.read_data_oneday(data,column_name,teacher_name,normalization_option=True)
                if i>= self.end:
                    break
                #
                stock_data = np.append(stock_data, x, axis=0)
                teacher_data = np.append(teacher_data, y, axis=0)
                # print(stock_data)
        today = datetime.datetime.today()
        todayname = str(today.strftime("%Y%m%d_%H%M%S"))

        np.savetxt('x_train_test_norm' + todayname +'code'+'.csv', stock_data, delimiter=',')
        np.savetxt('y_train_test_norm' + todayname +'code'+'.csv', teacher_data, delimiter=',')

        return stock_data, teacher_data

class load_data_separate_dekidaka(load_date_separate):
    def read_data_oneday(self,data,column_name,teacher_name,dekidaka_name,normalization_option):

        days = self.days
        # index_name = data.index
        stock_data_onedata = np.empty((0, 6 * days ), float)
        teacher_data_onedata = np.empty((0, 1), int)

        for stock_code in self.code:
            onedata = data.ix[stock_code, column_name]
            onedata2 = data.ix[stock_code, dekidaka_name]

            if normalization_option:
                onedata = zscore(onedata)
                onedata2 = zscore(onedata2)
                print(onedata)
            stock_data_tmp = np.array([np.hstack((onedata,onedata2))])
            teacher_data_tmp_tmp = data.ix[stock_code, teacher_name]
            if teacher_data_tmp_tmp[0] > 0:
                tmp = int(1)
            else:
                tmp = int(0)

            teacher_data_tmp = np.array([[tmp]])

            stock_data_onedata = np.append(stock_data_onedata, stock_data_tmp, axis=0)
            teacher_data_onedata = np.append(teacher_data_onedata, teacher_data_tmp, axis=0)

        return stock_data_onedata, teacher_data_onedata

    def read_stock_data(self):
        days = self.days
        i= 0
        basename = ['start', 'high', 'low', 'last','vwap']
        base_dekidakaka = ['number of sale']
        column_name = copy.copy(basename)
        dekidaka = copy.copy(base_dekidakaka)
        for i1 in range(days - 1):
            for j in basename:
                column_name.append(j + str(i1 + 1))
            for j in base_dekidakaka:
                dekidaka.append(j + str(i1 + 1))

        teacher_name = ['teacher_only_up_down']

        place = self.place
        directory = os.listdir(place)

        stock_data = np.empty((0, 6 * days), float)
        teacher_data = np.empty((0, 1), int)
        i = 0
        for name in directory:
            i += 1
            if i<self.start:
                pass
            else:
                print(name)
                os.chdir(place)
                data = pd.read_csv(name, encoding='cp932', index_col=0)
                x,y = self.read_data_oneday(data,column_name,teacher_name,dekidaka,normalization_option=True,)
                if i>= self.end:
                    break
                #
                stock_data = np.append(stock_data, x, axis=0)
                teacher_data = np.append(teacher_data, y, axis=0)
                # print(stock_data)
        today = datetime.datetime.today()
        todayname = str(today.strftime("%Y%m%d_%H%M%S"))

        np.savetxt('x_train_test_norm' + todayname +'code'+'.csv', stock_data, delimiter=',')
        np.savetxt('y_train_test_norm' + todayname +'code'+'.csv', teacher_data, delimiter=',')

        return stock_data, teacher_data
