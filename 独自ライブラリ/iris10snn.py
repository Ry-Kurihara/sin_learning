
# coding: utf-8

# In[4]:

from sklearn.datasets import load_iris
import numpy as np
import pandas as pd


# #### データを最大値最小値正規化、テストデータとトレーニングデータに分ける

# In[5]:

def Dataset(train,test):
    iris = load_iris()
    pd.DataFrame(iris.data, columns=iris.feature_names)
    mintrain = train/3 #　学習データをいくつにしてテストデータをいくつにするか
    mintest = test/3

    columnum1 = []
    columnum2 = []
    columnum3 = []
    columnum4 = []
    for i in range(150):
        columnum1.append(iris.data[i][0])
        columnum2.append(iris.data[i][1])
        columnum3.append(iris.data[i][2])
        columnum4.append(iris.data[i][3])
    '''
    print(columnum1)
    print("がくの長さ最大", max(columnum1), "最小", min(columnum1))
    print("がくの幅最大", max(columnum2), "最小", min(columnum2))
    print("花びらの長さ最大", max(columnum3), "最小", min(columnum3))
    print("花びらの幅最大", max(columnum4), "最小", min(columnum4))
    '''

    #ここで正規化
    M = 31
    m = 0
    for i in range(150):
        columnum1[i] = int(((columnum1[i]*10 - 43)/(79 - 43)) * (M-m) + m)
        columnum2[i] = int(((columnum2[i]*10 - 20)/(44 - 20)) * (M-m) + m)
        columnum3[i] = int(((columnum3[i]*10 - 10)/(69 - 10)) * (M-m) + m)
        columnum4[i] = int(((columnum4[i]*10 - 1)/(25 - 1)) * (M-m) + m)

    #トレーニングデータの入力値、教師値作成
    #入力値
    input_data = []
    for n in range(150):
        if 0 <= n < mintrain or 50 <= n < 50+mintrain or 100 <= n < 100+mintrain:
            input_data.append([columnum1[n], columnum2[n], columnum3[n], columnum4[n]])
    #print(input_data[5])#これが150パターンある

    #教師値
    teach_data = []
    for n in range(150):
        if 0 <= n < mintrain or 50 <= n < 50+mintrain or 100 <= n < 100+mintrain:
            if iris.target[n] == 0:
                teach_data.append([1, 0, 0])
            elif iris.target[n] == 1:
                teach_data.append([0, 1, 0])
            elif iris.target[n] == 2:
                teach_data.append([0, 0, 1])
    teach_data = np.array(teach_data)#これをしないと適応度計算のところでマイナスができなかった
    #print(teach_data[5])#これが150パターンある
    #print(columnum1)

    #教師データの入力値、教師値作成
    #入力値
    input_data_test = []
    for n in range(150):
        if 50-mintest <= n < 50 or 100-mintest <= n < 100 or 150-mintest <= n < 150:
            input_data_test.append([columnum1[n], columnum2[n], columnum3[n], columnum4[n]])
    #print(input_data_test[5])#これが150パターンある

    #教師値
    teach_data_test = []
    for n in range(150):
        if 50-mintest <= n < 50 or 100-mintest <= n < 100 or 150-mintest <= n < 150:
            if iris.target[n] == 0:
                teach_data_test.append([1, 0, 0])
            elif iris.target[n] == 1:
                teach_data_test.append([0, 1, 0])
            elif iris.target[n] == 2:
                teach_data_test.append([0, 0, 1])
    teach_data_test = np.array(teach_data_test)#これをしないと適応度計算のところでマイナスができなかった
    #print(teach_data_test[5])#これが150パターンある

    return input_data, input_data_test, teach_data, teach_data_test


# #### 入力データのビット列化（前詰め）

# In[6]:

def rate_simple(datanum, M, input_list):
    input_data_int = []
    for n in range(datanum):
        input_data_int_0 = []
        input_data_int_1 = []
        input_data_int_2 = []
        input_data_int_3 = []
        for input1 in range(M+1):#32（5bit）
            #1ニューロン目
            if input1 < input_list[n][0]:
                input_data_int_0.append(1)
            if input1 >= input_list[n][0]:
                input_data_int_0.append(0)
            #2ニューロン目
            if input1 < input_list[n][1]:
                input_data_int_1.append(1)
            if input1 >= input_list[n][1]:
                input_data_int_1.append(0)

            if input1 < input_list[n][2]:
                input_data_int_2.append(1)
            if input1 >= input_list[n][2]:
                input_data_int_2.append(0)

            if input1 < input_list[n][3]:
                input_data_int_3.append(1)
            if input1 >= input_list[n][3]:
                input_data_int_3.append(0)
        else:
            input_data_int.append(np.array([input_data_int_0, input_data_int_1, input_data_int_2, input_data_int_3]))

    #print(input_data_int[5])
    #print(input_data_int[5][:,3])#列の取り出し方
    return input_data_int


# #### 入力データのビット列化（分割）

# In[9]:

def rate_cut(datanum, M, input_list):
    list_B = [0,16, 8, 24, 4, 20, 12, 28, 2, 18, 10, 26, 6, 22, 14, 30, 1, 17, 9, 25, 5, 21, 13, 29, 3, 19, 11, 27, 7, 23, 15, 31]
    input_data_int = []
    for n in range(datanum):
        input_data_int_0 = []
        input_data_int_1 = []
        input_data_int_2 = []
        input_data_int_3 = []
        for input1 in range(M+1):#32（5bit）
            #1ニューロン目
            if input_list[n][0] > list_B[input1]:
                input_data_int_0.append(1)
            if input_list[n][0] <= list_B[input1]:
                input_data_int_0.append(0)
            #2ニューロン目
            if input_list[n][1] > list_B[input1]:
                input_data_int_1.append(1)
            if input_list[n][1] <= list_B[input1]:
                input_data_int_1.append(0)

            if input_list[n][2] > list_B[input1]:
                input_data_int_2.append(1)
            if input_list[n][2] <= list_B[input1]:
                input_data_int_2.append(0)

            if input_list[n][3] > list_B[input1]:
                input_data_int_3.append(1)
            if input_list[n][3] <= list_B[input1]:
                input_data_int_3.append(0)
        else:
            input_data_int.append(np.array([input_data_int_0, input_data_int_1, input_data_int_2, input_data_int_3]))

    #print(input_data_int[5])
    #print(input_data_int[5][:,2])#列の取り出し方
    return input_data_int


# #### 入力データのビット列化（タイミング）

# In[10]:

def timing(datanum, M, input_list):
    input_data_int = []
    for n in range(datanum):
        input_data_int_0 = []
        input_data_int_1 = []
        input_data_int_2 = []
        input_data_int_3 = []
        for input1 in range(M+1):#32（5bit）
            #1ニューロン目
            if input1 == input_list[n][0]:
                input_data_int_0.append(1)
            if input1 != input_list[n][0]:
                input_data_int_0.append(0)
            #2ニューロン目
            if input1 == input_list[n][1]:
                input_data_int_1.append(1)
            if input1 != input_list[n][1]:
                input_data_int_1.append(0)

            if input1 == input_list[n][2]:
                input_data_int_2.append(1)
            if input1 != input_list[n][2]:
                input_data_int_2.append(0)

            if input1 == input_list[n][3]:
                input_data_int_3.append(1)
            if input1 != input_list[n][3]:
                input_data_int_3.append(0)
        else:
            input_data_int.append(np.array([input_data_int_0, input_data_int_1, input_data_int_2, input_data_int_3]))

    #print(input_data_int[5])
    #print(input_data_int[5][:,3])#列の取り出し方
    return input_data_int


#2進数変換
def bitshift(datanum, timewindows, input_list):

    input_data_int = []
    for n in range(datanum):
        input_data_int_0 = []
        input_data_int_1 = []
        input_data_int_2 = []
        input_data_int_3 = []
        for timepulse in range(timewindows):#パルス幅5
            #1ニューロン目
            bitstr_0 = bin(input_list[n][0])
            lengthbit_0 = len(bitstr_0) - 2 #0bの分 最小で0b0の3
            if lengthbit_0 - timepulse > 0:
                input_data_int_0.append(int(bitstr_0[lengthbit_0 + 1 - timepulse]))# 0からはじまるので+1
            else:
                input_data_int_0.append(0)

            #2ニューロン目
            bitstr_1 = bin(input_list[n][1])
            lengthbit_1 = len(bitstr_1) - 2 #0bの分 最小で0b0の3
            if lengthbit_1 - timepulse > 0:
                input_data_int_1.append(int(bitstr_1[lengthbit_1 + 1 - timepulse]))
            else:
                input_data_int_1.append(0)

            #3ニューロン目
            bitstr_2 = bin(input_list[n][2])
            lengthbit_2 = len(bitstr_2) - 2 #0bの分 最小で0b0の3
            if lengthbit_2 - timepulse > 0:
                input_data_int_2.append(int(bitstr_2[lengthbit_2 + 1 - timepulse]))
            else:
                input_data_int_2.append(0)

            #4ニューロン目
            bitstr_3 = bin(input_list[n][3])
            lengthbit_3 = len(bitstr_3) - 2 #0bの分 最小で0b0の3
            if lengthbit_3 - timepulse > 0:
                input_data_int_3.append(int(bitstr_3[lengthbit_3 + 1 - timepulse]))
            else:
                input_data_int_3.append(0)

        else:
            input_data_int.append(np.array([input_data_int_0, input_data_int_1, input_data_int_2, input_data_int_3]))

    #print(input_data_int[5])
    #print(input_data_int[5][:,3])#列の取り出し方
    return input_data_int


#2進数→グレイコード変換
def bitgray(datanum, timewindows, input_list):
    graybit = [[[] for i in range(4)] for u in range(datanum)]
    for data in range(datanum):
        for propertys in range(4):
            graybit[data][propertys].append(input_list[data][propertys][0])
            for timestep in range(timewindows-1):
                if input_list[data][propertys][timestep] == input_list[data][propertys][timestep+1]:
                    graybit[data][propertys].append(0)
                else:
                    graybit[data][propertys].append(1)

    return graybit


#グレイコード→2進数変換
def graybit(input_list):#forで回す必要がない。Forward関数の中で使われているので
    graybit_eva = []
    for timestep in range(len(input_list)):
        if timestep == 0:
            graybit_eva.append(input_list[timestep])
        else:
            if input_list[timestep] == 1:
                graybit_eva.append(int(not(graybit_eva[timestep - 1])))
            elif input_list[timestep] == 0:
                graybit_eva.append(int(graybit_eva[timestep - 1]))

    return graybit_eva
