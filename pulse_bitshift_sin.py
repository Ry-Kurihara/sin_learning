import numpy as np


def rate_simple(datanum, normalization_max, input_list):#出力層ニューロンが1つ用
    input_data_int = []
    for n in range(datanum):
        input_data_int_0 = []
        for input1 in range(normalization_max):#32（5bit）
            #1ニューロン目
            if input1 < input_list[n]:
                input_data_int_0.append(1)
            elif input1 >= input_list[n]:
                input_data_int_0.append(0)
        else:
            input_data_int.append(np.array(input_data_int_0))

    #print(input_data_int[5])
    #print(input_data_int[5][:,3])#列の取り出し方
    return input_data_int


def rate_simple_single(normalization_max, input_list):
    input_data_int = []
    for input1 in range(normalization_max):#32（5bit）
        #1ニューロン目
        if input1 < input_list:
            input_data_int.append(1)
        elif input1 >= input_list:
            input_data_int.append(0)

    return np.array(input_data_int)


#分割手法の初期配列生成用
def hagio_bitconvert(max_bit):
    Max_BIT = max_bit
    Max_NUM = pow(2, Max_BIT) - 1
    #print('最大値は', Max_NUM)
    bit_base = [0 for n in range(Max_NUM + 1)]
    bit_reverse = [0 for n in range(Max_NUM + 1)]
    bit_eval = [0 for n in range(Max_NUM + 1)]
    num_eval = [0 for n in range(Max_NUM + 1)]

    for maxnum in range(Max_NUM + 1):
        bit_base[maxnum] = bin(maxnum)[2:]#0bは抜いたもの
        for ZEROplus in range(Max_BIT - 1):#最大で5回しか足さない
            if len(bit_base[maxnum]) < Max_BIT:
                bit_base[maxnum] = '0' + bit_base[maxnum]
            else:
                pass
        bit_reverse[maxnum] = bit_base[maxnum][::-1]
        bit_eval[maxnum] = '0b' + bit_reverse[maxnum]
        num_eval[maxnum] = eval(bit_eval[maxnum])

    return num_eval


def rate_cut(datanum, normalization_max, input_list):#出力層ニューロンが1つ用
    input_data_int = []
    list_B = hagio_bitconvert(7) #とりあえず直接、Normalization_maxって書き方がいまいちかも
    print(list_B)
    for n in range(datanum):
        input_data_int_0 = []
        for input1 in range(normalization_max):#現状、Normalization_maxがあると最後のlist_Bの要素が反映されない
            #1ニューロン目
            if input_list[n] > list_B[input1]:
                input_data_int_0.append(1)
            elif input_list[n] <= list_B[input1]:
                input_data_int_0.append(0)
        else:
            input_data_int.append(np.array(input_data_int_0))

    #print(input_data_int[5])
    #print(input_data_int[5][:,3])#列の取り出し方
    return input_data_int
