
# coding: utf-8

# In[1]:

import numpy as np
import time
import importlib

import iris10snn as irdata
import DefferEv as ev

importlib.reload(irdata)
importlib.reload(ev)

# #### クラスの記述

# In[8]:

class NNpulse(object):

    def __init__(self, DE_num, All_clock, Normalization_max):
        self.DE_num = DE_num
        self.All_clock = All_clock
        self.Normalization_max = Normalization_max

    def set_init(self, input_neuron, hidden_neuron_1, output_neuron):
        self.X0num = input_neuron
        self.X1num = hidden_neuron_1
        self.X2num = output_neuron
        self.pre_Y1 = [[0.0] * hidden_neuron_1] * self.DE_num
        self.out_Y1 = [[0] * hidden_neuron_1] * self.DE_num
        self.volt_Y1 = [[0.0] * hidden_neuron_1] * self.DE_num
        self.pre_Y2 = [[0.0] * output_neuron] * self.DE_num
        self.out_Y2 = [[0] * output_neuron] * self.DE_num
        self.volt_Y2 = [[0.0] * output_neuron] * self.DE_num
        if output_neuron == 1:
            self.pre_Y2 = [0.0 for a1 in range(self.DE_num)]
            self.out_Y2 = [0 for a1 in range(self.DE_num)]
            self.volt_Y2 = [0.0 for a1 in range(self.DE_num)]
        return None

    def input_decision(self, input_init_data, teach_init_data, input_train, teach_train):
        self.input_init_data = input_init_data#意味ある？
        self.teach_init_data = teach_init_data#意味ある？
        self.input_train = input_train
        self.teach_train = teach_train

    def clock(self, Y1, Y2, Y1_siki, Y2_siki, X, DE_SUM, out_clock):
        if out_clock == 0:
            #内部電位計算
            self.pre_Y1[DE_SUM] += np.dot(X, Y1[DE_SUM]) #+=にすることでfloat型にする 参照渡しになっていないか11/25
            self.volt_Y1[DE_SUM] = self.pre_Y1[DE_SUM] - Y1_siki[DE_SUM]
            #発火判定
            self.out_Y1[DE_SUM] = list(map(hebside, self.volt_Y1[DE_SUM]))#mapの返り値がリストを返さないためリストに直す
            self.pre_Y1[DE_SUM] = decay_Vol(self.out_Y1[DE_SUM], self.pre_Y1[DE_SUM], self.X1num)
            #print(self.pre_Y1)
            return self.out_Y2[DE_SUM]

        elif out_clock >= 1:
            #内部電位計算
            self.pre_Y1[DE_SUM] += np.dot(X, Y1[DE_SUM])
            self.volt_Y1[DE_SUM] = self.pre_Y1[DE_SUM] - Y1_siki[DE_SUM]
            self.pre_Y2[DE_SUM] += np.dot(self.out_Y1[DE_SUM] , Y2[DE_SUM])
            self.volt_Y2[DE_SUM] = self.pre_Y2[DE_SUM] - Y2_siki[DE_SUM]
            #発火判定
            self.out_Y1[DE_SUM] = list(map(hebside, self.volt_Y1[DE_SUM]))#mapの返り値がリストを返さないためリストに直す
            self.pre_Y1[DE_SUM] = decay_Vol(self.out_Y1[DE_SUM], self.pre_Y1[DE_SUM], self.X1num)
            #print(self.pre_Y1)
            if self.X2num ==1:#出力層によって分岐
                self.out_Y2[DE_SUM] = hebside(self.volt_Y2[DE_SUM])
            else:
                self.out_Y2[DE_SUM] = list(map(hebside, self.volt_Y2[DE_SUM]))

            self.pre_Y2[DE_SUM] = decay_Vol(self.out_Y2[DE_SUM], self.pre_Y2[DE_SUM], self.X2num)
            return self.out_Y2[DE_SUM]

    def clock_test(self, Y1, Y2, Y1_siki, Y2_siki, X, out_clock):
        if out_clock == 0:
            #初期化をここでやってしまう
            self.pre_Y1 = [0.0] * self.X1num
            self.out_Y1 = [0] * self.X1num
            self.volt_Y1 = [0.0] * self.X1num
            self.pre_Y2 = [0.0] * self.X2num
            self.out_Y2 = [0] * self.X2num
            self.volt_Y2 = [0.0] * self.X2num
            if self.X2num == 1:
                self.pre_Y2, self.out_Y2, self.volt_Y2 = 0.0, 0, 0.0


            #内部電位計算
            self.pre_Y1 += np.dot(X, Y1) #+=にすることでfloat型にする
            self.volt_Y1 = self.pre_Y1 - Y1_siki
            #発火判定
            self.out_Y1 = list(map(hebside, self.volt_Y1))#mapの返り値がリストを返さないためリストに直す
            self.pre_Y1 = decay_Vol(self.out_Y1, self.pre_Y1, self.X1num)
            #print(self.pre_Y1)
            return self.out_Y2

        elif out_clock >= 1:
            #内部電位計算
            self.pre_Y1 += np.dot(X, Y1)
            self.volt_Y1 = self.pre_Y1 - Y1_siki
            self.pre_Y2 += np.dot(self.out_Y1 , Y2)
            self.volt_Y2 = self.pre_Y2 - Y2_siki
            #発火判定
            self.out_Y1 = list(map(hebside, self.volt_Y1))#mapの返り値がリストを返さないためリストに直す
            self.pre_Y1 = decay_Vol(self.out_Y1, self.pre_Y1, self.X1num)
            #print(self.pre_Y1)
            if self.X2num ==1:#出力層によって分岐
                self.out_Y2 = hebside(self.volt_Y2)
            else:
                self.out_Y2 = list(map(hebside, self.volt_Y2))

            self.pre_Y2 = decay_Vol(self.out_Y2, self.pre_Y2, self.X2num)
            return self.out_Y2


    #ここから順伝搬
    def Forward(self, inputdata, Y1, Y2, Y1_siki, Y2_siki):
        #初期化
        Stational_Inspect = 0
        f_k_single = np.array([0 for a1 in range(self.DE_num)])
        f_k = np.array([np.array([0] * self.X2num)] * self.DE_num) #パターン分の出力箱
        F_k = [[[0 for a1 in range(self.X2num)] for a2 in range(len(inputdata))] for a3 in range(self.DE_num)] #内包表記を使わないと参照渡しになる
        F_k_single = [[0 for a1 in range(len(inputdata))] for a2 in range(self.DE_num)]
        #記述
        for num_DE in range(self.DE_num):
            out_clock = 0 #0~150*32 1個体につき1回リセット
            self.set_init(self.X0num, self.X1num, self.X2num)
            for data_number in range(len(inputdata)):#irisデータでは150回繰り返す
                for clock_now in range(self.All_clock):
                    if clock_now < self.Normalization_max:#とりあえず定数そのままでおく(32)
                        X = inputdata[data_number][clock_now]#注意：ALLclockが入力幅より増えたときなにを入力するか
                    else:
                        if self.X2num == 1:
                            X = 0
                        else:
                            X = np.array([np.array([0]) * self.X2num])

                    Stational_Inspect += 1
                    if Stational_Inspect <= 1:#過渡クロック
                        self.clock(Y1, Y2, Y1_siki, Y2_siki, X, num_DE, out_clock)
                    else:
                        if self.X2num == 1:#出力層ニューロンが１のとき
                            f_k_single[num_DE] += self.clock(Y1, Y2, Y1_siki, Y2_siki, X, num_DE, out_clock)
                        else: #出力層ニューロンが1以外のとき
                            f_k[num_DE] += self.clock(Y1, Y2, Y1_siki, Y2_siki, X, num_DE, out_clock)
                    out_clock += 1
                else:#(forelse)3つのニューロンのうち数が最も大きいものを1とする（内包表記）
                    if self.X2num == 1:
                        F_k_single[num_DE][data_number] = f_k_single[num_DE]
                        f_k_single[num_DE] = 0
                    else:# とりあえずiris用だけ記述しているが・・・汎用性はないかも
                        F_k[num_DE][data_number] = [1 if num==max(f_k[num_DE]) else 0 for num in f_k[num_DE]]#数値が同じ時どうしよう
                        f_k[num_DE] = 0
                    Stational_Inspect = 0
            else:#(forelse)
                out_clock = 0
        return F_k_single

    #学習過程
    def DE_epoc(self, roop_num, epoc_num):
        success_num = 0
        Success_Gene_list = []
        #最後に表示する用
        Error_min = []
        Error_s_min = []
        for roop_number in range(roop_num):#epoc_num=試行回数
            Y1, Y2, Y1_siki, Y2_siki = set_initial(self.X0num, self.X1num, self.X2num, self.DE_num)
            EPOC = epoc_num
            #成功率関係の変数
            ZeroNum = 0
            Success_Gene = 0
            for epoc in range(EPOC):
                Para_flat = ev.DE_conv(Y1, Y2, Y1_siki, Y2_siki, self.DE_num)
                Para_flat_vari = ev.DE_vari(Para_flat, self.DE_num)
                Para_flat_cross = ev.DE_test(Para_flat, Para_flat_vari, self.X0num, self.X1num, self.X2num, self.DE_num)
                Y1_test, Y2_test, Y1_siki_test, Y2_siki_test = ev.repair(Para_flat_cross, self.X0num, self.X1num, self.X2num, self.DE_num)
                #print(Y1_test, "Y1_test\n", Y2_test, "Y2_test\n", Y1_siki_test, "Y1_siki_test\n", Y2_siki_test)
                F_k = self.Forward(self.input_train, Y1, Y2 ,Y1_siki, Y2_siki)
                F_k_test = self.Forward(self.input_train, Y1_test, Y2_test, Y1_siki_test, Y2_siki_test)

                Error = ev.Calc_fitness(F_k, self.teach_train, self.X2num, self.DE_num)
                Error_test = ev.Calc_fitness(F_k_test, self.teach_train, self.X2num, self.DE_num)
                #print("epoc",epoc)

                for n in range(self.DE_num):
                    if Error_test[n] < Error[n]:
                        #print(n, "個体で両個体発見。", Error[n], "は", Error_test[n], "に変換されます")
                        Error[n] = Error_test[n]
                        Y1[n], Y2[n], Y1_siki[n], Y2_siki[n] = Y1_test[n][:], Y2_test[n][:], Y1_siki_test[n][:], Y2_siki_test[n]# もしかしたら参照渡し[:]をつけると治るかも
                        if Error[n] == 0:
                            ZeroNum = n
                            Success_Gene = epoc
                if epoc%(EPOC/100) == 0 or epoc == EPOC-1:
                    print(roop_number,"ループ目解析中、進行率", epoc*100/EPOC, "%")
                    print("ただいまの適応度", Error)
                if Error[ZeroNum] == 0:
                    print(ZeroNum, "個体の学習が成功しました。このときのパラメータ", "Y1", Y1[ZeroNum], "Y2", Y2[ZeroNum], "Y1_siki", Y1_siki[ZeroNum], "Y2_siki", Y2_siki[ZeroNum])
                    Success_Gene_list.append(Success_Gene)
                    success_num += 1
                    break
            #ここで汎化性能を検証する
            '''sin波の学習ではここ以降はいらない
            F_k_simu = Forward(input_s_data_int, Y1, Y2 ,Y1_siki, Y2_siki)
            Error_s = ev.Calc_fitness(F_k_simu, teach_s_data, self.X2num, self.DE_num)
            print("第", roop_number,"試行目の汎化性能検証結果", Error_s)
            minE = np.min(Error)
            Error_min.append(minE)#数個体の中で一番良い誤差関数
            #その個体の汎化性能を検証する
            minE_num = Error.index(minE)
            Error_s_min.append(Error_s[minE_num])
        Success_Gene_sum = np.sum(Success_Gene_list)
        print("学習成功率:", success_num * (100/roop_num))
        print("平均収束世代数:", Success_Gene_sum / success_num)
        print("最小誤差関数学習データ", Error_min, "テストデータ", Error_s_min)
        sumError_min = (sum(Error_s_min)/(len(Error_s_min)))
        print("テストデータ合計", sumError_min)
        '''

        #学習完了したネットワークを返す
        return Y1, Y2, Y1_siki, Y2_siki


# #### 常用関数
# In[4]:

#ヘビサイド関数の記述
def hebside(a):
    #print(a)
    if a >= 0:
        return 1
    else:
        return 0

#減衰率を適用
def decay_Vol(list1, list2, num1):
    if num1 == 1:#出力層ニューロン次第
        if list1 == 0:
            list2 = 0.9 * list2
        else:
            list2 = 0.0

    else:
        for n in range(num1):
            if list1[n]==0:
                list2[n] = 0.9 * list2[n] #intで小数点切捨て
            else:
                list2[n] = 0.0
    return list2


# #### 重み閾値初期化関数

# In[7]:

def set_initial(X0num, X1num, X2num, DE_num):
    Y1 = np.array([[[0 for a1 in range(X1num)] for a2 in range(X0num)] for a3 in range(DE_num)])
    Y1_siki = np.array([[0 for a1 in range(X1num)] for a2 in range(DE_num)])
    Y2 = np.array([[[0 for a1 in range(X2num)] for a2 in range(X1num)] for a3 in range(DE_num)])
    Y2_siki = np.array([[0 for a1 in range(X2num)] for a2 in range(DE_num)])
    for sent in range(DE_num):
        Y1[sent] = np.random.randint(-15,16,(X0num, X1num))
        Y1_siki[sent] = np.random.randint(-15,16,(X1num))
        Y2[sent] = np.random.randint(-15,16,(X1num, X2num))
        Y2_siki[sent] = np.random.randint(-15,16,(X2num))

    #入力値によって
    if X0num ==1:
        Y1 = Y1.reshape([-1, X1num])
    if X2num ==1:
        Y2 = Y2.reshape([-1, X1num])
        Y2_siki = Y2_siki.reshape([-1])

    return Y1, Y2, Y1_siki, Y2_siki
