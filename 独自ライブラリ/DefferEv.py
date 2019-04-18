
# coding: utf-8

# In[8]:

import numpy as np


# #### 重みのサイズ変換

# In[1]:

#1*サイズにする
def DE_conv(Y1, Y2, Y1siki, Y2siki, DE_num):
    #初期化
    Para_flat = [[0] for _ in range(DE_num)]
    for i in range(DE_num):
        Para_flat[i] = []
    #配列の要素数を変換してひとまとめ
    for full in range(DE_num):
        Para_flat[full] = np.append( Y1[full].flatten(), Y2[full].flatten() )
        Para_flat[full] = np.append( Para_flat[full], Y1siki[full].flatten() )
        Para_flat[full] = np.append( Para_flat[full], Y2siki[full].flatten() )

    return Para_flat


# #### 変異点生成

# In[2]:

def DE_vari(Paraflat, DE_num):
    #変異点生成に必要な初期化
    Para_flat_vari =  [[0] for _ in range(DE_num)]
    F = 0.7
    #記述
    for vari in range(DE_num):
        q1 = 0
        q2 = 0
        q3 = 0
        while not((q1 != q2) and (q1 != q3) and(q2 != q3)):
            q1 = np.random.randint(0, DE_num)
            q2 = np.random.randint(0, DE_num)
            q3 = np.random.randint(0, DE_num)
        Para_flat_vari[vari] = Paraflat[q1] + F * (Paraflat[q2] - Paraflat[q3])
        Para_flat_vari[vari] = [int(inter) for inter in Para_flat_vari[vari]]
        Para_flat_vari[vari] = np.array(Para_flat_vari[vari])

    return Para_flat_vari


# #### 試験点生成

# In[3]:

def DE_test(Paraflat, Paraflat_vari, X0num, X1num, X2num, DE_num):
    CR = 0.2
    cross = (X0num * X1num) + (X1num * X2num) + X1num + X2num#重みしきい値パラメータ数
    Paraflat_cross = [ [0]*cross for _ in range(DE_num)] #ここの参照渡しも気になる

    for denum in range(DE_num):
        Cross_num = np.random.randint(0, cross)#0~cross-1までの整数

        Paraflat_cross[denum][Cross_num] = Paraflat_vari[denum][Cross_num]
        #最初に変えた要素より前
        for test in range(Cross_num):
            rand_para = np.random.rand()
            if rand_para <= CR:
                Paraflat_cross[denum][test] = Paraflat_vari[denum][test]
            else:
                Paraflat_cross[denum][test] = Paraflat[denum][test]
        #最初に変えた要素より後ろ
        for test in range(cross - Cross_num -1):
            rand_para = np.random.rand()
            if rand_para <= CR:
                Paraflat_cross[denum][Cross_num+test+1] = Paraflat_vari[denum][Cross_num+test+1]
            else:
                Paraflat_cross[denum][Cross_num+test+1] = Paraflat[denum][Cross_num+test+1]

    return np.array(Paraflat_cross)


# #### 配列をもとの形に戻す

# In[7]:

def repair(Para_flat, X0num, X1num, X2num, DE_num):
    #初期化
    Y1_test = np.array([[[0 for a1 in range(X1num)] for a2 in range(X0num)] for a3 in range(DE_num)])
    Y1_siki_test = np.array([[0 for a1 in range(X1num)] for a2 in range(DE_num)])
    Y2_test = np.array([[[0 for a1 in range(X2num)] for a2 in range(X1num)] for a3 in range(DE_num)])
    Y2_siki_test = np.array([[0 for a1 in range(X2num)] for a2 in range(DE_num)])
    if X0num ==1:
        Y1_test = Y1_test.reshape([-1, X1num])
    if X2num ==1:
        Y2_test = Y2_test.reshape([-1, X1num])
        Y2_siki_test = Y2_siki_test.reshape([-1])

    #次元直し
    for test in range(DE_num):
        Y2_siki_test[test] = Para_flat[test][X0num * X1num + X1num * X2num + X1num :]# ?: ?を含む
        Y1_siki_test[test] = Para_flat[test][X0num * X1num + X1num * X2num : X0num * X1num + X1num * X2num + X1num]# :? 0～(?-1)まで
        Y2_test[test] = Para_flat[test][X0num * X1num : X0num * X1num + X1num * X2num]
        Y1_test[test] = Para_flat[test][: X0num * X1num]
        #重みはさらに変換（条件次第）
        if X0num != 1:
            Y1_test[test] = np.reshape(Y1_test[test], (X0num, X1num))
        if X2num != 1:
            Y2_test[test] = np.reshape(Y2_test[test], (X1num, X2num))


    return Y1_test, Y2_test, Y1_siki_test, Y2_siki_test


# #### 適応度計算

# In[9]:

def Calc_fitness(F_k_free, teach, X2num, DE_num):
    #適応度計算に必要な初期化
    Differ = [[0]* X2num] * DE_num
    Some_Dif = [[0]* X2num] * DE_num
    Error = [[0]* X2num] * DE_num
    #記述
    for sk in range(DE_num):
        #print(sk)
        Differ[sk] = F_k_free[sk] - teach
        #print(Differ[sk])
        if X2num == 1:
            Differ[sk] = [abs(num) for num in Differ[sk]]
        else:
            Differ[sk] = [abs(num) for alist in Differ[sk] for num in alist]#絶対値にしてからパターン分ひとつの配列に入れる
        #print(Differ[sk])
        Some_Dif[sk] = np.sum(Differ[sk])
        #print(Some_Dif[sk])
        Error[sk] = Some_Dif[sk]
        #Error[sk] = (Some_Dif[sk] / (X2num * len(F_k_free[0]))) * 100.0
        #print(Error[sk])
    return Error
