# sin_learning
Sin関数をSNNにより学習する

学習する関数

![sin_learning_function](https://user-images.githubusercontent.com/43668533/56569021-24c72500-65f3-11e9-958c-11eae398b8e5.png)

学習する関数のデータ点

![sin_learn_dot](https://user-images.githubusercontent.com/43668533/56688182-ffc9d380-6712-11e9-8580-112f7411ba71.png)

データ数は100データ。

入力：現在のYの値、出力：１時刻先のYの値

適応度計算（Xの刻み幅は100）
1回の入力に対して整数値として出力を出している
（出力値-教師値）*100

![sinlearnE](https://user-images.githubusercontent.com/43668533/56568783-ad919100-65f2-11e9-934d-5356264654cc.png)

E：適応度, P：学習データ数（今回は100）, tk：（k番目の教師値）, fk：（k番目の出力値）

rete_cut（数と密度によるパルス表現）でsin関数を学習

したいこと：
学習途中の出力関数の図を出す
学習後のパラメータ（重み、しきい値）を出力
