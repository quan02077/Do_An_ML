import numpy as np
import pandas as pd

# 1. lay du lieu thuc te tu file csv
df = pd.read_csv("Bank Customer Churn Prediction.csv")
mau = df.loc[[3, 9, 0, 2, 10, 35, 46], ["age", "balance", "churn"]].copy()

print("buoc 1: du lieu dau vao")
print(mau.to_string())

x = mau[["age", "balance"]].values
y = mau["churn"].values

print("\nma tran dac trung x:")
print(x)
print("mang nhan y:")
print(y)


# 2. tinh entropy goc cua nut cha ban dau
print("\nbuoc 2: tinh entropy cua nut cha")
so_nguoi_roi_bo = np.sum(y == 1)
so_nguoi_o_lai = np.sum(y == 0)
tong_so_nguoi = len(y)

print("so nguoi roi bo (1):", so_nguoi_roi_bo)
print("so nguoi o lai (0):", so_nguoi_o_lai)
print("tong so nguoi:", tong_so_nguoi)

p1 = so_nguoi_roi_bo / tong_so_nguoi
p0 = so_nguoi_o_lai / tong_so_nguoi
print(f"ti le p1 = {so_nguoi_roi_bo} / {tong_so_nguoi} = {p1}")
print(f"ti le p0 = {so_nguoi_o_lai} / {tong_so_nguoi} = {p0}")

# the so vao cong thuc shannon
entropy_cha = - (p0 * np.log2(p0) + p1 * np.log2(p1))
print(f"the so tinh entropy cha: - ({p0} * log2({p0}) + {p1} * log2({p1})) = {entropy_cha}")


# 3. thu cat theo cot tuoi tai nguong 40
print("\nbuoc 3: thu chia tai nguong tuoi <= 40")
cot_tuoi = x[:, 0]
print("cot tuoi cua 4 khach:", cot_tuoi)

nguong = 40
print("thu cat voi nguong =", nguong)

left_idx = cot_tuoi <= nguong
right_idx = cot_tuoi > nguong
print(f"so sanh {cot_tuoi[0]} <= 40 (dung), {cot_tuoi[1]} <= 40 (dung), {cot_tuoi[2]} <= 40 (sai), {cot_tuoi[3]} <= 40 (sai)")
print("mang boolean sang trai (tuoi <= 40):", left_idx)
print("mang boolean sang phai (tuoi > 40):", right_idx)


# 4. loc nhan sang 2 nhanh con
print("\nbuoc 4: loc nhan sang 2 nhanh con bang boolean mask")
y_trai = y[left_idx]
y_phai = y[right_idx]
print("nhan nhanh trai y[left_idx]:", y_trai)
print("nhan nhanh phai y[right_idx]:", y_phai)


# 5. tinh entropy cua tung nhanh con (the so truc tiep)
print("\nbuoc 5: tinh entropy cua tung nhanh con")

# tinh cho nhanh trai
so_1_trai = np.sum(y_trai == 1)
so_0_trai = np.sum(y_trai == 0)
n_trai = len(y_trai)
p1_trai = so_1_trai / n_trai
p0_trai = so_0_trai / n_trai
print(f"nhanh trai: p1 = {so_1_trai} / {n_trai} = {p1_trai}, p0 = {so_0_trai} / {n_trai} = {p0_trai}")

entropy_trai = 0.0
if p0_trai > 0:
    entropy_trai -= p0_trai * np.log2(p0_trai)
if p1_trai > 0:
    entropy_trai -= p1_trai * np.log2(p1_trai)
print(f"the so tinh entropy trai: - ({p0_trai} * log2({p0_trai})) = {entropy_trai}")

# tinh cho nhanh phai
so_1_phai = np.sum(y_phai == 1)
so_0_phai = np.sum(y_phai == 0)
n_phai = len(y_phai)
p1_phai = so_1_phai / n_phai
p0_phai = so_0_phai / n_phai
print(f"nhanh phai: p1 = {so_1_phai} / {n_phai} = {p1_phai}, p0 = {so_0_phai} / {n_phai} = {p0_phai}")

entropy_phai = 0.0
if p0_phai > 0:
    entropy_phai -= p0_phai * np.log2(p0_phai)
if p1_phai > 0:
    entropy_phai -= p1_phai * np.log2(p1_phai)
print(f"the so tinh entropy phai: - ({p1_phai} * log2({p1_phai})) = {entropy_phai}")


# 6. tinh trong so va information gain
print("\nbuoc 6: tinh do loi thong tin information gain")
w_trai = n_trai / tong_so_nguoi
w_phai = n_phai / tong_so_nguoi
print(f"trong so nhanh trai w_trai = {n_trai} / {tong_so_nguoi} = {w_trai}")
print(f"trong so nhanh phai w_phai = {n_phai} / {tong_so_nguoi} = {w_phai}")

entropy_split = w_trai * entropy_trai + w_phai * entropy_phai
print(f"the so tinh entropy trung binh e_split: {w_trai} * {entropy_trai} + {w_phai} * {entropy_phai} = {entropy_split}")

gain = entropy_cha - entropy_split
print(f"the so tinh gain: entropy_cha - e_split = {entropy_cha} - {entropy_split} = {gain}")


# 7. tao ket luan cho nut la bang cach bo phieu da so
print("\nbuoc 7: ket luan du doan tai nut la bang bo phieu da so")

# nhanh trai: dem xem nhan nao nhieu hon
dem_0_trai = np.sum(y_trai == 0)
dem_1_trai = np.sum(y_trai == 1)
if dem_0_trai >= dem_1_trai:
    leaf_trai = 0
else:
    leaf_trai = 1
print(f"nhanh trai: co {dem_0_trai} nguoi nhan 0 va {dem_1_trai} nguoi nhan 1 -> gia tri nut la = {leaf_trai} (o lai)")

# nhanh phai: dem xem nhan nao nhieu hon
dem_0_phai = np.sum(y_phai == 0)
dem_1_phai = np.sum(y_phai == 1)
if dem_1_phai >= dem_0_phai:
    leaf_phai = 1
else:
    leaf_phai = 0
print(f"nhanh phai: co {dem_0_phai} nguoi nhan 0 va {dem_1_phai} nguoi nhan 1 -> gia tri nut la = {leaf_phai} (roi bo)")


# 8. du doan cho mot khach hang moi
print("\nbuoc 8: thu du doan cho 1 khach hang moi")
khach_moi = np.array([35, 50000])
print("du lieu khach moi:", khach_moi)
print(f"kiem tra dieu kien: tuoi khach {khach_moi[0]} <= {nguong}")

if khach_moi[0] <= nguong:
    du_doan = leaf_trai
    print(f"vi {khach_moi[0]} <= {nguong} la dung -> re trai")
    print("ket qua du doan:", du_doan, "(o lai)")
else:
    du_doan = leaf_phai
    print(f"vi {khach_moi[0]} > {nguong} la sai -> re phai")
    print("ket qua du doan:", du_doan, "(roi bo)")
