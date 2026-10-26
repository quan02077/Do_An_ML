import numpy as np
import pandas as pd

print("=== CHUONG TRINH BOC TACH TUNG DONG THUAT TOAN DECISION TREE ===")
print("(Chay tuan tu tung phep tinh, khong su dung bat ky ham def nao)\n")

# 1. lay du lieu thuc te tu file csv
df = pd.read_csv("Bank Customer Churn Prediction.csv")
# lay 4 khach hang thuc te: 2 nguoi o lai (dong 3, 9) va 2 nguoi roi bo (dong 0, 2)
mau = df.loc[[3, 9, 0, 2], ["age", "balance", "churn"]].copy()

print("--- BUOC 1: DU LIEU DAU VAO (INPUT) ---")
print(mau.to_string())

X = mau[["age", "balance"]].values
y = mau["churn"].values

print("\nMa tran dac trung X (cot 0 la Tuoi, cot 1 la So du):")
print(X)
print("Mang nhan y (0 la o lai, 1 la roi bo):")
print(y)


# 2. tinh entropy goc cua nut cha ban dau
print("\n--- BUOC 2: TINH ENTROPY CUA NUT CHA ---")
so_nguoi_roi_bo = np.sum(y == 1)
so_nguoi_o_lai = np.sum(y == 0)
tong_so_nguoi = len(y)

print("So nguoi roi bo (nhan 1):", so_nguoi_roi_bo)
print("So nguoi o lai (nhan 0):", so_nguoi_o_lai)
print("Tong so nguoi:", tong_so_nguoi)

p1 = so_nguoi_roi_bo / tong_so_nguoi
p0 = so_nguoi_o_lai / tong_so_nguoi
print("Ti le p1 (roi bo) =", p1)
print("Ti le p0 (o lai)  =", p0)

# cong thuc shannon: - (p0*log2(p0) + p1*log2(p1))
entropy_cha = - (p0 * np.log2(p0) + p1 * np.log2(p1))
print("Output Entropy nut cha =", entropy_cha)


# 3. thu cat theo cot tuoi tai nguong 40
print("\n--- BUOC 3: THU CHIA TAI NGUONG TUOI <= 40 ---")
cot_tuoi = X[:, 0]
print("Gia tri cot tuoi cua 4 khach:", cot_tuoi)

nguong = 40
print("Thu nguong cat =", nguong)

# tao 2 mang boolean de loc du lieu
left_idx = cot_tuoi <= nguong
right_idx = cot_tuoi > nguong
print("Output mang boolean nhanh trai (Tuoi <= 40):", left_idx)
print("Output mang boolean nhanh phai (Tuoi > 40): ", right_idx)


# 4. loc nhan sang 2 nhanh con
print("\n--- BUOC 4: LOC NHAN SANG 2 NHANH CON BANG MAT NA BOOLEAN ---")
y_trai = y[left_idx]
y_phai = y[right_idx]
print("Nhan y roi vao nhanh trai y[left_idx]:", y_trai, "(gom khach 39t va 27t)")
print("Nhan y roi vao nhanh phai y[right_idx]:", y_phai, "(gom 2 khach 42t)")


# 5. tinh entropy cua tung nhanh con (tinh truc tiep khong dung ham)
print("\n--- BUOC 5: TINH ENTROPY CUA TUNG NHANH CON ---")

# tinh cho nhanh trai:
p1_trai = np.sum(y_trai == 1) / len(y_trai)
p0_trai = np.sum(y_trai == 0) / len(y_trai)
print("Nhanh trai: ti le p1 =", p1_trai, ", ti le p0 =", p0_trai)
entropy_trai = 0.0
if p0_trai > 0:
    entropy_trai -= p0_trai * np.log2(p0_trai)
if p1_trai > 0:
    entropy_trai -= p1_trai * np.log2(p1_trai)
print("Output Entropy nhanh trai =", entropy_trai, "(thuan khiet vi toan nhan 0)")

# tinh cho nhanh phai:
p1_phai = np.sum(y_phai == 1) / len(y_phai)
p0_phai = np.sum(y_phai == 0) / len(y_phai)
print("Nhanh phai: ti le p1 =", p1_phai, ", ti le p0 =", p0_phai)
entropy_phai = 0.0
if p0_phai > 0:
    entropy_phai -= p0_phai * np.log2(p0_phai)
if p1_phai > 0:
    entropy_phai -= p1_phai * np.log2(p1_phai)
print("Output Entropy nhanh phai =", entropy_phai, "(thuan khiet vi toan nhan 1)")


# 6. tinh trong so va information gain
print("\n--- BUOC 6: TINH DO LOI THONG TIN (INFORMATION GAIN) ---")
w_trai = len(y_trai) / tong_so_nguoi
w_phai = len(y_phai) / tong_so_nguoi
print("Trong so nhanh trai w_trai =", w_trai)
print("Trong so nhanh phai w_phai =", w_phai)

entropy_split = w_trai * entropy_trai + w_phai * entropy_phai
print("Entropy trung binh co trong so sau khi chia =", entropy_split)

gain = entropy_cha - entropy_split
print("Output Information Gain = Entropy_cha - Entropy_split =", gain)
print("=> Nhan xet: Gain = 1.0 la muc cao nhat, chung to nguong 40 chia hoan hao!")


# 7. tao ket luan cho nut la
print("\n--- BUOC 7: KET LUAN DU DOAN TAI NUT LA ---")
print("Vi ca 2 nhanh deu co Entropy = 0 (da thuan khiet) nen tao 2 nut la:")
leaf_trai = 0  # da so la 0
leaf_phai = 1  # da so la 1
print("Nut la ben trai (Tuoi <= 40): ket luan du doan =", leaf_trai, "(O LAI)")
print("Nut la ben phai (Tuoi > 40):  ket luan du doan =", leaf_phai, "(ROI BO)")


# 8. du doan cho mot khach hang moi (chay truc tiep bang if/else, khong can ham)
print("\n--- BUOC 8: THU DU DOAN CHO 1 KHACH HANG MOI ---")
khach_moi = np.array([35, 50000])  # tuoi 35, so du 50.000$
print("Du lieu khach hang moi:", khach_moi)
print("Kiem tra dieu kien: Khach co Tuoi <= 40 hay khong?")

if khach_moi[0] <= 40:
    du_doan = leaf_trai
    print("Ket qua so sanh: 35 <= 40 la DUNG -> Re sang nhanh trai")
    print("Output du doan cuoi cung:", du_doan, "==> KHACH HANG SE O LAI!")
else:
    du_doan = leaf_phai
    print("Ket qua so sanh: Tuoi > 40 la SAI -> Re sang nhanh phai")
    print("Output du doan cuoi cung:", du_doan, "==> KHACH HANG SE ROI BO!")

