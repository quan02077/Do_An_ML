import numpy as np
import pandas as pd

# ham tinh entropy
def tinh_entropy(nhan):
    if len(nhan) == 0:
        return 0
    p1 = np.mean(nhan)
    p0 = 1 - p1
    e = 0.0
    if p0 > 0:
        e = e - p0 * np.log2(p0)
    if p1 > 0:
        e = e - p1 * np.log2(p1)
    return e

# 1. trich xuat 8 khach hang dau tien tu file csv (lay 2 cot dac trung: so san pham va tuoi)
df = pd.read_csv("Bank Customer Churn Prediction.csv")
mau_df = df.iloc[:8][["products_number", "age", "churn"]].copy()
mau_df.insert(0, "khach", [f"K{i+1}" for i in range(8)])

print("bang 8 khach hang trich tu file csv (dung ve cay 3 tang):")
print(mau_df.to_string(index=False))
print("=" * 60)

# ma tran dac trung x: cot 0 la so san pham, cot 1 la tuoi
x = mau_df[["products_number", "age"]].values
y = mau_df["churn"].values


print("\n--- TANG 0: NUT GOC (DEPTH = 0) ---")
print("input nhan vao:")
print("  8 khach hang, nhan y =", y)
print("xu ly tinh toan:")
p1 = np.mean(y)
p0 = 1 - p1
e0 = tinh_entropy(y)
print("  co 4 nguoi o lai (0) va 4 nguoi roi bo (1) -> ti le 50/50")
print("  entropy nut goc e0 =", round(e0, 3))
print("  tim dc cau hoi tot nhat: [so san pham <= 2]")

left_0 = x[:, 0] <= 2
right_0 = x[:, 0] > 2
print("  -> nhanh phai (so sp > 2): gom K3 (3 sp) va K8 (4 sp)")
print("     nhan y =", y[right_0], "-> toan bo la 1 (thuan khiet)")
print("     => TAO NUT LA: value = 1 (ROI BO)")

print("  -> nhanh trai (so sp <= 2): gom 6 khach con lai (K1, K2, K4, K5, K6, K7)")
print("     nhan y =", y[left_0], "(chua thuan khiet: 4 nguoi 0, 2 nguoi 1)")
print("     => DE QUY TIEP XUONG TANG 1 (depth = 1)")


print("\n--- TANG 1: NHANH CON BEN TRAI (DEPTH = 1) ---")
x_1 = x[left_0]
y_1 = y[left_0]
print("input nhan vao:")
print("  6 khach hang, nhan y =", y_1)
print("xu ly tinh toan:")
e1 = tinh_entropy(y_1)
print("  entropy tai day e1 =", round(e1, 3))
print("  tim dc cau hoi tot nhat: [tuoi <= 41]")

left_1 = x_1[:, 1] <= 41
right_1 = x_1[:, 1] > 41
print("  -> nhanh trai (tuoi <= 41): gom K2 (41 tuoi) va K4 (39 tuoi)")
print("     nhan y =", y_1[left_1], "-> toan bo la 0 (thuan khiet)")
print("     => TAO NUT LA: value = 0 (O LAI)")

print("  -> nhanh phai (tuoi > 41): gom 4 khach (K1, K5, K6, K7)")
print("     nhan y =", y_1[right_1], "(chua thuan khiet: 2 nguoi 0, 2 nguoi 1)")
print("     => DE QUY TIEP XUONG TANG 2 (depth = 2)")


print("\n--- TANG 2: NHANH CON TIEP THEO (DEPTH = 2) ---")
x_2 = x_1[right_1]
y_2 = y_1[right_1]
print("input nhan vao:")
print("  4 khach hang: K1(42t), K5(43t), K6(44t), K7(50t)")
print("  nhan y =", y_2)
print("xu ly tinh toan:")
print("  tim dc cau hoi tot nhat: [tuoi <= 42]")

left_2 = x_2[:, 1] <= 42
right_2 = x_2[:, 1] > 42
print("  -> nhanh trai (tuoi <= 42): gom K1 (42 tuoi)")
print("     nhan y =", y_2[left_2], "-> toan bo la 1 (thuan khiet)")
print("     => TAO NUT LA: value = 1 (ROI BO)")

print("  -> nhanh phai (tuoi > 42): gom K5 (43t, y=0), K6 (44t, y=1), K7 (50t, y=0)")
print("     nhan y =", y_2[right_2], "(co 2 so 0 va 1 so 1)")
print("     LUC NAY: depth = 3 == max_depth (cham tran do sau toi da!)")
print("     dieu kien dung (depth >= self.max_depth) duoc kich hoat")
print("     bo phieu theo so dong: count_0 (2) > count_1 (1)")
print("     => TAO NUT LA: value = 0 (O LAI)")


print("\n" + "=" * 60)
print("SO DO CAY 3 TANG HOAN CHINH DE VE LEN BANG:")
so_do = """
                    [TANG 0: NUT GOC]
                 So san pham <= 2 ? (N=8)
                   /                 \\
           (Dung) /                   \\ (Sai)
                 v                     v
        [TANG 1: TRAI]            [NUT LA]
       Tuoi <= 41 ? (N=6)       value = 1 (ROI BO)
         /           \\            (K3, K8: > 2 sp)
 (Dung) /             \\ (Sai)
       v               v
   [NUT LA]      [TANG 2: PHAI]
  value = 0     Tuoi <= 42 ? (N=4)
   (O LAI)         /           \\
(K2, K4: <=41t)  /             \\
                v               v
            [NUT LA]        [NUT LA]
           value = 1        value = 0 (O LAI)
           (ROI BO)     (Cham max_depth=3,
        (K1: 42 tuoi)    bo phieu: K5,K6,K7)
"""
print(so_do)
print("=" * 60)
