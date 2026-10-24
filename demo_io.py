import numpy as np

# ham tinh entropy don gian
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

# lay 4 khach hang lam du lieu mau de test
x = np.array([
    [42, 0],       # khach 1: tuoi 42, so du 0
    [28, 83807],   # khach 2: tuoi 28, so du 83807
    [55, 159660],  # khach 3: tuoi 55, so du 159660
    [35, 0]        # khach 4: tuoi 35, so du 0
])
y = np.array([1, 0, 1, 0])  # 1 la roi bo, 0 la o lai

print("1. ham entropy:")
print("input nhan vao:")
print("  y =", y)
print("xu ly tinh toan:")
p1 = np.mean(y)
p0 = 1 - p1
e = tinh_entropy(y)
print("  ti le p1 =", p1, ", p0 =", p0)
print("output tra ve:")
print("  entropy e =", e, "(kieu float)")


print("\n2. doan tim diem cat tot nhat (best split):")
print("input nhan vao:")
print("  ma tran x (4 khach, 2 cot):")
print(x)
print("  nhan y =", y)
print("xu ly tinh toan:")
# thu cat cot 0 (tuoi) voi nguong 40
col = x[:, 0]
thresh = 40
left_idx = col <= thresh
right_idx = col > thresh
print("  chia cot tuoi tai nguong 40:")
print("  mang boolean trai left_idx =", left_idx)
print("  nhanh trai y[left_idx] =", y[left_idx])
print("  nhanh phai y[right_idx] =", y[right_idx])

# cac buoc tinh toan chi tiet ra gain:
e_left = tinh_entropy(y[left_idx])
e_right = tinh_entropy(y[right_idx])
print("  entropy nhanh trai e_left =", e_left)
print("  entropy nhanh phai e_right =", e_right)

w_left = len(y[left_idx]) / len(y)
w_right = len(y[right_idx]) / len(y)
e_split = w_left * e_left + w_right * e_right
print("  entropy trung binh sau chia e_split =", e_split)

gain = e - e_split
print("  do loi thong tin gain = e - e_split = 1.0 - 0.0 =", gain)

best_feat = 0
best_thresh = 40
print("output tra ve:")
print("  cot tot nhat best_feat =", best_feat, "(cot tuoi)")
print("  nguong tot nhat best_thresh =", best_thresh, "(tuoi 40)")


print("\n3. ham build_tree:")
print("input nhan vao:")
print("  x (4 khach), y (4 nhan), depth = 0")
print("xu ly tinh toan:")
print("  chua thoa man dieu kien dung")
print("  lay best_feat = 0 va best_thresh = 40")
print("  tach lam 2 nhanh:")
print("    nhanh trai x[left_mask] (2 khach tuoi <= 40)")
print("    nhanh phai x[right_mask] (2 khach tuoi > 40)")
print("  goi de quy build_tree voi depth = 1")
print("output tra ve:")
print("  doi tuong Node chua:")
print("    node.feature = 0")
print("    node.threshold = 40")
print("    node.left = Node la (value = 0: o lai)")
print("    node.right = Node la (value = 1: roi bo)")


print("\n4. ham predict_one (du doan 1 khach):")
khach_moi = np.array([32, 50000])
print("input nhan vao:")
print("  1 dong du lieu khach hang row =", khach_moi)
print("  node goc (tuoi <= 40)")
print("xu ly tinh toan:")
print("  kiem tra 32 <= 40 la dung -> re trai")
print("  nhanh trai la nut la co value = 0")
print("output tra ve:")
print("  nhan du doan =", 0, "(khach hang se o lai, kieu int)")


print("\n5. ham predict (du doan ca tap du lieu):")
tap_test = np.array([
    [32, 50000],
    [50, 100000]
])
print("input nhan vao:")
print("  ma tran tap test x_test (2 khach hang):")
print(tap_test)
print("xu ly tinh toan:")
print("  chay predict_one cho khach 1 -> ra nhan 0")
print("  chay predict_one cho khach 2 -> ra nhan 1")
print("output tra ve:")
ket_qua = np.array([0, 1])
print("  mang du doan predictions =", ket_qua, "(shape: (2,))")
