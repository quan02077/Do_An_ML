import numpy as np

# lay thu 4 khach hang lam du lieu mau
x = np.array([
    [42, 0],       # khach 1: 42 tuoi, so du 0
    [28, 83807],   # khach 2: 28 tuoi, so du 83807
    [55, 159660],  # khach 3: 55 tuoi, so du 159660
    [35, 0]        # khach 4: 35 tuoi, so du 0
])
y = np.array([1, 0, 1, 0])  # 1 la roi bo, 0 la o lai

print("du lieu mau 4 khach:")
print("x la:")
print(x)
print("nhan y la:")
print(y)

# 1. doan tinh entropy ban dau
print("\nbuoc tinh entropy:")
p1 = np.mean(y)
print("ti le nguoi roi bo p1:", p1)
p0 = 1 - p1
print("ti le nguoi o lai p0:", p0)

e = 0.0
if p0 > 0:
    e = e - p0 * np.log2(p0)
if p1 > 0:
    e = e - p1 * np.log2(p1)
print("entropy ban dau tinh dc la:", e)

# 2. doan cat thuoc tinh va nguong
print("\nbuoc thu cat cot tuoi:")
col = x[:, 0]
print("gia tri cot tuoi:", col)

thresholds = np.unique(col)
print("cac nguong duy nhat:", thresholds)

thresh = 40
print("gia su thu cat tai moc:", thresh)

left_idx = col <= thresh
right_idx = col > thresh
print("mang boolen sang trai:", left_idx)
print("mang boolen sang phai:", right_idx)

# 3. doan loc nhan bang mat na
y_left = y[left_idx]
y_right = y[right_idx]
print("nhan loc dc ben trai:", y_left)
print("nhan loc dc ben phai:", y_right)

# 4. tinh entropy 2 ben
def tinh_entropy(nhan):
    if len(nhan) == 0:
        return 0
    p_1 = np.mean(nhan)
    p_0 = 1 - p_1
    ent = 0.0
    if p_0 > 0:
        ent = ent - p_0 * np.log2(p_0)
    if p_1 > 0:
        ent = ent - p_1 * np.log2(p_1)
    return ent

e_left = tinh_entropy(y_left)
e_right = tinh_entropy(y_right)
print("entropy ben trai:", e_left)
print("entropy ben phai:", e_right)

# 5. tinh information gain
w_left = len(y_left) / len(y)
w_right = len(y_right) / len(y)
e_split = w_left * e_left + w_right * e_right
print("entropy trung binh sau chia:", e_split)

gain = e - e_split
print("do loi thong tin gain:", gain)

# 6. tao nut va du doan thu
print("\ntao nut quyet dinh:")
best_feat = 0
best_thresh = 40
print("chon cau hoi cot", best_feat, "voi nguong", best_thresh)

print("\nthu doan cho 1 khach moi:")
khach_moi = np.array([32, 50000])
print("khach moi 32 tuoi:")
if khach_moi[best_feat] <= best_thresh:
    print("tuoi 32 <= 40 nen re trai")
    print("du doan nhan la: 0 (o lai)")
else:
    print("tuoi > 40 nen re phai")
    print("du doan nhan la: 1 (roi bo)")

