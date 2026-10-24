import numpy as np


class Node:
    # 1 node trong cay: nut giua thi co cot va nguong de chia, nut la thi co ket qua du doan
    def __init__(self, feature=None, threshold=None, value=None):
        self.feature = feature      # cot thuoc tinh dung de so sanh
        self.threshold = threshold  # nguong so sanh
        self.left = None            # nhanh con ben trai (thoa man <=)
        self.right = None           # nhanh con ben phai (lon hon >)
        self.value = value          # ket qua du doan (chi co o nut la: 0 hoac 1)


class DecisionTree:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth  # do sau toi da cua cay de tranh bi hoc vet
        self.root = None            # nut goc ban dau de la none

    def entropy(self, y):
        # tinh do hon loan entropy cua tap nhan y
        if len(y) == 0:
            return 0

        p1 = np.mean(y)        # ti le nhan 1 (roi bo)
        p0 = 1 - p1            # ti le nhan 0 (o lai)
        
        e = 0.0
        # kiem tra > 0 de tranh loi tinh log2 cua 0
        if p0 > 0:
            e = e - p0 * np.log2(p0)
            
        if p1 > 0:
            e = e - p1 * np.log2(p1)
            
        return e

    def build_tree(self, X, y, depth=0):
        # ham de quy xay tung tang cua cay
        num_samples = len(y)
        
        # 1. kiem tra dieu kien dung de tao nut la
        # dung lai khi: dat do sau toi da, hoac toan bo cung 1 nhan, hoac con duoi 2 mau
        if depth >= self.max_depth or len(np.unique(y)) <= 1 or num_samples < 2:
            count_1 = np.sum(y == 1)
            count_0 = np.sum(y == 0)
            
            # nhan nao nhieu hon thi lay lam ket qua
            if count_1 >= count_0:
                leaf_value = 1
            else:
                leaf_value = 0
                
            return Node(value=leaf_value)

        # 2. tim cau hoi va nguong cat tot nhat
        best_gain = 0
        best_feat = None
        best_thresh = None
        
        current_entropy = self.entropy(y)

        # duyet qua tung cot thuoc tinh
        for feat in range(X.shape[1]):
            thresholds = np.unique(X[:, feat])
            
            # voi cot co qua nhieu gia tri lien tuc thi lay cach deu 20 moc cho chay nhanh
            if len(thresholds) > 20:
                step = len(thresholds) // 20
                thresholds = thresholds[::step]

            for thresh in thresholds:
                # tao mang boolean de tach lam 2 ben
                left_idx = X[:, feat] <= thresh
                right_idx = X[:, feat] > thresh

                # neu 1 ben bi trong rong thi bo qua
                if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
                    continue

                # tinh entropy 2 nhanh con
                e_left = self.entropy(y[left_idx])
                e_right = self.entropy(y[right_idx])
                
                # tinh trong so 2 ben
                weight_left = len(y[left_idx]) / num_samples
                weight_right = len(y[right_idx]) / num_samples
                
                # do hon loan trung binh sau khi chia
                e_split = weight_left * e_left + weight_right * e_right

                # tinh do loi thong tin information gain
                gain = current_entropy - e_split

                # neu tot hon thi luu lai ky luc
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = thresh

        # neu khong tim dc cach cat nao tot hon thi tao nut la
        if best_feat is None:
            count_1 = np.sum(y == 1)
            count_0 = np.sum(y == 0)
            
            if count_1 >= count_0:
                leaf_value = 1
            else:
                leaf_value = 0
                
            return Node(value=leaf_value)

        # 3. tao nut va goi de quy cho 2 nhanh con
        node = Node(feature=best_feat, threshold=best_thresh)
        
        left_mask = X[:, best_feat] <= best_thresh
        right_mask = X[:, best_feat] > best_thresh

        # nhanh trai thoa man dieu kien tang them 1 tang do sau
        node.left = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        # nhanh phai khong thoa man
        node.right = self.build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return node

    def fit(self, X, y):
        # ep kieu sang numpy array de khong bi loi index cua pandas
        X = np.array(X)
        y = np.array(y)
        
        # bat dau dung cay tu goc do sau 0
        self.root = self.build_tree(X, y, depth=0)

    def predict_one(self, row, node):
        # du doan 1 khach hang di tu goc xuong nut la
        if node.value is not None:
            return node.value

        # so sanh gia tri voi nguong de re trai hoac phai
        if row[node.feature] <= node.threshold:
            return self.predict_one(row, node.left)
        else:
            return self.predict_one(row, node.right)

    def predict(self, X):
        # du doan cho toan bo danh sach khach hang
        X = np.array(X)
        predictions = []
        
        for row in X:
            label = self.predict_one(row, self.root)
            predictions.append(label)
            
        return np.array(predictions)

