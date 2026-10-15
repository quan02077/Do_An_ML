import numpy as np


class Node:
    """
    Một Node trong Cây Quyết Định dự đoán khách hàng rời bỏ ngân hàng (Bank Churn):
    - Nếu là Nút hỏi điều kiện: lưu câu hỏi ví dụ 'Tuổi <= 40?' hoặc 'Số dư <= 50000$?'
    - Nếu là Nút lá: đưa ra quyết định cuối cùng:
        + 0: Khách hàng Ở LẠI ngân hàng.
        + 1: Khách hàng RỜI BỎ ngân hàng (Churn).
    """
    def __init__(self, feature=None, threshold=None, value=None):
        self.feature = feature      # Vị trí cột thuộc tính (ví dụ: cột 3 là age, cột 5 là balance...)
        self.threshold = threshold  # Ngưỡng số để so sánh (ví dụ: tuổi <= 40)
        self.left = None            # Nhánh con bên trái (nhóm khách hàng thỏa mãn điều kiện <= threshold)
        self.right = None           # Nhánh con bên phải (nhóm khách hàng ngược lại, > threshold)
        self.value = value          # Kết quả dự đoán: 0 (Ở lại) hoặc 1 (Rời bỏ). CHỈ CÓ ở nút lá!


class DecisionTree:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth  # Chiều sâu tối đa của cây (đặt là 5 tầng để cây không bị học vẹt / Overfitting)
        self.root = None            # Gốc của cây (Root Node), ban đầu chưa học thì bằng None

    def entropy(self, y):
        """
        Tính độ hỗn loạn thông tin (Entropy) của tập khách hàng:
        - Nếu 100% khách hàng đều ở lại (toàn nhãn 0) -> Entropy = 0 (thuần khiết).
        - Nếu 100% khách hàng đều rời bỏ (toàn nhãn 1) -> Entropy = 0 (thuần khiết).
        - Nếu 50% ở lại và 50% rời bỏ -> Entropy = 1 (hỗn loạn tối đa, khó đoán nhất).
        """
        if len(y) == 0:
            return 0

        # Tỉ lệ khách hàng rời bỏ (nhãn 1) trong nhóm hiện tại
        p1 = np.mean(y)
        
        # Tỉ lệ khách hàng ở lại (nhãn 0) trong nhóm hiện tại
        p0 = 1 - p1
        
        # Tính độ hỗn loạn theo công thức Shannon: Entropy = - p0*log2(p0) - p1*log2(p1)
        e = 0.0
        
        # Chỉ tính log2 khi tỉ lệ > 0 để tránh lỗi toán học chia cho 0
        if p0 > 0:
            e = e - p0 * np.log2(p0)
            
        if p1 > 0:
            e = e - p1 * np.log2(p1)
            
        return e

    def build_tree(self, X, y, depth=0):
        """
        Hàm đệ quy xây dựng từng tầng của cây quyết định:
        1. Kiểm tra xem đã đủ điều kiện dừng chưa.
        2. Nếu chưa dừng: Tìm xem hỏi về thuộc tính nào (Tuổi, Số dư, Thẻ tín dụng...) 
           và tại ngưỡng nào thì phân loại khách hàng tốt nhất.
        3. Tách tập khách hàng làm 2 nhóm và mọc tiếp nhánh trái, nhánh phải.
        """
        num_samples = len(y)
        
        # -------------------------------------------------------------------------
        # BƯỚC 1: KIỂM TRA ĐIỀU KIỆN DỪNG (Tạo Nút Lá)
        # -------------------------------------------------------------------------
        # Cây sẽ dừng mọc nhánh và chốt kết quả nếu:
        # - Đã chạm đến độ sâu tối đa (depth >= max_depth = 5)
        # - Hoặc toàn bộ khách hàng trong nhóm đều cùng 1 loại (đều ở lại hoặc đều rời bỏ)
        # - Hoặc số lượng khách hàng còn lại quá ít (< 2 người)
        if depth >= self.max_depth or len(np.unique(y)) <= 1 or num_samples < 2:
            count_1 = np.sum(y == 1)  # Số lượng người rời bỏ
            count_0 = np.sum(y == 0)  # Số lượng người ở lại
            
            # Bỏ phiếu theo số đông: Nếu số người rời bỏ >= số người ở lại thì dự đoán là 1, ngược lại là 0
            if count_1 >= count_0:
                leaf_value = 1
            else:
                leaf_value = 0
                
            return Node(value=leaf_value)

        # -------------------------------------------------------------------------
        # BƯỚC 2: TÌM CÂU HỎI VÀ NGƯỠNG PHÂN CHIA TỐT NHẤT (Best Split)
        # -------------------------------------------------------------------------
        best_gain = 0           # Lưu độ lợi thông tin lớn nhất tìm được
        best_feat = None        # Lưu cột tốt nhất (ví dụ: cột 'age')
        best_thresh = None      # Lưu ngưỡng tốt nhất (ví dụ: 40 tuổi)
        
        # Độ hỗn loạn ban đầu của nhóm khách hàng trước khi phân chia
        current_entropy = self.entropy(y)

        # Duyệt qua từng thuộc tính của khách hàng (credit_score, country, gender, age, balance...)
        for feat in range(X.shape[1]):
            # Lấy các giá trị khác nhau trong cột đó để làm ứng viên ngưỡng
            thresholds = np.unique(X[:, feat])
            
            # Với các cột liên tục như số dư (balance) hoặc lương (salary) có hàng ngàn số khác nhau,
            # Ta chỉ cần lấy 20 mốc đại diện để máy tính chạy nhanh trong vài giây
            if len(thresholds) > 20:
                step = len(thresholds) // 20
                thresholds = thresholds[::step]

            # Thử từng ngưỡng giá trị
            for thresh in thresholds:
                # Chia khách hàng làm 2 nhóm theo câu hỏi: "Thuộc tính <= ngưỡng đúng không?"
                left_idx = X[:, feat] <= thresh  # Nhóm thỏa mãn (sang trái)
                right_idx = X[:, feat] > thresh  # Nhóm không thỏa mãn (sang phải)

                # Nếu chia ra mà có một nhóm rỗng (tất cả khách hàng dồn hết sang 1 bên) -> Ngưỡng này vô nghĩa
                if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
                    continue

                # Tính độ hỗn loạn của nhóm bên trái và nhóm bên phải
                e_left = self.entropy(y[left_idx])
                e_right = self.entropy(y[right_idx])
                
                # Tỉ trọng số lượng khách hàng rơi vào mỗi nhóm
                weight_left = len(y[left_idx]) / num_samples
                weight_right = len(y[right_idx]) / num_samples
                
                # Độ hỗn loạn trung bình của cả 2 nhóm sau khi chia
                e_split = weight_left * e_left + weight_right * e_right

                # Information Gain (Độ lợi thông tin): Chênh lệch độ hỗn loạn trước và sau khi chia
                # Gain càng cao tức là câu hỏi này phân loại khách hàng càng rõ ràng
                gain = current_entropy - e_split

                # Nếu câu hỏi này tốt hơn kỷ lục trước đó -> Ghi nhận lại
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = thresh

        # Nếu duyệt hết mà không có câu hỏi nào làm giảm độ hỗn loạn (best_feat là None) -> Dừng và tạo nút lá
        if best_feat is None:
            count_1 = np.sum(y == 1)
            count_0 = np.sum(y == 0)
            
            if count_1 >= count_0:
                leaf_value = 1
            else:
                leaf_value = 0
                
            return Node(value=leaf_value)

        # -------------------------------------------------------------------------
        # BƯỚC 3: TẠO NÚT HỎI VÀ ĐỆ QUY CHO 2 NHÁNH CON
        # -------------------------------------------------------------------------
        # Tạo nút trung gian lưu câu hỏi tốt nhất tìm được (ví dụ: Tuổi <= 40)
        node = Node(feature=best_feat, threshold=best_thresh)
        
        # Tách tập khách hàng thành 2 nửa
        left_mask = X[:, best_feat] <= best_thresh
        right_mask = X[:, best_feat] > best_thresh

        # Đệ quy: Nhóm khách hàng thỏa mãn (<= threshold) đi sang nhánh trái và tiếp tục chia ở tầng tiếp theo (depth + 1)
        node.left = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        
        # Đệ quy: Nhóm khách hàng không thỏa mãn (> threshold) đi sang nhánh phải và tiếp tục chia ở tầng tiếp theo (depth + 1)
        node.right = self.build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return node

    def fit(self, X, y):
        """
        Bắt đầu huấn luyện mô hình cây quyết định trên tập dữ liệu ngân hàng.
        """
        # Ép kiểu dữ liệu X và nhãn y sang mảng numpy để tránh lỗi lệch index từ Pandas
        X = np.array(X)
        y = np.array(y)
        
        # Bắt đầu trồng cây từ nút gốc với độ sâu ban đầu là 0
        self.root = self.build_tree(X, y, depth=0)

    def predict_one(self, row, node):
        """
        Dự đoán cho 1 khách hàng cụ thể:
        Cầm thông tin của khách hàng đi từ nút gốc (Root), trả lời lần lượt từng câu hỏi if/else
        để rẽ trái hoặc phải cho đến khi chạm tới nút lá.
        """
        # Nếu đã chạm tới nút lá -> Đưa ra kết luận ngay: 0 (Ở lại) hoặc 1 (Rời bỏ)
        if node.value is not None:
            return node.value

        # Nếu chưa phải lá, so sánh thông tin của khách hàng với ngưỡng của nút:
        if row[node.feature] <= node.threshold:
            # Nếu thỏa mãn điều kiện -> Rẽ sang nhánh trái
            return self.predict_one(row, node.left)
        else:
            # Nếu không thỏa mãn -> Rẽ sang nhánh phải
            return self.predict_one(row, node.right)

    def predict(self, X):
        """
        Dự đoán cho TOÀN BỘ danh sách khách hàng trong tập kiểm tra X.
        """
        X = np.array(X)
        predictions = []
        
        # Lần lượt dự đoán cho từng khách hàng
        for row in X:
            label = self.predict_one(row, self.root)
            predictions.append(label)
            
        # Trả về mảng kết quả dự đoán (gồm các số 0 và 1)
        return np.array(predictions)