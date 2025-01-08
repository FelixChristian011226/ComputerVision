import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# 数据路径
train_path = './data/train/'
test_path = './data/test/'

# 获取图片路径和标签
def load_images_and_labels(path):
    images = []
    labels = []
    label_map = {}
    label_idx = 0
    for label in os.listdir(path):
        label_dir = os.path.join(path, label)
        if os.path.isdir(label_dir):
            if label not in label_map:
                label_map[label] = label_idx
                label_idx += 1
            for img_name in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_name)
                if img_path.endswith(('.jpg', '.png', '.jpeg')):
                    images.append(img_path)
                    labels.append(label_map[label])
    return images, np.array(labels), label_map

# Tiny Images 特征提取
def extract_tiny_images(image_paths, size=(16, 16)):
    features = []
    for img_path in tqdm(image_paths, desc="Extracting Tiny Images"):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, size).flatten()
        # 零均值和单位长度归一化
        img_resized = img_resized - np.mean(img_resized)
        img_resized = img_resized / np.linalg.norm(img_resized)
        features.append(img_resized)
    return np.array(features)

# SIFT 特征提取
def extract_sift_features(image_paths, step_size=8, sift=None):
    if sift is None:
        sift = cv2.SIFT_create()
    descriptors = []
    for img_path in tqdm(image_paths, desc="Extracting SIFT Features"):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        # 使用密集采样
        keypoints = [cv2.KeyPoint(x, y, step_size) for y in range(0, img.shape[0], step_size)
                     for x in range(0, img.shape[1], step_size)]
        _, desc = sift.compute(img, keypoints)
        if desc is not None:
            descriptors.append(desc)
    return descriptors

# 生成视觉单词表
def build_vocabulary(descriptors_list, vocab_size=50):
    all_descriptors = np.vstack(descriptors_list)
    print("Clustering descriptors to build vocabulary...")
    kmeans = KMeans(n_clusters=vocab_size, random_state=42)
    kmeans.fit(all_descriptors)
    return kmeans

# 使用视觉单词表生成 Bag of Words 特征
def extract_bow_features(image_paths, kmeans, step_size=8, sift=None):
    if sift is None:
        sift = cv2.SIFT_create()
    features = []
    for img_path in tqdm(image_paths, desc="Extracting Bag of Words Features"):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        keypoints = [cv2.KeyPoint(x, y, step_size) for y in range(0, img.shape[0], step_size)
                     for x in range(0, img.shape[1], step_size)]
        _, desc = sift.compute(img, keypoints)
        if desc is not None:
            # 计算最近的聚类中心
            words = kmeans.predict(desc)
            bow_hist = np.bincount(words, minlength=kmeans.n_clusters)
            # 归一化直方图
            bow_hist = bow_hist / np.linalg.norm(bow_hist)
            features.append(bow_hist)
        else:
            features.append(np.zeros(kmeans.n_clusters))  # 如果没有提取到特征
    return np.array(features)

# 类别准确率计算函数
def calculate_class_accuracy(predictions, labels, label_map):
    class_accuracy = {}
    for label_name, label_idx in label_map.items():
        # 找出属于当前类别的样本索引
        indices = np.where(labels == label_idx)
        # 计算该类别的准确率
        correct_predictions = np.sum(predictions[indices] == labels[indices])
        total_samples = len(indices[0])
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0
        class_accuracy[label_name] = accuracy
    return class_accuracy

# 输出类别准确率
def print_class_accuracies(class_accuracies):
    print("\nClass-wise Accuracy:")
    for label_name, accuracy in class_accuracies.items():
        print(f"  {label_name}: {accuracy * 100:.2f}%")
    mean_accuracy = np.mean(list(class_accuracies.values()))
    print(f"\nMean Accuracy Across All Classes: {mean_accuracy * 100:.2f}%")

# 加载数据
train_images, train_labels, label_map = load_images_and_labels(train_path)
test_images, test_labels, _ = load_images_and_labels(test_path)

# Tiny Image 方法
print("Running Tiny Images + Nearest Neighbor...")
tiny_train_features = extract_tiny_images(train_images)
tiny_test_features = extract_tiny_images(test_images)

knn_tiny = KNeighborsClassifier(n_neighbors=1)
knn_tiny.fit(tiny_train_features, train_labels)
tiny_predictions = knn_tiny.predict(tiny_test_features)

# 按类别计算 Tiny Images 的准确率
tiny_class_accuracies = calculate_class_accuracy(tiny_predictions, test_labels, label_map)
print("\nTiny Images Class-wise Accuracy:")
print_class_accuracies(tiny_class_accuracies)

# Bag of SIFT 方法
print("Running Bag of SIFT + Nearest Neighbor...")
sift = cv2.SIFT_create()
train_descriptors = extract_sift_features(train_images, sift=sift)
test_descriptors = extract_sift_features(test_images, sift=sift)

# 构建视觉单词表
vocab_size = 50  # 可调整
kmeans = build_vocabulary(train_descriptors, vocab_size)

# 提取 Bag of Words 特征
bow_train_features = extract_bow_features(train_images, kmeans, sift=sift)
bow_test_features = extract_bow_features(test_images, kmeans, sift=sift)

knn_bow = KNeighborsClassifier(n_neighbors=1)
knn_bow.fit(bow_train_features, train_labels)
bow_predictions = knn_bow.predict(bow_test_features)

# 按类别计算 Bag of SIFT 的准确率
bow_class_accuracies = calculate_class_accuracy(bow_predictions, test_labels, label_map)
print("\nBag of SIFT Class-wise Accuracy:")
print_class_accuracies(bow_class_accuracies)