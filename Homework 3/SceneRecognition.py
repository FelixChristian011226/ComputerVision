import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from glob import glob
from os.path import join
from collections import defaultdict


# 数据路径
TRAIN_DATA = './data/train/'
TEST_DATA = './data/test/'

# 类别映射
label_map = {
    'coast': 0, 'forest': 1, 'highway': 2, 'insidecity': 3, 'mountain': 4,
    'office': 5, 'opencountry': 6, 'street': 7, 'suburb': 8, 'tallbuilding': 9,
    'bedroom': 10, 'industrial': 11, 'kitchen': 12, 'livingroom': 13, 'store': 14
}
inv_label_map = {v: k for k, v in label_map.items()}


# -------------------- Tiny Image Representation -------------------- #
def load_tiny_imgs(data_path, img_size=(16, 16)):
    """
    加载 tiny image 特征，将图像缩放为 16x16，返回特征和对应标签。
    """
    features = []
    labels = []
    for category in sorted(os.listdir(data_path)):
        category_path = join(data_path, category)
        if not os.path.isdir(category_path):
            continue
        for img_path in glob(join(category_path, '*.jpg')):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, img_size).astype(np.float32)
            img = (img - np.mean(img)) / np.std(img)  # 归一化
            features.append(img.flatten())
            labels.append(label_map[category.lower()])
    return np.array(features), np.array(labels)


def knn_classifier(train_features, train_labels, test_features, k=1):
    """
    KNN 分类器，基于 L2 距离计算。
    """
    predictions = []
    for test_feature in test_features:
        distances = np.sqrt(np.sum((train_features - test_feature) ** 2, axis=1))
        nearest_indices = np.argsort(distances)[:k]
        nearest_labels = train_labels[nearest_indices]
        # 投票决定类别
        predictions.append(np.argmax(np.bincount(nearest_labels)))
    return np.array(predictions)


# -------------------- Bag of SIFT Representation -------------------- #
def load_sift_features(data_path):
    """
    加载 SIFT 特征，返回每个类别的特征字典。
    """
    sift = cv2.SIFT_create()
    features = defaultdict(list)
    for category in sorted(os.listdir(data_path)):
        category_path = join(data_path, category)
        if not os.path.isdir(category_path):
            continue
        for img_path in glob(join(category_path, '*.jpg')):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            _, desc = sift.detectAndCompute(img, None)
            if desc is not None:
                features[category.lower()].extend(desc)
    return features


def build_visual_vocab(features_dict, vocab_size=50):
    """
    构建视觉词汇表，使用 KMeans 聚类。
    """
    all_features = []
    for category_features in features_dict.values():
        all_features.extend(category_features)
    all_features = np.array(all_features)
    print(f"Clustering {len(all_features)} SIFT descriptors into {vocab_size} clusters...")
    kmeans = KMeans(n_clusters=vocab_size, n_init='auto', random_state=42)
    kmeans.fit(all_features)
    return kmeans


def extract_bow_histograms(data_path, kmeans, vocab_size):
    """
    提取图像的 Bag of Words 直方图特征。
    """
    sift = cv2.SIFT_create()
    histograms = []
    labels = []
    for category in sorted(os.listdir(data_path)):
        category_path = join(data_path, category)
        if not os.path.isdir(category_path):
            continue
        for img_path in glob(join(category_path, '*.jpg')):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            _, desc = sift.detectAndCompute(img, None)
            if desc is not None:
                words = kmeans.predict(desc)
                histogram, _ = np.histogram(words, bins=np.arange(vocab_size + 1))
                histogram = histogram.astype(np.float32)
                histogram /= np.linalg.norm(histogram)  # 归一化
                histograms.append(histogram)
                labels.append(label_map[category.lower()])
    return np.array(histograms), np.array(labels)


# -------------------- Evaluation -------------------- #
def compute_accuracy(predictions, labels):
    """
    根据预测值和真实标签计算 accuracy。
    """
    correct = predictions == labels
    accuracy_per_class = {}
    for label in np.unique(labels):
        class_indices = labels == label
        accuracy_per_class[inv_label_map[label]] = np.sum(correct[class_indices]) / np.sum(class_indices)
    return accuracy_per_class, np.mean(list(accuracy_per_class.values()))


if __name__ == '__main__':
    # Tiny Image + KNN
    print("Running Tiny Image + KNN...")
    train_tiny_features, train_tiny_labels = load_tiny_imgs(TRAIN_DATA)
    test_tiny_features, test_tiny_labels = load_tiny_imgs(TEST_DATA)
    tiny_predictions = knn_classifier(train_tiny_features, train_tiny_labels, test_tiny_features)
    tiny_acc_per_class, tiny_avg_acc = compute_accuracy(tiny_predictions, test_tiny_labels)
    print("\nTiny Image Results:")
    for category, acc in tiny_acc_per_class.items():
        print(f"{category}: {acc:.4f}")
    print(f"Average Accuracy: {tiny_avg_acc:.4f}\n")

    # Bag of SIFT + KNN
    print("Running Bag of SIFT + KNN...")
    sift_features = load_sift_features(TRAIN_DATA)
    vocab_sizes = [10, 30, 50, 70, 100]  # 不同的视觉词汇表大小
    for vocab_size in vocab_sizes:
        print(f"\nVisual Vocabulary Size: {vocab_size}")
        kmeans = build_visual_vocab(sift_features, vocab_size=vocab_size)
        train_bow_features, train_bow_labels = extract_bow_histograms(TRAIN_DATA, kmeans, vocab_size)
        test_bow_features, test_bow_labels = extract_bow_histograms(TEST_DATA, kmeans, vocab_size)
        bow_predictions = knn_classifier(train_bow_features, train_bow_labels, test_bow_features)
        bow_acc_per_class, bow_avg_acc = compute_accuracy(bow_predictions, test_bow_labels)
        print("Bag of SIFT Results:")
        for category, acc in bow_acc_per_class.items():
            print(f"{category}: {acc:.4f}")
        print(f"Average Accuracy: {bow_avg_acc:.4f}")