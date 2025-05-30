import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
# 混淆矩阵数据
# confusion_matrix1 = np.array([[2970, 252, 84],
#                               [317, 2669, 249],
#                               [81, 298, 2904]])
# confusion_matrix2 = np.array([[3052, 305, 100],
#                               [249, 2680, 225],
#                               [67, 234, 2912]])
# confusion_matrix3 = np.array([[3050, 260, 76],
#                               [263, 2728, 213],
#                               [55, 231, 2948]])

confusion_matrix3 = np.array([[706, 88],[136, 1196]])
# 标签
labels = ['Class 0', 'Class 1', 'Class 2']

# 绘制热力图
plt.figure()


# plt.subplot(2,2,1)
# sns.heatmap(confusion_matrix1, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')
# plt.title('Confusion Matrix (by K-BERT)')
#
#
# plt.subplot(2,2,2)
# sns.heatmap(confusion_matrix2, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')
# plt.title('Confusion Matrix (by ESIM)')
#
# plt.subplot(2,2,3)
sns.heatmap(confusion_matrix3, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (by KB-ESIM)')



plt.show()
