import pandas as pd

# 读取CSV文件
df = pd.read_csv('/home/sharing/disk3/wangyifan/sujianhua/TEXTOIR_ct_adb_three/open_intent_detection/results/results_ADB.csv')

# 选择需要计算平均值的列
columns_to_average = ['F1-known', 'F1-open', 'F1', 'Acc']

# 分组计算平均值
grouped_df = df.groupby(['dataset', 'method', 'backbone', 'known_cls_ratio', 'labeled_ratio', 'loss', 'train_epochs', 'lr', 'safe1', 'safe2', 'alpha', 'alpha_loss', 'margin', 'p', 'loss_weight', 'supcon_loss_weight', 'triplet_loss_weight'])[columns_to_average].mean().reset_index()

# 保存结果到新的CSV文件
grouped_df.to_csv('averaged_results.csv', index=False)

print(grouped_df)