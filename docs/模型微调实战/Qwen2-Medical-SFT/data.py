from modelscope.msdatasets import MsDataset
import json
import random

# 设置随机种子以确保可重复性
random.seed(42)

# 加载数据集
ds = MsDataset.load('krisfu/delicate_medical_r1_data', subset_name='default', split='train')

# 将数据集转换为列表
data_list = list(ds)

# 随机打乱数据
random.shuffle(data_list)

# 计算分割点
split_idx = int(len(data_list) * 0.9)

# 分割数据
train_data = data_list[:split_idx]
val_data = data_list[split_idx:]

# 保存训练集
with open('train.jsonl', 'w', encoding='utf-8') as f:
    for item in train_data:
        json.dump(item, f, ensure_ascii=False)
        f.write('\n')

# 保存验证集
with open('val.jsonl', 'w', encoding='utf-8') as f:
    for item in val_data:
        json.dump(item, f, ensure_ascii=False)
        f.write('\n')

print(f"数据集已分割完成：")
print(f"训练集大小：{len(train_data)}")
print(f"验证集大小：{len(val_data)}")