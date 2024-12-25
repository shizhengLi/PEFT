import os
import random
import json

# 原始 JSON 文件路径
input_dir = "/home/lishizheng/code/peft_study/open-instruct/data/tulu-3-sft-mixture-json"
output_dir = "/home/lishizheng/code/peft_study/open-instruct/data/tulu-3-sft-mixture-json-sampled"

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)
random.seed(42)


# 采样函数
def sample_json(input_file, output_file, num_samples):
    # 读取原始 JSON 数据
    with open(input_file, "r") as f:
        data = [json.loads(line) for line in f]

    # 随机采样
    sampled_data = random.sample(data, num_samples)

    # 保存采样后的数据
    with open(output_file, "w") as f:
        for item in sampled_data:
            f.write(json.dumps(item) + "\n")

# 遍历所有划分 (train, validation 等)
for split in os.listdir(input_dir):
    input_file = os.path.join(input_dir, split)
    base_name = os.path.splitext(split)[0]  # 获取文件名（不带扩展名）

    # 采样 90k 数据
    sample_json(
        input_file=input_file,
        output_file=os.path.join(output_dir, f"{base_name}_sampled_90k.json"),
        num_samples=90000,
    )

    # 采样 9k 数据
    sample_json(
        input_file=input_file,
        output_file=os.path.join(output_dir, f"{base_name}_sampled_9k.json"),
        num_samples=9000,
    )

print("采样完成！")
