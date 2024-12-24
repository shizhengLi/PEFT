from datasets import load_dataset

# 加载数据集
dataset = load_dataset("allenai/tulu-3-sft-mixture")

# 指定保存路径
output_dir = "/home/lishizheng/code/peft_study/open-instruct/data/tulu-3-sft-mixture-json"

# 将数据集转换为 JSON 格式
for split in dataset.keys():  # 处理所有划分 (train, validation 等)
    dataset[split].to_json(f"{output_dir}/{split}.json", orient="records", lines=True)
