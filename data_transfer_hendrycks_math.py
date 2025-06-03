import json
import os
from tqdm import tqdm

# 建立資料夾
base_dir = "datasets/hendrycks_math"
llm_dir = os.path.join(base_dir, "llm")
os.makedirs(llm_dir, exist_ok=True)

train_data_paths = os.listdir(os.path.join(base_dir, "train"))
train_data = []
for file in train_data_paths:
    with open(os.path.join(base_dir, "train", file), "r", encoding="utf-8") as f:
        data = json.load(f)
        for item in data:
            train_data.append({
                "input": item["input"],
                "process": item.get("process", ""),
                "label": item["label"]
            })

test_data_paths = os.listdir(os.path.join(base_dir, "test"))
test_data = []
for file in test_data_paths:
    with open(os.path.join(base_dir, "test", file), "r", encoding="utf-8") as f:
        data = json.load(f)
        for item in data:
            test_data.append({
                "input": item["input"],
                "process": item.get("process", ""),
                "label": item["label"]
            })
# 儲存訓練資料
with open(os.path.join(base_dir, "algebra_train.json"), "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)
# 儲存測試資料
with open(os.path.join(base_dir, "algebra_test.json"), "w", encoding="utf-8") as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)

# 設定分割大小
CHUNK_SIZE = 500

# 提取 CoT 推理文本
def extract_cot_text(item):
    return item.get("process") # + "\nThe answer is " + item.get("label", "")

# 儲存 CoT 分割檔案
def save_chunked_cot(cot_list, prefix):
    for i in range(0, len(cot_list), CHUNK_SIZE):
        chunk = cot_list[i:i + CHUNK_SIZE]
        chunk_file = os.path.join(llm_dir, f"{prefix}_CoT_{i // CHUNK_SIZE}.json")
        with open(chunk_file, "w", encoding="utf-8") as f:
            json.dump(chunk, f, ensure_ascii=False, indent=2)

# 處理訓練資料
cot_train = []
print("Processing training data...")
for item in tqdm(train_data, desc="Train"):
    cot = extract_cot_text(item)
    if cot:
        cot_train.append(cot)
save_chunked_cot(cot_train, "train")

# 處理測試資料
cot_test = []
print("Processing test data...")
for item in tqdm(test_data, desc="Test"):
    cot = extract_cot_text(item)
    if cot:
        cot_test.append(cot)
save_chunked_cot(cot_test, "test")