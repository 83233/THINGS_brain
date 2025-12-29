import os
import json

def create_class_mapping(image_root_dir, output_json_path):
    class_names = sorted([
        name for name in os.listdir(image_root_dir)
        if os.path.isdir(os.path.join(image_root_dir, name))
    ])

    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

    with open(output_json_path, 'w') as f:
        json.dump(class_to_idx, f, indent=4)

    print(f"类别映射已保存到 {output_json_path}")
    return class_to_idx

def load_mapping(json_path):
    with open(json_path, 'r') as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return class_to_idx, idx_to_class

# 示例使用
if __name__ == '__main__':
    image_root = "./image/_image_database_things/images"
    output_json = "class_mapping.json"

    create_class_mapping(image_root, output_json)

    class_to_idx, idx_to_class = load_mapping(output_json)
    print("类别 -> 数字:", class_to_idx['dog'])
    print("数字 -> 类别:", idx_to_class[12])
