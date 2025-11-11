import json
from collections import defaultdict
import re

def parse_version(version_str):
    """
    将版本字符串解析为可比较的元组
    例如："1.81.0" -> (1, 81, 0)
    """
    # 使用正则表达式提取版本号，处理可能的前缀或后缀
    match = re.search(r'(\d+)\.(\d+)\.(\d+)', version_str)
    if match:
        return tuple(map(int, match.groups()))
    return (0, 0, 0)  # 如果无法解析，返回最低版本

def filter_latest_version_apis(input_file, output_file):
    """
    过滤API数据，对于module::name相同的API，只保留to_version最新的条目
    
    Args:
        input_file: 输入的JSON文件路径
        output_file: 输出的JSON文件路径
    """
    # 读取API数据
    with open(input_file, 'r', encoding='utf-8') as f:
        api_data = json.load(f)
    
    print(f"原始API总数: {len(api_data)}")
    
    # 按module::name分组
    api_groups = defaultdict(list)
    for entry in api_data:
        key = f"{entry.get('module', '')}::{entry['name']}"
        api_groups[key].append(entry)
    
    # 为每个组选择最新版本的API
    latest_apis = []
    for key, entries in api_groups.items():
        if len(entries) == 1:
            # 如果组内只有一个条目，直接添加
            latest_apis.append(entries[0])
        else:
            # 找出to_version最高的条目
            latest_entry = max(entries, key=lambda x: parse_version(x.get('to_version', '0.0.0')))
            latest_apis.append(latest_entry)
            
            # 打印被过滤掉的条目信息（可选）
            filtered_entries = [e for e in entries if e != latest_entry]
            if filtered_entries:
                print(f"组 '{key}' 保留了版本 {latest_entry.get('to_version', 'N/A')}，过滤掉了 {len(filtered_entries)} 个条目:")
                for i, e in enumerate(filtered_entries, 1):
                    print(f"  {i}. 版本: {e.get('to_version', 'N/A')}")
    
    # 写入结果到输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(latest_apis, f, indent=2, ensure_ascii=False)
    
    print(f"\n过滤后的API总数: {len(latest_apis)}")
    print(f"已保存到: {output_file}")

if __name__ == "__main__":
    # 替换为你的文件路径
    input_file = './reports/rust_api_changes_20250308_175536.json'
    output_file = './reports/rust_api_changes_latest_version.json'
    
    filter_latest_version_apis(input_file, output_file)