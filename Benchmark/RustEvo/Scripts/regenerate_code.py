"""
检查函数签名是否存在于代码中，不存在重新generate code
"""

import json
import re
import sys

def check_function_in_code(code: str, signature: str) -> bool:
    """检查签名是否存在于代码中，支持trait路径的不同表示方式。"""
    # 移除前后空白
    signature = signature.strip()
    
    # 从签名中提取函数名
    fn_name_match = re.search(r'fn\s+([a-zA-Z0-9_]+)', signature)
    if not fn_name_match:
        return False
    
    fn_name = fn_name_match.group(1)
    
    # 基本检查 - 函数名和大致结构
    basic_pattern = r'fn\s+' + re.escape(fn_name) + r'\s*\([^{]*{'
    if re.search(basic_pattern, code):
        return True
    
    # 提取函数名和参数部分（不包括泛型约束）
    fn_params_match = re.search(r'fn\s+([a-zA-Z0-9_]+)(?:<.*>)?\s*\(([^)]*)\)', signature)
    if not fn_params_match:
        return False
    
    extracted_name = fn_params_match.group(1)
    params = fn_params_match.group(2).strip()
    
    # 在代码中查找匹配相同函数名和参数的函数定义
    # 注意这里我们只匹配函数名和参数部分，忽略泛型约束
    in_code_pattern = r'fn\s+' + re.escape(extracted_name) + r'(?:<.*>)?\s*\(\s*' + re.escape(params) + r'\s*\)'
    
    return bool(re.search(in_code_pattern, code))

def main():
    # 检查命令行参数
    if len(sys.argv) != 2:
        print("使用方法: python script.py <json文件路径>")
        sys.exit(1)
    
    json_file_path = sys.argv[1]
    
    try:
        # 读取JSON文件
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # 初始化计数器
        missing_signatures_count = 0
        missing_signatures = []
        
        # 遍历数据，检查每个条目
        for index, item in enumerate(data):
            if "code" not in item or "function_signature" not in item:
                print(f"警告: 第{index+1}条数据缺少必要字段")
                continue
                
            code = item["code"]
            signature = item["function_signature"]
            version = item["to_version"]
            
            # 检查签名是否存在于代码中
            if not check_function_in_code(code, signature):
                missing_signatures_count += 1
                missing_signatures.append({
                    "index": index,
                    "signature": signature,
                    "version": version
                })
        
        # 输出结果
        print(f"\n统计结果:")
        print(f"总数据条数: {len(data)}")
        print(f"签名不存在于代码中的数量: {missing_signatures_count}")
        
        # 输出详细的不匹配信息
        if missing_signatures:
            print("\n不匹配的签名列表:")
            for item in missing_signatures:
                print(f"索引 {item['index']}: {item['version']}")
            
    except FileNotFoundError:
        print(f"错误: 找不到文件 '{json_file_path}'")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"错误: '{json_file_path}' 不是有效的JSON文件")
        sys.exit(1)
    except Exception as e:
        print(f"发生错误: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()