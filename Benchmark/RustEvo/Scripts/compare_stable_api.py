import json
import os
import sys
from typing import Dict, Any
#from generate_query import generate_task
#from generate_code import generate_and_validate_rust_code
#from generate_test import process_item
from unit import generate_task,generate_and_validate_rust_code,process_item
# 配置信息
API_KEY = ''
BASE_URL = ''
MODEL = 'deepseek-v3'


def process_api_entry(entry: Dict[str, Any], entry_id: str) -> Dict[str, Any]:
    """处理单个API条目"""
    # Step 1: 生成查询和签名
    query_entry = {
        'change_type': 'stabilized',  # 根据实际情况调整
        'name': entry.get('name', ''),
        'to_version': entry.get('stability', '1.0.0'),
        'documentation': entry.get('description', ''),
        'examples': entry.get('examples', []),
        'deprecated': entry.get('deprecated', False),
        'crate': 'std'  # 标准库
    }

    # 生成查询
    try:
        query_result = generate_task(query_entry)
        entry['generated_query'] = query_result.get('query', '')
        entry['function_signature'] = query_result.get('function_signature', '')
    except Exception as e:
        entry['error'] = f"Query generation failed: {str(e)}"
        return entry

    # Step 2: 生成代码
    code_entry = {
        'query': entry['generated_query'],
        'function_signature': entry['function_signature'],
        'name': entry['name'],
        'to_version': entry.get('stability', '1.0.0'),
        'crate': 'std',
        'documentation': entry.get('description', ''),
        'examples': entry.get('examples', [])
    }

    try:
        code_result = generate_and_validate_rust_code(code_entry)
        entry['generated_code'] = code_result.get('code', '')
        entry['validation_status'] = code_result.get('validation_status', '')
    except Exception as e:
        entry['error'] = f"Code generation failed: {str(e)}"
        return entry

    # Step 3: 生成测试并运行
    test_entry = {
        'code': entry['generated_code'],
        'function_signature': entry['function_signature'],
        'query': entry['generated_query'],
        'to_version': entry.get('stability', '1.0.0'),
        'crate': 'std'
    }

    try:
        test_result = process_item(test_entry, 0)  # index参数仅用于日志
        entry['generated_test'] = test_result.get('test_program', '')
        entry['test_status'] = 'success' if test_result['test_program'] not in [
            'INCORRECT CODE', 'INCORRECT TEST'] else 'failed'
    except Exception as e:
        entry['error'] = f"Test generation failed: {str(e)}"

    return entry


def process_all_entries(input_file: str, output_file: str):
    """处理所有API条目"""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    processed_data = {}
    for idx, (api_name, api_info) in enumerate(data.items()):
        print(f"Processing {idx + 1}/{len(data)}: {api_name}")
        processed_data[api_name] = process_api_entry(api_info, api_name)

        # 每处理10个保存一次进度
        if (idx + 1) % 10 == 0:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)

    # 保存最终结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)
    print(f"Processing complete. Results saved to {output_file}")


if __name__ == "__main__":
    #python compare_stable_api.py rust_api_info_stable_1.json results_rust_api_info_stable_1.json

    if len(sys.argv) != 3:
        print("Usage: python main_pipeline.py input.json output.json")
        sys.exit(1)

    input_json = sys.argv[1]
    output_json = sys.argv[2]

    if not os.path.exists(input_json):
        print(f"Input file {input_json} not found!")
        sys.exit(1)

    process_all_entries(input_json, output_json)