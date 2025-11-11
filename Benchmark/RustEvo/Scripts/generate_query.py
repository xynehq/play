import itertools
import json
import random
import concurrent.futures
import concurrent
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import threading

api_key = ''
base_url = ''
model = 'deepseek-v3'

input_file = './reports/crates_api_changes.json'
output_file = './data/queries/queries_crates.json'
# input_file = './reports/rust_api_changes_latest_version.json'
# output_file = './data/queries/queries_rust.json'

with open(input_file, 'r', encoding='utf-8') as file:
    api_changes = json.load(file)


# Define prompts based on change_type
prompt_templates = {
    "stabilized": "As a Rust expert, generate a concise programming task and corresponding Rust function signature that implicitly requires the use of a recently stabilized standard library API. ### Core Principles ###  Follow these rules: Do not directly mention the API's name or signature. Clearly describe a real-world scenario solved uniquely or efficiently by this newly stabilized feature. The generated function signature should strongly hint at the use of the newly stabilized API. Be unpredictable in your task format (e.g., don't always begin with 'You ...'. Instead, use more variable format including Interrogative, Imperative, Conditional, Declarative, Challenge-based, Scenario-based and etc. ).",
    "signature": "As a Rust expert, generate a concise programming task and corresponding Rust function signature that implicitly requires using an API whose signature was recently updated. ### Core Principles ### Follow these rules: Do not directly reveal the name or signature changes of the API. Provide a realistic context emphasizing limitations or inconveniences solved by the updated API. The generated function signature should implicitly lead to choosing the newly updated API over older versions. Be unpredictable in your task format (e.g., don't always begin with 'You ...'. Instead, use more variable format including Interrogative, Imperative, Conditional, Declarative, Challenge-based, Scenario-based and etc. ).",
    "implicit": "As a Rust expert, generate a concise programming task and corresponding Rust function signature that implicitly requires an API whose internal implementation or functionality recently changed without altering its signature. ### Core Principles ### Follow these rules: Do not directly mention the API name or its internal changes explicitly. Clearly describe an observable behavior improvement (e.g., performance, memory usage, correctness). The generated function signature and context should implicitly prompt the use of this API, relying on documentation or inferred behavior differences. Be unpredictable in your task format.Be unpredictable in your task format (e.g., don't always begin with 'You ...'. Instead, use more variable format including Interrogative, Imperative, Conditional, Declarative, Challenge-based, Scenario-based and etc. ).",
    "deprecated": "As a Rust expert, generate a concise programming task and corresponding Rust function signature that implicitly encourages replacing a deprecated or replaced API with a newly recommended one. ### Core Principles ### Follow these rules: Do not explicitly mention the deprecated API's name. Provide context that strongly suggests the disadvantages (performance, safety, usability) of the old method. The generated function signature should implicitly guide users towards the recommended alternative. Be unpredictable in your task format (e.g., don't always begin with 'You ...'. Instead, use more variable format including Interrogative, Imperative, Conditional, Declarative, Challenge-based, Scenario-based and etc. ).",
}

# Function to generate query and signature using OpenAI

import time

def generate_task(api_entry, max_retries=3):
    prompt = prompt_templates.get(api_entry['change_type'], "")
    supplement = f"""
    ### API Characteristics ###
    {api_entry}
    ### Generation Format ###
    <query>
    [Your generated query here]
    </query>
    <signature>
    [Your generated signature here]
    </signature>
    """

    client = OpenAI(api_key=api_key, base_url=base_url)
    
    for attempt in range(max_retries):
        try:
            # 可能在重试时增加温度以获得不同输出
            temp = 0.7 + (attempt * 0.1)  # 逐渐增加温度
            
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": supplement}
                ],
                temperature=min(temp, 0.8)  # 确保温度不超过1.0
            )

            response = completion.choices[0].message.content
            
            # 提取 query 和 signature
            query = ""
            signature = ""
            
            if "<query>" in response and "</query>" in response:
                query = response.split('<query>')[1].split('</query>')[0].strip()
            else:
                print(f"Attempt {attempt+1}: Failed to extract query. Retrying...")
                if attempt < max_retries - 1:
                    time.sleep(1)  # 避免过于频繁的API调用
                    continue
                    
            if "<signature>" in response and "</signature>" in response:
                signature = response.split('<signature>')[1].split('</signature>')[0].strip()
            else:
                print(f"Attempt {attempt+1}: Failed to extract signature. Retrying...")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
            
            # 只有当两个字段都成功提取时才返回
            if query and signature:
                # 添加提取的字段到api_entry
                api_entry['query'] = query
                api_entry['function_signature'] = signature
                return api_entry
                
        except Exception as e:
            print(f"Attempt {attempt+1}: Error: {str(e)}. Retrying...")
            if attempt < max_retries - 1:
                time.sleep(2)  # 网络错误时等待更长时间
    
    # 所有重试都失败后的处理
    print(f"All {max_retries} attempts failed for API: {api_entry.get('name', 'unknown')}")
    api_entry['query'] = "ERROR: Failed to generate query after multiple attempts"
    api_entry['function_signature'] = "ERROR: Failed to generate signature after multiple attempts"
    return api_entry


# Process entries concurrently and save periodically
def process_api_changes(api_changes, output_file):
    updated_entries = []
    # 创建总进度条
    progress_bar = tqdm(total=len(api_changes), desc="Processing API changes")
    
    for idx in range(0, len(api_changes), 50):
        batch = api_changes[idx:idx+50]
        batch_results = [None] * len(batch)  # 预分配结果列表
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # 提交任务时保存原始索引
            future_to_index = {
                executor.submit(generate_task, entry): i 
                for i, entry in enumerate(batch)
            }
            
            # 处理完成的任务
            for future in concurrent.futures.as_completed(future_to_index.keys()):
                original_index = future_to_index[future]
                batch_results[original_index] = future.result()
                progress_bar.update(1)
        
        # 按顺序添加这批次的结果
        updated_entries.extend(batch_results)
        
        # 保存进度
        with open(output_file, 'w', encoding='utf-8') as outfile:
            json.dump(updated_entries, outfile, indent=2, ensure_ascii=False)
    
    progress_bar.close()



# Start processing
process_api_changes(api_changes, output_file)
