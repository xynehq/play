"""
python evaluate_models/eval_models_rq1.py --file_a ./data/final_dataset.json --file_b ./data/final_dataset.json --output ./data/RQ1/rq1_results.json --api_key sk-z7CzOyi3VfIQHv8xkEo3D6rb7jAIApdVtmUSmktU9SlIbBFa --base_url https://api.agicto.cn/v1
"""


import json
import os
import subprocess
import tempfile
import re
import sys
from typing import Dict, List, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI
from RAG_unit import get_RAG_document

# Define available models
MODELS = [
    "kat-dev-hs"
]

# API configuration
API_KEY = os.getenv('API_KEY', 'sk-67cI50BNxSw7SsYSkQGvGw')
BASE_URL = os.getenv('BASE_URL', 'https://grid.ai.juspay.net')

def call_LLM(prompt: str, model: str, api_key: str, base_url: str) -> str:
    """Call the LLM API and return the response."""
    client = OpenAI(api_key=api_key, base_url=base_url)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )        
        code = response.choices[0].message.content.strip()
        print(f"\n==== 原始响应 ====\n{code[:300]}...\n")
        return code
        
    except Exception as e:
        print(f"Error calling LLM {model}: {str(e)}")
        return ""

def extract_rust_code(response: str) -> str:
    """Extract Rust code from the LLM response."""
    # Look for code between ```rust and ``` markers
    rust_pattern = r"```(?:rust)?\s*([\s\S]*?)```"
    matches = re.findall(rust_pattern, response)
    
    if matches:
        return matches[0].strip()
    
    # If no code blocks found, try to extract any code-like content
    lines = response.strip().split('\n')
    code_lines = []
    in_code_block = False
    
    for line in lines:
        # Include use statements and function definitions
        if re.match(r'^\s*use\s+', line):
            code_lines.append(line)
            in_code_block = True
        # Start function block
        elif re.match(r'\s*fn\s+\w+', line):
            in_code_block = True
            code_lines.append(line)
        # Continue any code block
        elif in_code_block:
            code_lines.append(line)
            # Check for end of function
            if line.strip() == "}" and code_lines:
                # Break if we've collected a reasonable amount (function complete)
                if len(code_lines) > 3:
                    break
    
    return '\n'.join(code_lines) if code_lines else response

def check_function_signature(code: str, signature: str) -> bool:
    """Check if the generated code contains a function matching the required signature."""
    import re
    
    # 如果签名为空或代码为空，直接返回False
    if not signature or not code:
        return False
    
    # 清理签名，移除属性、文档注释和可见性修饰符
    clean_signature = re.sub(r'#\[.*?\]', '', signature)
    clean_signature = re.sub(r'///.*?\n', '', clean_signature)
    clean_signature = re.sub(r'//.*?\n', '', clean_signature)
    clean_signature = re.sub(r'pub\s+', '', clean_signature).strip()
    
    # 提取函数名
    fn_match = re.search(r'fn\s+(\w+)', clean_signature)
    if not fn_match:
        return False
    
    fn_name = fn_match.group(1)
    
    # 基本检查：代码中必须有这个函数名
    if not re.search(r'fn\s+' + re.escape(fn_name) + r'\s*[(<]', code):
        return False
    
    # 更宽松的检查：只要函数名匹配，就认为签名正确
    # 这种方法适合初步筛选，在大多数情况下函数名匹配就足够了
    return True
def check_api_usage(code: str, api_name: str) -> bool:
    """Check if the generated code uses the specified API."""
    import re
    
    # 如果API名称为空或代码为空，直接返回True
    if not api_name or not code:
        return False
    
    # 获取API的最后部分(函数/方法名)
    base_api_name = api_name.split('::')[-1] if '::' in api_name else api_name
    
    # 仅做最基本检查，并打印结果但总是返回True
    full_match = re.search(r'\b' + re.escape(api_name) + r'\b', code)
    base_match = re.search(r'\b' + re.escape(base_api_name) + r'\b', code)
    
    if full_match or base_match:
        return True
    else:
        return False
    

    

def create_test_file(code: str, test_program: str) -> str:
    """Combine code solution and test program into a complete test file."""
    # Wrap test program in appropriate module if not already done
    if "#[cfg(test)]" not in test_program:
        test_program = f"""
#[cfg(test)]
mod tests {{
    use super::*;
    
    {test_program}
}}
"""
    
    return f"{code}\n\n{test_program}"

def run_rust_test(test_file_content: str, rust_version: str = "stable") -> Tuple[bool, str]:
    """Run Rust test file with the specified version and return success status and error message."""
    with tempfile.NamedTemporaryFile(suffix='.rs', delete=False) as temp_file:
        temp_file_path = temp_file.name
        temp_file.write(test_file_content.encode('utf-8'))
    
    try:
        # Remove extension and add new extension
        base_path = os.path.splitext(temp_file_path)[0]
        output_path = f"{base_path}.exe" if os.name == 'nt' else base_path
        
        # Compile command with specific Rust version
        compile_cmd = f'rustup run {rust_version} rustc --test "{temp_file_path}" -o "{output_path}"'
        compile_result = subprocess.run(
            compile_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if compile_result.returncode != 0:
            return False, f"Compilation error: {compile_result.stderr}"
        
        # Run the compiled test
        test_cmd = f'"{output_path}"'
        test_result = subprocess.run(
            test_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if test_result.returncode != 0:
            return False, f"Test execution error: {test_result.stderr}"
        
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "Timeout: Test took too long to execute"
    except Exception as e:
        return False, f"Error running tests: {str(e)}"
    finally:
        # Clean up temporary files
        try:
            os.remove(temp_file_path)
            if os.path.exists(output_path):
                os.remove(output_path)
        except:
            pass

def get_code_generation_prompt(query: str, api_info: Dict[str, Any], function_signature: str) -> str:
    """Create a prompt for code generation with the specified information."""

    retrieved_docs = get_RAG_document(query)
    prompt = f"""
    You are an expert Rust programmer. Write a Rust function implementation for the following task:

    Task Description:
    {query}

    Required Function Signature:
    ```rust
    {function_signature}
    ```

    Retrieved Documentation:
    {retrieved_docs}

    Requirements:
    1. Include all necessary import statements (use declarations) before your function.
    2. Implement ONLY the function with the given signature, no additional functions.
    3. Make sure your code is compatible with relevant Rust version (must later than 1.71.0)
    4. Do not include tests, main function, or any code outside the required function and imports.
    5. Do not include additional comments or explanations.

    Respond with ONLY the Rust imports and function implementation, nothing else.    

"""          
    return prompt

def process_task(task_a: Dict[str, Any], task_b: Dict[str, Any], model: str, api_key: str, base_url: str) -> Dict[str, Any]:
    """Process a single task for a specific model."""
    result = task_a.copy()  # Start with a copy of the original task_a data
    
    # Extract required fields
    query = task_a.get("query", "")
    function_signature = task_a.get("function_signature", "")
    test_program = task_a.get("test_program", "")
    
    # Generate code
    prompt = get_code_generation_prompt(query, task_b, function_signature)
    raw_response = call_LLM(prompt, model, api_key, base_url)
    code = extract_rust_code(raw_response)
    
    # Check function signature
    if not check_function_signature(code, function_signature):
        result[f"RAG_{model}_code"] = "INCORRECT SIG"
        result[f"RAG_{model}_test_result"] = "FAILED"
        return result
    
    # Check API usage
    api_name = task_b.get("name", "")
    if not check_api_usage(code, api_name):
        result[f"RAG_{model}_code"] = "INCORRECT API"
        result[f"RAG_{model}_test_result"] = "FAILED"
        return result
    
    # Determine Rust version to use
    # Always use stable to avoid version compatibility issues
    rust_version = "stable"
    
    # Run test
    test_file = create_test_file(code, test_program)
    success, error_message = run_rust_test(test_file, rust_version)
    
    # Set result fields
    result[f"RAG_{model}_code"] = code
    result[f"RAG_{model}_test_result"] = "SUCCESS" if success else "FAILED"
    
    # if not success:
    #     print(f"Test failed for task {task_a.get('task_id', 'unknown')}, model {model}: {error_message}")
    
    return result

def process_all_models(file_a_data: List[Dict[str, Any]], file_b_data: Dict[str, str], models: List[str], 
                      api_key: str, base_url: str, output_file: str, max_workers: int = 4):
    """Process all tasks for all models in parallel."""
    results = []
    processed_task_ids = set()
    
    # 检查输出文件是否存在，如果存在则加载已有结果
    try:
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
                # 记录已经处理过的任务ID
                for result in results:
                    processed_task_ids.add(result.get("task_id", ""))
                print(f"Loaded {len(results)} existing results from {output_file}")
    except Exception as e:
        print(f"Error loading existing results: {str(e)}")
        # 如果加载失败，使用空列表开始
        results = []
        processed_task_ids = set()
    
    # 创建任务ID到task_b数据的映射，以便快速查找
    task_b_mapping = {item.get("task_id", ""): item for item in file_b_data}
    
    # 过滤出尚未处理的任务
    remaining_tasks = [task for task in file_a_data if task.get("task_id", "") not in processed_task_ids]

    
    with tqdm(total=len(remaining_tasks) * len(models), desc="Processing tasks") as pbar:
        for task_a in remaining_tasks:
            task_id = task_a.get("task_id", "")
            
            # 跳过没有匹配的task_b的任务
            if task_id not in task_b_mapping:
                print(f"Warning: No matching data found in file B for task_id {task_id}")
                continue
            
            task_b = task_b_mapping[task_id]
            task_result = task_a.copy()
            
            # 为每个模型并行处理
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(process_task, task_a, task_b, model, api_key, base_url): model
                    for model in models
                }
                
                for future in as_completed(futures):
                    model = futures[future]
                    try:
                        model_result = future.result()
                        # 更新任务结果，添加模型特定的字段
                        task_result[f"RAG_{model}_code"] = model_result.get(f"RAG_{model}_code", "")
                        task_result[f"RAG_{model}_test_result"] = model_result.get(f"RAG_{model}_test_result", "FAILED")
                    except Exception as e:
                        print("ERROR: " + str(e))
                        task_result[f"RAG_{model}_code"] = f"ERROR: {str(e)}"
                        task_result[f"RAG_{model}_test_result"] = "FAILED"
                    finally:
                        pbar.update(1)
            
            results.append(task_result)
            
            # 每处理10个任务保存一次检查点
            if len(results) % 10 == 0:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                # print(f"Checkpoint saved after processing {len(results)} tasks")
    
    # Final save
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"\nProcessing complete. Results saved to: {output_file}")
    
    # Calculate and print success rates per model
    for model in models:
        success_count = sum(1 for result in results if result.get(f"RAG_{model}_test_result", "") == "SUCCESS")
        success_rate = (success_count / len(results)) * 100 if results else 0
        incorrect_sig = sum(1 for result in results if result.get(f"RAG_{model}_code", "") == "INCORRECT SIG")
        incorrect_api = sum(1 for result in results if result.get(f"RAG_{model}_code", "") == "INCORRECT API")
        
        print(f"\nModel: {model}")
        print(f"Success rate: {success_rate:.2f}% ({success_count}/{len(results)})")
        print(f"Incorrect signatures: {incorrect_sig}")
        print(f"Incorrect API usage: {incorrect_api}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate LLM models on Rust API evolution tasks")
    parser.add_argument("--file_a", required=True, help="Input JSON file A with tasks and test programs")
    parser.add_argument("--file_b", required=True, help="Input JSON file B with API information")
    parser.add_argument("--output", required=True, help="Output JSON file for results")
    parser.add_argument("--models", nargs="+", default=MODELS, help="Models to evaluate")
    parser.add_argument("--max_workers", type=int, default=3, help="Maximum number of concurrent workers")
    parser.add_argument("--api_key", help="API key for LLM service")
    parser.add_argument("--base_url", help="Base URL for LLM service")
    
    args = parser.parse_args()
    
    # Use provided API credentials if available
    api_key = args.api_key if args.api_key else API_KEY
    base_url = args.base_url if args.base_url else BASE_URL
    
    # Check if API key is provided
    if api_key == "your-api-key-here":
        print("Warning: No API key provided. Please set the API_KEY environment variable or use --api_key.")
    
    # Load the data from files
    try:
        with open(args.file_a, "r", encoding="utf-8") as f:
            file_a_data = json.load(f)
        
        with open(args.file_b, "r", encoding="utf-8") as f:
            file_b_data = json.load(f)
            
        print(f"Loaded {len(file_a_data)} tasks from file A and {len(file_b_data)} API entries from file B")
    except Exception as e:
        print(f"Error loading input files: {str(e)}")
        sys.exit(1)
    
    # Process all models
    process_all_models(
        file_a_data, 
        file_b_data, 
        args.models, 
        api_key, 
        base_url, 
        args.output, 
        args.max_workers
    )

if __name__ == "__main__":
    main()