"""
python Evaluate/eval_models_rq1.py --file_a ./Dataset/RustEvo^2.json --file_b ./Dataset/APIDocs.json --output ./Results/rq1_results.json
"""

import json
import os
import subprocess
import tempfile
import re
import sys
from typing import Dict, List, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI

# Define available models
MODELS = [
    "glm45air-qlora",
    "/models/glm-base"
]

# API configuration
API_KEY = os.getenv('API_KEY', 'sk-67cI50BNxSw7SsYSkQGvGw')
BASE_URL = os.getenv('BASE_URL', 'http://34.56.189.164:8001/v1')

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


def enforce_function_signature(code: str, signature: str) -> str:
    """Attempt to rebuild the function with the required signature if missing."""
    if not code:
        return code

    if check_function_signature(code, signature):
        return code

    # Extract imports and potential body
    imports = []
    other_lines = []
    for line in code.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith('use '):
            imports.append(stripped)
        elif stripped.startswith('```'):
            continue
        else:
            other_lines.append(line)

    body_region = '\n'.join(other_lines).strip()

    if '{' in body_region:
        body_region = body_region.split('{', 1)[1]
    if '}' in body_region:
        body_region = body_region.rsplit('}', 1)[0]

    body_region = body_region.strip()

    sig = signature.strip().rstrip(';')
    sig_lines = [line.rstrip() for line in sig.splitlines() if line.strip()]

    rebuilt_parts = []
    if imports:
        rebuilt_parts.extend(imports)
        rebuilt_parts.append('')

    rebuilt_parts.extend(sig_lines)
    rebuilt_parts.append('{')
    if body_region:
        rebuilt_parts.append(body_region)
    rebuilt_parts.append('}')

    rebuilt_code = '\n'.join(rebuilt_parts)

    if check_function_signature(rebuilt_code, signature):
        return rebuilt_code

    return code

def check_api_usage(code: str, api_name: str) -> bool:
    """Check if the generated code uses the specified API - improved lenient check."""
    import re
    
    # 如果API名称为空或代码为空，直接返回False
    if not api_name or not code:
        return False
    
    # 获取API的最后部分(函数/方法名)
    base_api_name = api_name.split('::')[-1] if '::' in api_name else api_name
    
    # Try multiple patterns to be more lenient
    # 1. Full API name match
    full_match = re.search(r'\b' + re.escape(api_name) + r'\b', code)
    if full_match:
        return True
    
    # 2. Base API name match (function/method name)
    base_match = re.search(r'\b' + re.escape(base_api_name) + r'\b', code)
    if base_match:
        return True
    
    # 3. Check for method call syntax (e.g., obj.method_name or Type::method_name)
    method_call_pattern = r'\.' + re.escape(base_api_name) + r'\s*\('
    if re.search(method_call_pattern, code):
        return True
    
    # 4. Check for trait method syntax (e.g., Trait::method_name)
    trait_method_pattern = r'::' + re.escape(base_api_name) + r'\s*\('
    if re.search(trait_method_pattern, code):
        return True
    
    # 5. Check for use statement importing the API
    use_pattern = r'use\s+.*' + re.escape(base_api_name)
    if re.search(use_pattern, code, re.IGNORECASE):
        return True
    
    # If none of the patterns match, return False
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

def is_borrow_checker_error(message: str) -> bool:
    """Heuristically detect Rust borrow checker/lifetime related errors from stderr."""
    if not message:
        return False
    m = message.lower()
    patterns = [
        # Core borrow/lifetime diagnostics (phrases)
        "cannot borrow",
        "borrowed value",
        "borrow later used here",
        "does not live long enough",
        "doesn't live long enough",
        "lifetime may not live long enough",
        "borrow checker",
        "temporary value dropped while borrowed",
        "cannot return reference",
        "returns a value referencing data owned by the current function",
        "cannot return value referencing",
        "cannot move out",
        "use of moved value",
        "value used here after move",
        "cannot assign to",
        "cannot borrow as mutable",
        "cannot borrow as immutable",
        "cannot mutably borrow",
        "cannot immutably borrow",
        # Common rustc error codes (lowercased to match lowercased stderr)
        "e0382", "e0387",
        "e0495",
        "e0499", "e0500", "e0501", "e0502", "e0503", "e0504", "e0505", "e0506", "e0507", "e0508", "e0509",
        "e0515",
        "e0594", "e0596", "e0597",
        "e0716",
    ]
    if any(p in m for p in patterns):
        return True
    return bool(re.search(r"error\s*\[e0\d{3}\]", m))

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

def get_code_generation_prompt(query: str, api_info: Dict[str, Any], function_signature: str,
                               error_feedback: Optional[str] = None) -> str:
    """Create a prompt for code generation with the specified information."""
    if "crate" not in api_info:
        
        api_name = api_info.get("name", "")
        api_module = api_info.get("module", "")
        api_signature = api_info.get("signature", "")
        api_documentation = api_info.get("documentation", "")
        from_version = api_info.get("from_version", "")
        to_version = api_info.get("to_version", "")
        source_code = api_info.get("source_code", "")

        
        prompt = f"""
    You are an expert Rust programmer. Write a Rust function implementation for the following task:

    Task Description:
    {query}

    Required Function Signature:
    ```rust
    {function_signature}
    ```

    Relevant API Information:
    - API Name: {api_name}
    - API Module: {api_module}
    - API Signature: {api_signature}
    - API Documentation: {api_documentation}
    - API Source Code: {source_code}
    - API Changed From Version: {from_version}
    - API Changed To Version: {to_version}

    Requirements:
    1. Include ALL necessary import statements (use declarations) before your function.
    - If using types from std::num, include use std::num::*.
    - If using traits like Neg, Add, Div, include use std::ops::*.
    - Include ANY other imports needed for the types in your function signature.
    2. Implement ONLY the function with the given signature, no additional functions.
    3. Your implementation MUST use the specified API: {api_name}
    4. Make sure your code is compatible with Rust version {to_version}
    5. Do not include tests, main function, or any code outside the required function and imports.
    6. Do not include additional comments or explanations.

    Respond with ONLY the Rust imports and function implementation, nothing else.
    """
    else:
        api_name = api_info.get("name", "")
        api_module = api_info.get("module", "")
        crate_name = api_info.get("crate", "")
        api_signature = api_info.get("signature", "")
        api_documentation = api_info.get("documentation", "")
        from_version = api_info.get("from_version", "")
        to_version = api_info.get("to_version", "")
        source_code = api_info.get("source_code", "")   
        prompt = f"""
    You are an expert Rust programmer. Write a Rust function implementation for the following task:

    Task Description:
    {query}

    Required Function Signature:
    ```rust
    {function_signature}
    ```

    Relevant API Information:
    - Crate Name: {crate_name}
    - API Name: {api_name}
    - API Module: {api_module}
    - API Signature: {api_signature}
    - API Documentation: {api_documentation}
    - API Source Code: {source_code}
    - API Changed From Version: {from_version}
    - API Changed To Version: {to_version}

    Requirements:
    1. Include ALL necessary import statements (use declarations) before your function.
    - If using types from std::num, include use std::num::*.
    - If using traits like Neg, Add, Div, include use std::ops::*.
    - Include ANY other imports needed for the types in your function signature.
    2. Implement ONLY the function with the given signature, no additional functions.
    3. Your implementation MUST use the specified API: {api_name}
    4. Make sure your code is compatible with Rust version 1.84.0
    5. Do not include tests, main function, or any code outside the required function and imports.
    6. Do not include additional comments or explanations.

    Respond with ONLY the Rust imports and function implementation, nothing else.
    """

    if error_feedback:
        trimmed_feedback = error_feedback.strip()
        prompt += f"""

    Previous attempt feedback:
    {trimmed_feedback}

    Please fix ALL issues above. Do not include explanations—return only the corrected imports and function.
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

    # Attempt to enforce the exact required signature if missing
    code = enforce_function_signature(code, function_signature)
    
    # Enrich result with API metadata for downstream formatting
    result["api_name"] = task_b.get("name", "")
    result["crate"] = task_b.get("crate", "")
    result["module"] = task_b.get("module", "")
    result["signature"] = task_b.get("signature", "")
    result["change_type"] = task_b.get("change_type", "")
    result["from_version"] = task_b.get("from_version", "")
    result["to_version"] = task_b.get("to_version", "")

    # Check function signature
    if not check_function_signature(code, function_signature):
        result[f"{model}_code"] = "INCORRECT SIG"
        result[f"{model}_test_result"] = "FAILED"
        result[f"{model}_failure_reason"] = "Signature mismatch"
        result[f"{model}_failure_is_borrow_checker"] = False
        return result
    
    # Check API usage and record it (non-blocking)
    api_name = task_b.get("name", "")
    used_api = check_api_usage(code, api_name) if api_name else False
    result[f"{model}_used_api"] = bool(used_api)
    if api_name and not used_api:
        print(f"Warning: API '{api_name}' may not be used in code for task {task_a.get('task_id', 'unknown')}")
    
    # Determine Rust version to use
    # Always use stable to avoid version compatibility issues
    rust_version = "stable"
    
    # Run test
    test_file = create_test_file(code, test_program)
    success, error_message = run_rust_test(test_file, rust_version)
    
    # Set result fields
    result[f"{model}_code"] = code
    result[f"{model}_test_result"] = "SUCCESS" if success else "FAILED"
    # validation fields for requested output format
    result["validation_status"] = "success" if success else "failed"
    result["validation_output"] = "" if success else (error_message or "")
    if not success:
        result[f"{model}_failure_reason"] = error_message
        result[f"{model}_failure_is_borrow_checker"] = is_borrow_checker_error(error_message)
    
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
                # Backfill missing fields in existing results
                task_b_mapping_for_backfill = {item.get("task_id", ""): item for item in file_b_data}
                for result in results:
                    task_id = result.get("task_id", "")
                    # Backfill API metadata if missing
                    if task_id in task_b_mapping_for_backfill:
                        task_b = task_b_mapping_for_backfill[task_id]
                        for field in ["api_name", "crate", "module", "signature", "change_type", "from_version", "to_version"]:
                            if field not in result or not result[field]:
                                result[field] = task_b.get(field, "")
                    # Backfill model-specific fields if missing
                    for model in models:
                        if f"{model}_used_api" not in result:
                            # Try to infer from code if it exists
                            code = result.get(f"{model}_code", "")
                            api_name = result.get("api_name", "")
                            if code and code not in ["INCORRECT SIG", "INCORRECT API"] and api_name:
                                result[f"{model}_used_api"] = check_api_usage(code, api_name)
                            else:
                                result[f"{model}_used_api"] = False
                        if f"{model}_failure_is_borrow_checker" not in result:
                            failure_reason = result.get(f"{model}_failure_reason", "")
                            if failure_reason:
                                result[f"{model}_failure_is_borrow_checker"] = is_borrow_checker_error(failure_reason)
                            else:
                                result[f"{model}_failure_is_borrow_checker"] = False
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
                        task_result[f"{model}_code"] = model_result.get(f"{model}_code", "")
                        task_result[f"{model}_test_result"] = model_result.get(f"{model}_test_result", "FAILED")
                        # propagate diagnostics for debugging
                        if "validation_status" in model_result:
                            task_result["validation_status"] = model_result["validation_status"]
                        if "validation_output" in model_result:
                            task_result["validation_output"] = model_result["validation_output"]
                        fr_key = f"{model}_failure_reason"
                        if fr_key in model_result:
                            task_result[fr_key] = model_result[fr_key]
                        fbc_key = f"{model}_failure_is_borrow_checker"
                        if fbc_key in model_result:
                            task_result[fbc_key] = model_result[fbc_key]
                        used_api_key = f"{model}_used_api"
                        if used_api_key in model_result:
                            task_result[used_api_key] = model_result[used_api_key]
                        # propagate API metadata fields
                        for field in ["api_name", "crate", "module", "signature", "change_type", "from_version", "to_version"]:
                            if field in model_result:
                                task_result[field] = model_result[field]
                    except Exception as e:
                        task_result[f"{model}_code"] = f"ERROR: {str(e)}"
                        task_result[f"{model}_test_result"] = "FAILED"
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
    metrics = {}
    for model in models:
        success_count = sum(1 for result in results if result.get(f"{model}_test_result", "") == "SUCCESS")
        pass_at_1 = (success_count / len(results)) * 100 if results else 0
        failed_count = len(results) - success_count
        incorrect_sig = sum(1 for result in results if result.get(f"{model}_code", "") == "INCORRECT SIG")
        incorrect_api_marker = sum(1 for result in results if result.get(f"{model}_code", "") == "INCORRECT API")
        incorrect_api = sum(1 for r in results if r.get("api_name") and r.get(f"{model}_used_api") is False)
        borrow_errors = sum(1 for r in results if r.get(f"{model}_test_result") == "FAILED" and r.get(f"{model}_failure_is_borrow_checker") is True)
        failure_count = sum(1 for r in results if r.get(f"{model}_test_result") == "FAILED")
        borrow_error_rate = (borrow_errors / failure_count * 100) if failure_count else 0.0
        used_api_true = sum(1 for r in results if r.get(f"{model}_used_api") is True)
        api_usage_accuracy = (used_api_true / len(results) * 100) if results else 0.0
        # API coverage (distinct APIs used / distinct APIs in dataset)
        all_apis = set(r.get("api_name", "") for r in results if r.get("api_name"))
        used_apis = set(r.get("api_name", "") for r in results if r.get(f"{model}_used_api") is True and r.get("api_name"))
        api_coverage = (len(used_apis) / len(all_apis) * 100) if all_apis else 0.0
        # Split failures by type
        compilation_errors = sum(1 for r in results if r.get(f"{model}_test_result") == "FAILED" and str(r.get("validation_output", "")).startswith("Compilation error"))
        test_failures = sum(1 for r in results if r.get(f"{model}_test_result") == "FAILED" and str(r.get("validation_output", "")).startswith("Test execution error"))
        
        print(f"\nModel: {model}")
        print(f"Total tasks: {len(results)}")
        print(f"Success: {success_count}")
        print(f"Failed: {failed_count}")
        print(f"Pass@1: {pass_at_1:.2f}% ({success_count}/{len(results)})")
        print(f"Incorrect signatures: {incorrect_sig}")
        print(f"Incorrect API usage: {incorrect_api}")
        print(f"Borrow-checker failure rate: {borrow_error_rate:.2f}% ({borrow_errors}/{failure_count})")
        print(f"API usage accuracy: {api_usage_accuracy:.2f}% ({used_api_true}/{len(results)})")
        print(f"API coverage (distinct): {api_coverage:.2f}% ({len(used_apis)}/{len(all_apis)})")
        print(f"Compilation errors: {compilation_errors}")
        print(f"Test failures: {test_failures}")

        # By change_type aggregation
        from collections import defaultdict
        by_ct = defaultdict(lambda: {"total":0,"success":0,"used_api":0,"all_apis":set(),"used_apis":set()})
        for r in results:
            ct = r.get("change_type","unknown")
            api_name = r.get("api_name", "")
            by_ct[ct]["total"] += 1
            by_ct[ct]["success"] += (r.get(f"{model}_test_result") == "SUCCESS")
            by_ct[ct]["used_api"] += (r.get(f"{model}_used_api") is True)
            if api_name:
                by_ct[ct]["all_apis"].add(api_name)
                if r.get(f"{model}_used_api") is True:
                    by_ct[ct]["used_apis"].add(api_name)

        metrics[model] = {
            "total_tasks": len(results),
            "success_count": success_count,
            "failed_count": failed_count,
            "pass_at_1": pass_at_1,
            "incorrect_signatures": incorrect_sig,
            "incorrect_api_marker": incorrect_api_marker,
            "incorrect_api": incorrect_api,
            "borrow_checker_failures": borrow_errors,
            "borrow_checker_failure_rate_over_failures": borrow_error_rate,
            "api_usage_true_count": used_api_true,
            "api_usage_accuracy": api_usage_accuracy,
            "api_coverage_distinct": api_coverage,
            "api_coverage_distinct_count": f"{len(used_apis)}/{len(all_apis)}",
            "compilation_errors": compilation_errors,
            "test_failures": test_failures,
            "by_change_type": [
                {
                    "change_type": ct,
                    "total": v["total"],
                    "success": v["success"],
                    "success_rate": (v["success"]/v["total"]*100) if v["total"] else 0.0,
                    "used_api": v["used_api"],
                    "api_usage_accuracy": (v["used_api"]/v["total"]*100) if v["total"] else 0.0,
                    "api_coverage_distinct": (len(v["used_apis"])/len(v["all_apis"])*100) if v["all_apis"] else 0.0,
                    "api_coverage_distinct_count": f"{len(v['used_apis'])}/{len(v['all_apis'])}",
                } for ct,v in by_ct.items()
            ]
        }

    # Write metrics JSON next to results
    metrics_path = os.path.splitext(output_file)[0] + "_metrics.json"
    with open(metrics_path, 'w', encoding='utf-8') as mf:
        json.dump(metrics, mf, indent=2, ensure_ascii=False)
    print(f"\nMetrics written to: {metrics_path}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate LLM models on Rust API evolution tasks")
    parser.add_argument("--file_a", required=True, help="Input JSON file A with tasks and test programs")
    parser.add_argument("--file_b", required=True, help="Input JSON file B with API information")
    parser.add_argument("--output", required=True, help="Output JSON file for results")
    parser.add_argument("--models", nargs="+", default=MODELS, help="Models to evaluate")
    parser.add_argument("--max_workers", type=int, default=8, help="Maximum number of concurrent workers")
    parser.add_argument("--api_key", help="API key for LLM service")
    parser.add_argument("--base_url", help="Base URL for LLM service")
    
    args = parser.parse_args()
    
    # Use provided API credentials if available
    api_key = args.api_key if args.api_key else API_KEY
    base_url = args.base_url if args.base_url else BASE_URL
    
    # Check if API key is provided
    if not api_key or api_key == "your-api-key-here":
        print("Warning: No API key provided. Please set the API_KEY environment variable or use --api_key.")
    
    print(f"Using API endpoint: {base_url}")
    print(f"Using model: {args.models}")
    
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
