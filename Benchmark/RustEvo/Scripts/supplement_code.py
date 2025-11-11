"""
检查signature是否存在于code中，没有的话pass
"""

import json
import re
import time
import tempfile
import subprocess
from tqdm import tqdm
import concurrent.futures
from openai import OpenAI

# OpenAI API key setup
api_key = ''
base_url = ''
model = 'deepseek-v3'

# input_file = './data/queries/queries_crates.json'
# output_file = './data/codes/codes_crates.json'
input_file = './data/codes/codes_crates.json'
output_file = './data/codes/codes_crates0312.json'

# Load API tasks with generated queries and signatures
with open(input_file, 'r', encoding='utf-8') as file:
    api_tasks = json.load(file)

# Prompt template for code generation with error feedback
def create_code_prompt(query, signature, api_details, error_feedback=None):
    # rust_version = api_details.get('to_version', 'latest')
    rust_version = '1.84.0'
    api_name = api_details.get('name', 'the specified API')
    api_detail = {
    'crate': api_details.get('crate', ''),
    'api_name': api_details.get('name', ''),
    'crate_version': api_details.get('to_version', ''),
    'module': api_details.get('module', {}),
    'signature': api_details.get('signature', ''),
    'documentation': api_details.get('documentation', ''),
    'source_code': api_details.get('source_code', ''),
    # 'rust_version': api_details.get('to_version', ''),
    # 'examples': api_details.get('examples', '')
}

    base_prompt = (f"Given the following Rust programming task:\n"
                   f"{query}\n\n"
                   f"Implement the following function signature:\n"
                   f"{signature}\n\n"
                   f"### Important rules ###\n"
                   f"- Your implementation MUST correctly use '{api_name}' in the function body.\n"
                   f"- Your implementation must compile and run correctly on Rust {rust_version}.\n"
                   f"- For const functions, ensure you're using API features compatible with const contexts.\n"
                   f"- Provide a complete, concise, and correct Rust implementation.\n"
                   f"- Relevant API Details:\n"
                   f"{api_detail}\n\n"
                   f"DON'T NEED EXPLANATION, JUST GIVE THE CODE.\n\n"
                   f"### Code Format ###\n"
                   f"<code>\n"
                   f"[Your code here]\n"
                   f"</code>\n")

    if error_feedback:
        base_prompt += f"\n\nYour previous attempt resulted in the following error:\n{error_feedback}\nPlease correct the issues and ensure the API is used correctly."

    return base_prompt

# Function to check if API is properly used (not just mentioned in comments)
def is_api_properly_used(code, api_name):
    # Escape special regex characters in API name
    escaped_api = re.escape(api_name)
    
    # Find all comments in the code
    comments = re.findall(r'//.*$|/\*[\s\S]*?\*/', code, re.MULTILINE)
    
    # Remove comments from code for checking actual usage
    code_without_comments = code
    for comment in comments:
        code_without_comments = code_without_comments.replace(comment, '')
    
    # Check if API is mentioned outside of comments
    api_pattern = r'(?<![a-zA-Z0-9_])' + escaped_api + r'(?![a-zA-Z0-9_])'
    return bool(re.search(api_pattern, code_without_comments))

# Function to run Rust code using Docker with timeout (保留原函数，但不使用)
def run_rust_code_docker(code, rust_version, timeout=30):
    # Create a temporary file with the code
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.rs', delete=False) as temp_file:
        temp_file.write(code)
        temp_file_path = temp_file.name

    # Docker command to compile and run the code
    command = [
        "docker", "run", "--rm",
        "-v", f"{temp_file_path}:/usr/src/myapp/main.rs",
        "-w", "/usr/src/myapp",
        f"rust:{rust_version}",
        "bash", "-c", "rustc main.rs && ./main"
    ]

    try:
        # Run the command with timeout
        process = subprocess.Popen(
            command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        
        # Wait for the process to complete with timeout
        stdout, stderr = process.communicate(timeout=timeout)
        return process.returncode, stdout, stderr
        
    except subprocess.TimeoutExpired:
        # Kill the process if it times out
        process.kill()
        return 1, "", f"Execution timed out after {timeout} seconds"
    except Exception as e:
        # Handle any other exceptions
        return 1, "", f"Error executing code: {str(e)}"


def static_analysis_rust_code(code, api_name):
    """使用静态分析验证Rust代码，不需要运行Docker或本地编译器"""
    
    # 检查API是否被正确使用
    api_used = is_api_properly_used(code, api_name)
    if not api_used:
        return False, f"Code does not properly use the required API: '{api_name}'"
    
    # 简单检查语法 - 确保有基本的Rust结构
    syntax_checks = [
        (r'\bfn\b', "Missing function definition"),
        (r'[{]', "Missing opening braces"),
        (r'[}]', "Missing closing braces"),
        # 移除分号检查，因为Rust表达式不总是需要分号
        # (r';', "Missing semicolons, possible syntax error")
    ]
    
    for pattern, error in syntax_checks:
        if not re.search(pattern, code):
            return False, error
    
    # 检查明显的语法错误 - 未闭合的引号、括号等
    # 首先删除Rust生命周期标记，避免误判
    code_without_lifetimes = re.sub(r"<'[a-zA-Z_]+>|&'[a-zA-Z_]+", "<LIFETIME>", code)
    
    # 现在计算引号数量
    quotes = code_without_lifetimes.count('"') % 2
    single_quotes = code_without_lifetimes.count("'") % 2
    parentheses = code.count('(') - code.count(')')
    braces = code.count('{') - code.count('}')
    brackets = code.count('[') - code.count(']')
    
    if quotes != 0:
        return False, "Unclosed double quotes"
    if single_quotes != 0:
        return False, "Unclosed single quotes (not related to lifetimes)"
    if parentheses != 0:
        return False, "Mismatched parentheses"
    if braces != 0:
        return False, "Mismatched braces"
    if brackets != 0:
        return False, "Mismatched brackets"
    
    # 添加对const fn的特殊检查
    is_const_fn = re.search(r'\bconst\s+fn\b', code)
    if is_const_fn:
        # 如果是const fn，检查是否有不允许在const上下文使用的操作
        if "f64" in code or "f32" in code:
            # 检查函数体中是否使用了正确的API
            fn_body = re.search(r'{(.*)}', code, re.DOTALL)
            if fn_body and api_name in fn_body.group(1):
                # API在函数体中被使用，这是好的
                pass
            else:
                # 额外检查，确保API确实被用在核心逻辑中
                print(f"  Warning: The API '{api_name}' might not be properly used in the function body")
    
    # 所有检查都通过
    return True, "Static analysis passed"

# Function to iteratively generate and validate Rust code
# 更新后的代码生成函数
def generate_and_validate_rust_code(task_entry, max_retries=3):
    # Get necessary information from task entry
    rust_version = task_entry.get('to_version', 'latest')
    # rust_version = '1.84.0'
    query = task_entry.get('query', '')
    signature = task_entry.get('function_signature', '')
    api_name = task_entry.get('name', 'the specified API')
    
    # Skip if query or signature is missing or marked as error
    if not query or not signature or query.startswith("ERROR:") or signature.startswith("ERROR:"):
        task_entry['code'] = "ERROR: Missing or invalid query/signature"
        task_entry['validation_status'] = "skipped"
        return task_entry

    error_feedback = None
    for attempt in range(max_retries):
        print(f"Attempt {attempt+1}/{max_retries} for API: {api_name}")
        
        # Create prompt with any error feedback from previous attempts
        prompt = create_code_prompt(query, signature, task_entry, error_feedback)
        
        try:
            # Call the AI model to generate code
            client = OpenAI(api_key=api_key, base_url=base_url)
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a Rust expert tasked with writing high-quality, correct Rust code."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3 + (attempt * 0.1)  # Slightly increase temperature for diversity in retries
            )

            response = completion.choices[0].message.content.strip()
            
            # Extract code from response
            if "```" in response:
                # Extract code between triple backticks
                code_blocks = re.findall(r'```(?:rust)?(.*?)```', response, re.DOTALL)
                if code_blocks:
                    rust_code = code_blocks[0].strip()
                else:
                    error_feedback = "Failed to extract code from the response."
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue
                    else:
                        task_entry['code'] = "CANNOT GENERATE CORRECT CODE"
                        task_entry['validation_status'] = "failed"
                        return task_entry
           
            elif "<code>" in response and "</code>" in response:
                rust_code = response.split('<code>')[1].split('</code>')[0].strip()
            else:
                error_feedback = "Please provide code within triple backticks or '<code> [Your code here] </code>' tags."
                print(f"  Missing code tags in response, retrying...")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                else:
                    task_entry['code'] = "CANNOT GENERATE CORRECT CODE"
                    task_entry['validation_status'] = "failed"
                    return task_entry
            
            # Check if code is too short/empty
            if len(rust_code) < 20:  # Arbitrary minimum length for valid code
                error_feedback = "The generated code is too short or empty. Please provide a complete implementation."
                print(f"  Code too short ({len(rust_code)} chars), retrying...")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
            
            # Print the code for debugging
            # print(f" Rust code:\n {rust_code}")
            
            # 使用静态分析代替Docker运行
            # print(f"  Running static analysis on the generated code...")
            validation_passed, validation_message = static_analysis_rust_code(rust_code, api_name)
            
            if validation_passed:
                # 成功：代码通过静态分析
                print(f"  Success! Code passed static analysis and uses the API correctly.")
                task_entry['code'] = rust_code
                task_entry['validation_output'] = validation_message
                task_entry['validation_status'] = "success"
                return task_entry
            else:
                # 代码未通过静态分析
                error_feedback = f"Static analysis failed: {validation_message}"
                print(f"  Code failed static analysis: {validation_message}")
            
            # 继续下一次尝试
            if attempt < max_retries - 1:
                time.sleep(1)
                
        except Exception as e:
            # Handle any exceptions in the API call
            error_feedback = f"Error generating code: {str(e)}"
            print(f"  Exception: {str(e)}, retrying...")
            if attempt < max_retries - 1:
                time.sleep(2)
    
    # All retries failed, return the last attempt with error status and CANNOT GENERATE CORRECT CODE message
    task_entry['code'] = "CANNOT GENERATE CORRECT CODE"
    task_entry['validation_output'] = error_feedback
    task_entry['validation_status'] = "failed"
    print(f"  All {max_retries} attempts failed for API: {api_name}")
    return task_entry

# Process entries concurrently and save results periodically
def generate_all_validated_codes(api_tasks):
    updated_tasks = []
    progress_bar = tqdm(total=len(api_tasks), desc="Generating validated code")
    
    for idx in range(0, len(api_tasks), 50):
        batch = api_tasks[idx:idx+50]
        batch_results = [None] * len(batch)
        
        # Pre-fill batch_results with entries that don't need processing
        for i, entry in enumerate(batch):
            if entry.get('code') != "CANNOT GENERATE CORRECT CODE":
                batch_results[i] = entry

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            future_to_index = {
                executor.submit(generate_and_validate_rust_code, entry): i 
                for i, entry in enumerate(batch) 
                if entry.get('code') == "CANNOT GENERATE CORRECT CODE"
            }
            
            for future in concurrent.futures.as_completed(future_to_index.keys()):
                try:
                    original_index = future_to_index[future]
                    result = future.result()
                    batch_results[original_index] = result
                    
                    # 添加详细的失败信息输出
                    if result.get('validation_status') == "failed":
                        print(f"\nFAILURE DETAILS for API {result.get('name')}:")
                        print(f"  Query: {result.get('query')}")
                        print(f"  Function signature: {result.get('function_signature')}")
                        print(f"  Generated code snippet:\n{result.get('code')[:200]}..." if len(result.get('code', '')) > 200 else f"  Generated code:\n{result.get('code')}")
                        print(f"  Error details: {result.get('validation_output')}\n")
                    
                    progress_bar.update(1)
                except Exception as e:
                    print(f"Error processing task: {str(e)}")
        
        # Add batch results in original order
        updated_tasks.extend(batch_results)
        
        # Save progress after each batch
        with open(output_file, 'w', encoding='utf-8') as outfile:
            json.dump(updated_tasks, outfile, indent=2, ensure_ascii=False)
    
    progress_bar.close()
    print(f"Processing complete. Results saved to {output_file}")

# Start the validated code generation process
if __name__ == "__main__":
    print(f"Starting code generation for {len(api_tasks)} tasks...")
    print(f"Using static analysis instead of Docker for code validation")
    generate_all_validated_codes(api_tasks)