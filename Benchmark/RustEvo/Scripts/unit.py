def generate_task(api_entry, max_retries=3):
    from openai import OpenAI
    api_key = ''
    base_url = ''
    model = 'deepseek-v3'
    # Define prompts based on change_type
    prompt_templates = {
        "stabilized": "As a Rust expert, generate a concise programming task and corresponding Rust function signature that implicitly requires the use of a recently stabilized standard library API. ### Core Principles ###  Follow these rules: Do not directly mention the API's name or signature. Clearly describe a real-world scenario solved uniquely or efficiently by this newly stabilized feature. The generated function signature should strongly hint at the use of the newly stabilized API. Be unpredictable in your task format (e.g., don't always begin with 'You ...'. Instead, use more variable format including Interrogative, Imperative, Conditional, Declarative, Challenge-based, Scenario-based and etc. ).",
        "signature": "As a Rust expert, generate a concise programming task and corresponding Rust function signature that implicitly requires using an API whose signature was recently updated. ### Core Principles ### Follow these rules: Do not directly reveal the name or signature changes of the API. Provide a realistic context emphasizing limitations or inconveniences solved by the updated API. The generated function signature should implicitly lead to choosing the newly updated API over older versions. Be unpredictable in your task format (e.g., don't always begin with 'You ...'. Instead, use more variable format including Interrogative, Imperative, Conditional, Declarative, Challenge-based, Scenario-based and etc. ).",
        "implicit": "As a Rust expert, generate a concise programming task and corresponding Rust function signature that implicitly requires an API whose internal implementation or functionality recently changed without altering its signature. ### Core Principles ### Follow these rules: Do not directly mention the API name or its internal changes explicitly. Clearly describe an observable behavior improvement (e.g., performance, memory usage, correctness). The generated function signature and context should implicitly prompt the use of this API, relying on documentation or inferred behavior differences. Be unpredictable in your task format.Be unpredictable in your task format (e.g., don't always begin with 'You ...'. Instead, use more variable format including Interrogative, Imperative, Conditional, Declarative, Challenge-based, Scenario-based and etc. ).",
        "deprecated": "As a Rust expert, generate a concise programming task and corresponding Rust function signature that implicitly encourages replacing a deprecated or replaced API with a newly recommended one. ### Core Principles ### Follow these rules: Do not explicitly mention the deprecated API's name. Provide context that strongly suggests the disadvantages (performance, safety, usability) of the old method. The generated function signature should implicitly guide users towards the recommended alternative. Be unpredictable in your task format (e.g., don't always begin with 'You ...'. Instead, use more variable format including Interrogative, Imperative, Conditional, Declarative, Challenge-based, Scenario-based and etc. ).",
    }

    # Function to generate query and signature using OpenAI

    import time
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
                print(f"Attempt {attempt + 1}: Failed to extract query. Retrying...")
                if attempt < max_retries - 1:
                    time.sleep(1)  # 避免过于频繁的API调用
                    continue

            if "<signature>" in response and "</signature>" in response:
                signature = response.split('<signature>')[1].split('</signature>')[0].strip()
            else:
                print(f"Attempt {attempt + 1}: Failed to extract signature. Retrying...")
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
            print(f"Attempt {attempt + 1}: Error: {str(e)}. Retrying...")
            if attempt < max_retries - 1:
                time.sleep(2)  # 网络错误时等待更长时间

    # 所有重试都失败后的处理
    print(f"All {max_retries} attempts failed for API: {api_entry.get('name', 'unknown')}")
    api_entry['query'] = "ERROR: Failed to generate query after multiple attempts"
    api_entry['function_signature'] = "ERROR: Failed to generate signature after multiple attempts"
    return api_entry

# Prompt template for code generation with error feedback
def create_code_prompt(query, signature, api_details, error_feedback=None):
    rust_version = api_details.get('to_version', 'latest')
    # rust_version = '1.84.0'
    api_name = api_details.get('name', 'the specified API')
    api_detail = {
    'crate': api_details.get('crate', ''),
    'api_name': api_details.get('name', ''),
    # 'crate_version': api_details.get('to_version', ''),
    'module': api_details.get('module', {}),
    'signature': api_details.get('signature', ''),
    'documentation': api_details.get('documentation', ''),
    'source_code': api_details.get('source_code', ''),
    'rust_version': api_details.get('to_version', ''),
    'examples': api_details.get('examples', '')
}

    base_prompt = (f"Given the following Rust programming task:\n"
                   f"{query}\n\n"
                   f"Implement the following function signature:\n"
                   f"{signature}\n\n"
                   f"### Important rules ###\n"
                   f"- Your implementation MUST use '{api_name}'.\n"
                   f"- Your implementation must compile and run correctly on Rust {rust_version}.\n"
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
        (r';', "Missing semicolons, possible syntax error")
    ]

    for pattern, error in syntax_checks:
        if not re.search(pattern, code):
            return False, error

    # 检查明显的语法错误 - 未闭合的引号、括号等
    # 首先删除Rust生命周期标记，避免误判
    # 替换类似 <'a> 或 &'a 的生命周期标记
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

    # 检查一些常见Rust错误
    if "unwrap()" in code and not "Result" in code and not "Option" in code:
        # 这不是确定性错误，但可能表明代码存在问题
        print(f"  Warning: Code uses unwrap() but does not seem to handle Result/Option properly")

    # 所有检查都通过
    return True, "Static analysis passed"


# Function to iteratively generate and validate Rust code
# 更新后的代码生成函数

def generate_and_validate_rust_code(task_entry, max_retries=3):
    from openai import OpenAI
    import re
    import time
    api_key = 'sk-z7CzOyi3VfIQHv8xkEo3D6rb7jAIApdVtmUSmktU9SlIbBFa'
    base_url = 'https://api.agicto.cn/v1'
    model = 'deepseek-v3'

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
        print(f"Attempt {attempt + 1}/{max_retries} for API: {api_name}")

        # Create prompt with any error feedback from previous attempts
        prompt = create_code_prompt(query, signature, task_entry, error_feedback)

        try:
            # Call the AI model to generate code
            client = OpenAI(api_key=api_key, base_url=base_url)
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system",
                     "content": "You are a Rust expert tasked with writing high-quality, correct Rust code."},
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


import json
import os
import subprocess
import tempfile
import time
import concurrent.futures
import re
from typing import Dict, List, Any, Tuple

from tqdm import tqdm
from openai import OpenAI

# Configure OpenAI API
api_key = 'sk-9ouhFf2ESssqkkpVOMZLZLWqIrflALwmLQ62Jr4jJfn7xAdD'
base_url = 'https://api.agicto.cn/v1'
# model = 'claude-3-5-sonnet-20241022'
model = 'claude-3-7-sonnet-20250219'


# model = 'deepseek-v3'

def check_function_in_code(code: str, signature: str) -> bool:
    """检查签名是否存在于代码中。"""
    # 移除前后空白
    signature = signature.strip()

    # 从签名中提取函数名
    fn_name_match = re.search(r'fn\s+([a-zA-Z0-9_]+)', signature)
    if not fn_name_match:
        return False

    fn_name = fn_name_match.group(1)

    # 检查代码中是否包含函数名和完整函数体
    # 这种方法更可靠，能处理各种复杂情况
    basic_pattern = r'fn\s+' + re.escape(fn_name) + r'\s*\([^{]*{'

    # 如果找到完整的函数定义，直接返回True
    if re.search(basic_pattern, code):
        return True

    # 否则尝试更严格的匹配
    # 从签名中提取参数部分
    params_match = re.search(r'fn\s+[a-zA-Z0-9_]+\s*\(([^)]*)\)', signature)
    if not params_match:
        return False

    params = params_match.group(1).strip()

    # 构建严格的匹配模式，包含函数名和参数
    strict_pattern = r'fn\s+' + re.escape(fn_name) + r'\s*\(\s*' + re.escape(params) + r'\s*\)'

    return bool(re.search(strict_pattern, code))


def generate_test_code(code: str, signature: str, query: str, max_retries: int = 3) -> str:
    """Generate test code using OpenAI model."""

    prompt = f"""
You are an expert Rust developer tasked with writing tests for a Rust function.

Here is the function signature:
```rust
{signature}
```

Here is the query describing the function:
{query}

Here is the function code:
```rust
{code}
```

Your task is to write Rust test code that thoroughly tests this function.
DO NOT include the original function code in your response.
ONLY provide the test code that would be placed in a test module.

IMPORTANT GUIDELINES FOR WRITING TESTS:
1. Make sure to include all necessary imports and crates at the top of your test
2. Use only stable Rust features compatible with Rust 1.71.0
3. When testing generic functions, provide concrete type implementations
4. For functions with complex types (like NonZeroI32), properly handle construction and error cases
5. Make sure your test references match the function signature exactly
6. For testing functions with trait bounds, implement simple test structs that satisfy those bounds
7. Do not use non-deterministic functions like random number generation in tests without fixed seeds

Respond ONLY with the Rust test code, nothing else.
"""
    client = OpenAI(api_key=api_key, base_url=base_url)
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Error getting OpenAI response, retrying ({attempt + 1}/{max_retries}): {e}")
                time.sleep(2)  # Adding a delay before retrying
            else:
                print(f"Failed to get response from OpenAI after {max_retries} attempts: {e}")
                return ""

    return ""


def create_test_file(code: str, test_code: str, crate_info: Dict[str, str] = None) -> str:
    """Combine original code and test code into a complete test file with optional crate dependencies."""

    crate_section = ""
    if crate_info:
        crate_lines = []
        for crate_name, crate_version in crate_info.items():
            crate_lines.append(f'{crate_name} = "{crate_version}"')

        if crate_lines:
            crate_section = f"""
[package]
name = "rust_test"
version = "0.1.0"
edition = "2021"

[dependencies]
{chr(10).join(crate_lines)}
"""

    if crate_section:
        # If we have crates, we need to create a proper Cargo.toml structure
        return f"""// Cargo.toml content:
/*
{crate_section}
*/

// src/lib.rs content:
{code}

#[cfg(test)]
mod tests {{
    use super::*;

{test_code}
}}
"""
    else:
        # Simple test file without dependencies
        return f"""
{code}

#[cfg(test)]
mod tests {{
    use super::*;

{test_code}
}}
"""


def run_rust_test(test_file_content: str, rust_version: str, crate_info: Dict[str, str] = None,
                  item_index: int = None, rust_files_dir: str = None) -> Tuple[bool, str, str]:
    """运行Rust测试文件，修复Windows路径问题"""
    if rust_files_dir and item_index is not None:
        os.makedirs(rust_files_dir, exist_ok=True)
        rust_file_path = os.path.join(rust_files_dir, f"test_{item_index:05d}.rs")
        with open(rust_file_path, 'w', encoding='utf-8') as f:
            f.write(test_file_content)

    # 如果有crate依赖，需要创建cargo项目
    if crate_info:
        # 创建临时cargo项目
        with tempfile.TemporaryDirectory() as temp_dir:
            # 提取Cargo.toml内容
            cargo_toml_match = re.search(r'\/\/ Cargo\.toml content:\n\/\*\n(.*?)\*\/',
                                         test_file_content, re.DOTALL)
            if not cargo_toml_match:
                return False, "Failed to extract Cargo.toml content from test file", ""

            cargo_toml_content = cargo_toml_match.group(1)

            # 提取lib.rs内容
            lib_rs_match = re.search(r'\/\/ src\/lib\.rs content:\n(.*)',
                                     test_file_content, re.DOTALL)
            if not lib_rs_match:
                return False, "Failed to extract lib.rs content from test file", ""

            lib_rs_content = lib_rs_match.group(1)

            # 创建Cargo.toml
            cargo_toml_path = os.path.join(temp_dir, "Cargo.toml")
            with open(cargo_toml_path, 'w', encoding='utf-8') as f:
                f.write(cargo_toml_content)

            # 创建src目录和lib.rs
            os.makedirs(os.path.join(temp_dir, "src"), exist_ok=True)
            lib_rs_path = os.path.join(temp_dir, "src", "lib.rs")
            with open(lib_rs_path, 'w', encoding='utf-8') as f:
                f.write(lib_rs_content)

            # 运行cargo test（修复Windows路径问题）
            try:
                # 不使用shell=True，并引用路径
                cmd = f'cd "{temp_dir}" && rustup run {rust_version} cargo test'
                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=60
                )

                # 获取详细输出用于日志
                detailed_output = f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"

                if result.returncode != 0:
                    return False, f"Test execution error with cargo", detailed_output

                return True, "", detailed_output
            except subprocess.TimeoutExpired:
                return False, "Timeout: Test took too long to execute", "TIMEOUT"
            except Exception as e:
                return False, f"Error running cargo tests: {str(e)}", str(e)
    else:
        # 没有依赖的常规rustc编译
        with tempfile.NamedTemporaryFile(suffix='.rs', delete=False) as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(test_file_content.encode('utf-8'))

        try:
            # 移除扩展名并添加新扩展名
            base_path = os.path.splitext(temp_file_path)[0]
            output_path = f"{base_path}.exe"  # Windows可执行文件

            # 修改编译命令，正确引用路径
            compile_cmd = f'rustup run {rust_version} rustc --test "{temp_file_path}" -o "{output_path}"'
            compile_result = subprocess.run(
                compile_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )

            # 获取详细输出用于日志
            detailed_output = f"COMPILATION STDOUT:\n{compile_result.stdout}\n\nCOMPILATION STDERR:\n{compile_result.stderr}"

            if compile_result.returncode != 0:
                return False, f"Compilation error", detailed_output

            # 运行编译后的测试
            test_cmd = f'"{output_path}"'
            test_result = subprocess.run(
                test_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )

            # 添加测试执行输出
            detailed_output += f"\n\nTEST EXECUTION STDOUT:\n{test_result.stdout}\n\nTEST EXECUTION STDERR:\n{test_result.stderr}"

            if test_result.returncode != 0:
                return False, f"Test execution error", detailed_output

            return True, "", detailed_output
        except subprocess.TimeoutExpired:
            return False, "Timeout: Test took too long to execute", "TIMEOUT"
        except Exception as e:
            return False, f"Error running tests: {str(e)}", str(e)
        finally:
            # 清理临时文件
            try:
                os.remove(temp_file_path)
                if os.path.exists(output_path):
                    os.remove(output_path)
            except:
                pass


def fix_test_with_feedback(code: str, test_code: str, error_message: str, signature: str, query: str,
                           detailed_output: str) -> str:
    """Send error feedback to OpenAI to get improved test code."""

    error_lines = []
    if "error" in detailed_output.lower():
        error_pattern = r'error(\[.*?\])?: (.*?)(?=\n)'
        error_matches = re.findall(error_pattern, detailed_output)
        if error_matches:
            error_lines = [f"{match[1].strip()}" for match in error_matches[:3]]  # 仅取前3个错误

    error_feedback = "\n".join(error_lines)

    prompt = f"""
You are an expert Rust developer tasked with fixing failing test code.

Original function signature:
```rust
{signature}
```

Function description:
{query}

Function code:
```rust
{code}
```

Original test code:
```rust
{test_code}
```

Compilation/test errors:
```
{error_feedback}
```

Full detailed output (showing exact errors):
```
{error_message}
```

Please fix the test code to address the specific errors. Pay special attention to:
1. Making sure all imports are correct
2. Types match exactly with the function signature
3. Test values are valid for the expected types
4. Properly handling expected errors or panics
5. Using correct syntax for the testing Rust version (1.71.0)

DO NOT include the original function implementation in your response.
ONLY provide the corrected test code.
Respond ONLY with the fixed Rust test code, nothing else.
"""
    client = OpenAI(api_key=api_key, base_url=base_url)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error getting OpenAI fix response: {e}")
        return test_code  # Return original if we can't get a fix


def parse_crate_info(item: Dict[str, Any]) -> Dict[str, str]:
    """Parse crate information from the item dictionary."""
    crate_info = {}

    if "crate" in item:
        # If it's a string, assume it's the crate name
        if isinstance(item["crate"], str):
            crate_name = item["crate"]
            # Get version from to_version or default to latest
            crate_version = item.get("to_version", "*")
            crate_info[crate_name] = crate_version
        # If it's a dict, it might contain multiple crates
        elif isinstance(item["crate"], dict):
            for crate_name, crate_data in item["crate"].items():
                if isinstance(crate_data, str):
                    crate_info[crate_name] = crate_data
                elif isinstance(crate_data, dict) and "version" in crate_data:
                    crate_info[crate_name] = crate_data["version"]

    return crate_info


def get_installed_toolchains():
    """获取系统上已安装的Rust工具链列表"""
    try:
        result = subprocess.run(
            ["rustup", "toolchain", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            print(f"获取工具链列表失败: {result.stderr}")
            return ["stable"]  # 默认回退到stable

        # 解析输出获取已安装的工具链
        toolchains = []
        for line in result.stdout.strip().split('\n'):
            # 每行格式通常是 "1.76.0-x86_64-pc-windows-msvc (default)"
            # 或 "stable-x86_64-pc-windows-msvc (default)"
            match = re.search(r'^([^\s]+)', line)
            if match:
                toolchains.append(match.group(1))

        if not toolchains:
            print("未找到已安装的工具链，使用stable")
            return ["stable"]

        return toolchains
    except Exception as e:
        print(f"获取工具链时出错: {e}")
        return ["stable"]  # 出错时回退到stable


def find_best_toolchain(requested_version, installed_toolchains):
    """根据请求的版本寻找最佳可用工具链"""
    # 如果请求的版本已安装，直接使用
    if requested_version in installed_toolchains:
        return requested_version

    # 如果请求了stable，nightly或beta，并且该版本已安装
    if requested_version in ["stable", "nightly", "beta"]:
        for toolchain in installed_toolchains:
            if toolchain.startswith(requested_version):
                return toolchain

    # 如果请求了特定版本，查找匹配的已安装版本
    if re.match(r'^\d+\.\d+\.\d+$', requested_version):
        requested_major, requested_minor, requested_patch = map(int, requested_version.split('.'))

        # 尝试找到最接近的已安装版本
        best_match = None
        min_diff = float('inf')

        for toolchain in installed_toolchains:
            # 提取版本号
            version_match = re.match(r'^(\d+)\.(\d+)\.(\d+)', toolchain)
            if not version_match:
                continue

            major, minor, patch = map(int, version_match.groups())

            # 计算版本差异 (优先匹配主要版本号)
            major_diff = abs(major - requested_major) * 10000
            minor_diff = abs(minor - requested_minor) * 100
            patch_diff = abs(patch - requested_patch)
            total_diff = major_diff + minor_diff + patch_diff

            if total_diff < min_diff:
                min_diff = total_diff
                best_match = toolchain

        if best_match:
            return best_match

    # 如果都没有匹配，回退到stable
    for toolchain in installed_toolchains:
        if toolchain.startswith("stable"):
            return toolchain

    # 最后回退到任何可用的工具链
    return installed_toolchains[0] if installed_toolchains else "stable"


def process_item(item: Dict[str, Any], item_index: int, rust_files_dir: str = None) -> Dict[str, Any]:
    """Process a single dataset item and return the updated item."""
    code = item.get("code", "")
    signature = item.get("function_signature", "")
    query = item.get("query", "")

    print(f"Processing item {item_index}: {signature[:50]}...")

    # 检查函数是否存在于代码中
    if not check_function_in_code(code, signature):
        print(f"Item {item_index}: INCORRECT CODE - Function not found in code")
        item["test_program"] = "INCORRECT CODE"
        return item

    # 解析crate信息
    crate_info = parse_crate_info(item)

    # 确定Rust版本 - 遵循原始逻辑
    if crate_info:
        rust_version = "1.84.0"  # 固定版本用于有crate的项目

        # 确保crate依赖已下载
        if not ensure_crate_downloaded(crate_info):
            print(f"Item {item_index}: WARNING - 无法下载所需crate依赖")
            # 继续尝试，万一本地已经有了
    else:
        # 如果没有crate，to_version是Rust版本
        requested_version = item.get("to_version", "stable")

        # 获取已安装的工具链列表
        installed_toolchains = get_installed_toolchains()
        rust_version = find_best_toolchain(requested_version, installed_toolchains)

    print(f"Item {item_index}: Using Rust version {rust_version}")
    if crate_info:
        print(f"Item {item_index}: Using crates: {crate_info}")

    # 生成初始测试代码
    test_code = generate_test_code(code, signature, query)

    # Try to run tests with retries
    max_attempts = 5
    success = False

    for attempt in range(max_attempts):
        print(f"Item {item_index}: Test attempt {attempt + 1}/{max_attempts}")

        full_test_file = create_test_file(code, test_code, crate_info)
        success, error_message, detailed_output = run_rust_test(
            full_test_file, rust_version, crate_info,
            item_index=item_index,
            rust_files_dir=rust_files_dir
        )

        # Log the detailed output to a file for debugging
        if rust_files_dir:
            log_dir = os.path.join(rust_files_dir, "logs")
            os.makedirs(log_dir, exist_ok=True)
            with open(os.path.join(log_dir, f"test_{item_index:05d}_attempt_{attempt + 1}.log"), 'w',
                      encoding='utf-8') as f:
                f.write(
                    f"ATTEMPT: {attempt + 1}\nSUCCESS: {success}\nERROR: {error_message}\n\nDETAILED OUTPUT:\n{detailed_output}")

        if success:
            print(f"Item {item_index}: Test SUCCESSFUL on attempt {attempt + 1}")
            break

        print(f"Item {item_index}: Test FAILED on attempt {attempt + 1}: {error_message}")

        if attempt < max_attempts - 1:
            # Get improved test code with error feedback
            test_code = fix_test_with_feedback(code, test_code, error_message, signature, query, detailed_output)

    # Store the result
    if success:
        item["test_program"] = test_code
    else:
        item["test_program"] = "INCORRECT TEST"

    return item


def ensure_crate_downloaded(crate_info: Dict[str, str]) -> bool:
    """
    确保指定的crate依赖已下载
    只在用到时才下载，不进行预先批量下载

    参数:
    crate_info: 包含crate名称和版本的字典

    返回:
    bool: 下载成功为True，否则为False
    """
    if not crate_info:
        return True  # 没有crate需要下载

    print(f"正在确保crate依赖可用: {crate_info}")

    # 对于有crate依赖的项目，使用固定的Rust版本
    rust_version = "1.84.0"  # 原始代码中对有crate的项目使用固定版本

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # 初始化临时Cargo项目
            init_cmd = f'cd "{temp_dir}" && rustup run {rust_version} cargo init --lib'
            init_result = subprocess.run(
                init_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )

            if init_result.returncode != 0:
                print(f"创建临时Cargo项目失败: {init_result.stderr}")
                return False

            # 读取并修改Cargo.toml
            cargo_toml_path = os.path.join(temp_dir, "Cargo.toml")
            with open(cargo_toml_path, 'r', encoding='utf-8') as f:
                cargo_content = f.read()

            with open(cargo_toml_path, 'w', encoding='utf-8') as f:
                f.write(cargo_content)
                f.write("\n[dependencies]\n")
                for crate_name, crate_version in crate_info.items():
                    f.write(f'{crate_name} = "{crate_version}"\n')

            # 下载依赖（但不编译）
            check_cmd = f'cd "{temp_dir}" && rustup run {rust_version} cargo fetch --quiet'
            check_result = subprocess.run(
                check_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=120
            )

            if check_result.returncode != 0:
                print(f"下载crate依赖失败: {check_result.stderr}")
                return False

            print(f"Crate依赖下载成功!")
            return True

        except Exception as e:
            print(f"下载依赖时出错: {str(e)}")
            return False
