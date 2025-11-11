"""
check函数签名不对之后会尝试重新generate
python scripts/supplement_test.py --input_file ./data/codes/codes_rust_filtered.json --output_file ./data/test_programs/0312regenerate_rust.json --comparison_file ./data/dataset/final_docs.json --rust_files_dir ./rust_tests --max_workers 8
"""

import json
import os
import subprocess
import tempfile
import time
import concurrent.futures
import re
from typing import Dict, List, Any, Tuple, Optional

from tqdm import tqdm
from openai import OpenAI

# Configure OpenAI API
api_key = ''
base_url = ''
model = 'deepseek-v3'

# 存储已处理过的项目的缓存
processed_items_cache = {}

def create_code_prompt(query, signature, api_details, error_feedback=None):
    rust_version = api_details.get('to_version', 'latest')
    # rust_version = '1.84.0'
    api_name = api_details.get('name', 'the specified API')
    api_detail = {
    # 'crate': api_details.get('crate', ''),
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

def generate_rust_code(query: str, signature: str, api_details: Dict[str, Any], error_feedback: str = None) -> str:
    """使用OpenAI模型生成包含正确签名的Rust代码。"""
    prompt = create_code_prompt(query, signature, api_details, error_feedback)
    
    client = OpenAI(api_key=api_key, base_url=base_url)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        code = response.choices[0].message.content.strip()
        print(f"\n==== 原始响应 ====\n{code[:300]}...\n") 
        
        # 提取<code>标签内的代码
        code_pattern = r'<code>([\s\S]*?)</code>'
        code_match = re.search(code_pattern, code)
        
        if code_match:
            extracted_code = code_match.group(1).strip()
            print(f"\n==== 提取的代码 ====\n{extracted_code[:300]}...\n")
            return extracted_code
        else:
            # 如果没有特定的标签格式，寻找代码块
            code_block_pattern = r'```(?:rust)?\s*([\s\S]*?)```'
            block_match = re.findall(code_block_pattern, code)
            
            if block_match:
                extracted_code = '\n'.join(block_match).strip()
                print(f"\n==== 提取的代码块 ====\n{extracted_code[:300]}...\n")
                return extracted_code
            
            # 如果既没有标签也没有代码块，返回整个响应
            return code
    except Exception as e:
        print(f"生成Rust代码时出错: {e}")
        return ""

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


def load_comparison_json(comparison_file: str) -> Dict[str, List[Dict[str, Any]]]:
    """加载比较用的JSON文件并创建索引以便快速查找。"""
    if not os.path.exists(comparison_file):
        print(f"比较文件 {comparison_file} 不存在，将处理所有项目。")
        return {}
    
    try:
        with open(comparison_file, 'r', encoding='utf-8') as f:
            comparison_data = json.load(f)
        
        # 创建索引结构
        index = {}
        for item in comparison_data:
            name = item.get("name")
            module = item.get("module")
            
            if name and module:
                # 使用name和module的组合作为键
                key = f"{name}:{module}"
                if key not in index:
                    index[key] = []
                index[key].append(item)
        
        return index
    except Exception as e:
        print(f"加载比较文件时出错: {e}")
        return {}

# def should_process_item(item: Dict[str, Any], comparison_index: Dict[str, List[Dict[str, Any]]]) -> bool:
#     """检查项目是否应该被处理（如果在比较文件中不存在）。"""
#     if not comparison_index:
#         return True  # 如果没有比较索引，处理所有项目
    
#     name = item.get("name")
#     module = item.get("module")
    
#     if not name or not module:
#         return True  # 如果项目缺少name或module字段，处理它
    
#     # 检查组合键是否存在于索引中
#     key = f"{name}:{module}"
#     return key not in comparison_index

def should_process_item(item: Dict[str, Any], comparison_index: Dict[str, List[Dict[str, Any]]]) -> bool:
    """检查项目是否应该被处理（如果在比较文件中不存在）。"""
    # 如果项目已经有test_program字段，不需要处理
    if "test_program" in item:
        return False
        
    if not comparison_index:
        return True  # 如果没有比较索引，处理所有项目
    
    name = item.get("name")
    module = item.get("module")
    
    if not name or not module:
        return True  # 如果项目缺少name或module字段，处理它
    
    # 检查组合键是否存在于索引中
    key = f"{name}:{module}"
    return key not in comparison_index

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
            test_code = response.choices[0].message.content.strip()
            
            # Remove markdown code block delimiters if they exist
            if test_code.startswith("```"):
                # Extract content between code blocks
                pattern = r"```(?:rust)?\s*([\s\S]*?)```"
                matches = re.findall(pattern, test_code)
                if matches:
                    test_code = "\n".join(matches)
                else:
                    # If regex didn't work, try a simple trim approach
                    test_code = test_code.replace("```rust", "").replace("```", "").strip()
            
            return test_code
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Error getting OpenAI response, retrying ({attempt+1}/{max_retries}): {e}")
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

def fix_test_with_feedback(code: str, test_code: str, error_message: str, signature: str, query: str, detailed_output: str) -> str:
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
        fixed_code = response.choices[0].message.content.strip()
        
        # Remove markdown code block delimiters if they exist
        # 首先移除任何前导的解释文本，只保留代码块内容
        pattern = r"```(?:rust)?\s*([\s\S]*?)```"
        matches = re.findall(pattern, fixed_code)
        if matches:
            fixed_code = "\n".join(matches)
        else:
            # 如果没有代码块，尝试更严格地清理内容
            # 删除可能包含解释性文本的行
            lines = fixed_code.split('\n')
            code_lines = []
            in_code = False
            for line in lines:
                if line.strip().startswith("```"):
                    in_code = not in_code
                    continue
                if in_code or not (line.startswith("I ") or line.startswith("Let me ") or "`" in line):
                    code_lines.append(line)
            fixed_code = "\n".join(code_lines).strip()

        
        return fixed_code
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

def process_item(item: Dict[str, Any], item_index: int, comparison_index: Dict[str, List[Dict[str, Any]]],
                rust_files_dir: str = None) -> Dict[str, Any]:
    """处理单个数据项并返回更新后的项目。"""
    
    # 检查是否应该处理该项目
    if not should_process_item(item, comparison_index):
        print(f"跳过项目 {item_index}: 在比较文件中已存在相同的name和module")
        item["test_program"] = "SKIPPED"
        return item
    
    code = item.get("code", "")
    signature = item.get("function_signature", "")
    query = item.get("query", "")
    
    print(f"处理项目 {item_index}: {signature[:50]}...")
    
    # 检查函数是否存在于代码中，如果不存在则尝试生成正确的代码
    if not check_function_in_code(code, signature):
        print(f"项目 {item_index}: 未找到函数签名，尝试生成正确的代码")
        
        # 最多尝试3次生成正确的代码
        max_gen_attempts = 3
        for gen_attempt in range(max_gen_attempts):
            # 生成代码
            generated_code = generate_rust_code(query, signature, item)
            
            if generated_code and check_function_in_code(generated_code, signature):
                print(f"项目 {item_index}: 成功生成包含正确签名的代码 (尝试 {gen_attempt+1}/{max_gen_attempts})")
                code = generated_code
                # 更新项目中的代码
                item["code"] = code
                break
            else:
                error_feedback = "生成的代码必须指定的函数签名，请确保完全按照签名要求实现函数。"
                print(f"项目 {item_index}: 生成的代码不正确，重试 ({gen_attempt+1}/{max_gen_attempts})")
        
        # 如果尝试多次后仍未成功生成正确的代码
        if not check_function_in_code(code, signature):
            print(f"项目 {item_index}: 无法生成正确的代码，跳过该项目")
            item["test_program"] = "INCORRECT CODE"
            return item
    
    # 解析crate信息
    crate_info = parse_crate_info(item)
    
    # 确定Rust版本
    if crate_info:
        rust_version = "1.84.0"  # 固定版本用于有crate的项目
        
        # 确保crate依赖已下载
        if not ensure_crate_downloaded(crate_info):
            print(f"项目 {item_index}: 警告 - 无法下载所需crate依赖")
            # 继续尝试，万一本地已经有了
    else:
        # 如果没有crate，to_version是Rust版本
        requested_version = item.get("to_version", "stable")
        
        # 获取已安装的工具链列表
        installed_toolchains = get_installed_toolchains()
        rust_version = find_best_toolchain(requested_version, installed_toolchains)
    
    print(f"项目 {item_index}: 使用Rust版本 {rust_version}")
    if crate_info:
        print(f"项目 {item_index}: 使用crates: {crate_info}")
    
    # 生成初始测试代码
    test_code = generate_test_code(code, signature, query)
    
    # 尝试运行测试，有重试机制
    max_attempts = 5
    success = False
    
    for attempt in range(max_attempts):
        print(f"项目 {item_index}: 测试尝试 {attempt+1}/{max_attempts}")
        
        full_test_file = create_test_file(code, test_code, crate_info)
        success, error_message, detailed_output = run_rust_test(
            full_test_file, rust_version, crate_info,
            item_index=item_index, 
            rust_files_dir=rust_files_dir
        )
        
        # 将详细输出记录到文件以便调试
        if rust_files_dir:
            log_dir = os.path.join(rust_files_dir, "logs")
            os.makedirs(log_dir, exist_ok=True)
            with open(os.path.join(log_dir, f"test_{item_index:05d}_attempt_{attempt+1}.log"), 'w', encoding='utf-8') as f:
                f.write(f"ATTEMPT: {attempt+1}\nSUCCESS: {success}\nERROR: {error_message}\n\nDETAILED OUTPUT:\n{detailed_output}")
        
        if success:
            print(f"项目 {item_index}: 测试成功，尝试次数 {attempt+1}")
            break
        
        print(f"项目 {item_index}: 测试失败，尝试次数 {attempt+1}: {error_message}")
        
        if attempt < max_attempts - 1:
            # 使用错误反馈获取改进的测试代码
            test_code = fix_test_with_feedback(code, test_code, error_message, signature, query, detailed_output)
    
    # 存储结果
    if success:
        item["test_program"] = test_code
    else:
        item["test_program"] = "INCORRECT TEST"
    
    return item

def process_dataset_concurrent(input_file: str, output_file: str, comparison_file: str = None,
                              rust_files_dir: str = None, max_workers: int = 4) -> None:
    """使用并发工作线程处理数据集，同时保持顺序。"""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 加载比较文件（如果提供）
    comparison_index = load_comparison_json(comparison_file)
    
    total_items = len(data)
    results = [None] * total_items  # 预分配结果列表以保持顺序
    
    # 创建进度条
    progress_bar = tqdm(total=total_items, desc="处理Rust代码样本")
    
    # 用于跟踪进度的计数器
    completed_count = 0
    incorrect_code_count = 0
    incorrect_test_count = 0
    skipped_count = 0
    already_has_test_count = 0
    
    # 并发处理项目，同时保持顺序
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务并保存futures及其原始索引
        future_to_idx = {}
        
        # 只为需要处理的项目创建任务
        for idx, item in enumerate(data):
            if "test_program" in item:
                # 直接保留原项目，不做任何修改
                results[idx] = item.copy()
                already_has_test_count += 1
                progress_bar.update(1)            
            # 检查是否需要处理此项目
            elif should_process_item(item, comparison_index):
                future = executor.submit(process_item, item, idx, comparison_index, rust_files_dir)
                future_to_idx[future] = idx
            else:
                # 如果不需要处理，直接标记为跳过
                results[idx] = item.copy()
                results[idx]["test_program"] = "SKIPPED"
                skipped_count += 1
                progress_bar.update(1)
        
        # 处理已完成的futures
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                results[idx] = result
                
                # 更新计数器
                completed_count += 1
                if result.get("test_program") == "INCORRECT CODE":
                    incorrect_code_count += 1
                elif result.get("test_program") == "INCORRECT TEST":
                    incorrect_test_count += 1
                
                # 更新进度条
                progress_bar.update(1)
                
                # 每40个项目保存一次检查点
                if completed_count % 40 == 0:
                    # 将已处理的结果复制到数据中
                    for i, res in enumerate(results):
                        if res is not None:
                            data[i] = res
                    
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    print(f"\n已保存检查点，进度 {completed_count + skipped_count}/{total_items} 项目")
                
            except Exception as e:
                print(f"\n处理项目 {idx} 时出错: {e}")
                # 出错时保留原始项目
                results[idx] = data[idx]
                results[idx]["test_program"] = "ERROR: " + str(e)
                progress_bar.update(1)
    
    progress_bar.close()
    
    # 使用所有结果更新数据
    for i, result in enumerate(results):
        if result is not None:
            data[i] = result
    
    # 计算成功率
    processed_items = total_items - skipped_count
    valid_items = processed_items - incorrect_code_count
    success_rate = 0
    if valid_items > 0:
        success_rate = (valid_items - incorrect_test_count) / valid_items * 100
    
    print(f"\n测试结果:")
    print(f"总项目数: {total_items}")
    print(f"已有test_program的项目: {already_has_test_count}")
    print(f"跳过的项目: {skipped_count}")
    print(f"代码不正确的项目(未找到函数): {incorrect_code_count}")
    print(f"重试后测试失败的项目: {incorrect_test_count}")
    print(f"成功率: {success_rate:.2f}%")
    
    # 保存最终结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"最终结果已保存到 {output_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="使用OpenAI生成的测试用例测试Rust代码")
    parser.add_argument("--input_file", help="包含Rust代码样本的输入JSON文件")
    parser.add_argument("--output_file", help="结果输出JSON文件")
    parser.add_argument("--comparison_file", help="用于比较的JSON文件，只处理name和module不同时存在于此文件中的项目")
    parser.add_argument("--rust_files_dir", help="保存Rust测试文件的目录（可选）")
    parser.add_argument("--max_workers", type=int, default=4, help="最大并发工作线程数")
    
    args = parser.parse_args()
    
    process_dataset_concurrent(
        args.input_file, 
        args.output_file, 
        args.comparison_file,
        args.rust_files_dir, 
        args.max_workers
    )