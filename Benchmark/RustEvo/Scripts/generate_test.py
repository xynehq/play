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
api_key = ''
base_url = ''
model = 'deepseek-v3'

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
        if fixed_code.startswith("```"):
            # Extract content between code blocks
            pattern = r"```(?:rust)?\s*([\s\S]*?)```"
            matches = re.findall(pattern, fixed_code)
            if matches:
                fixed_code = "\n".join(matches)
            else:
                # If regex didn't work, try a simple trim approach
                fixed_code = fixed_code.replace("```rust", "").replace("```", "").strip()
        
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
        print(f"Item {item_index}: Test attempt {attempt+1}/{max_attempts}")
        
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
            with open(os.path.join(log_dir, f"test_{item_index:05d}_attempt_{attempt+1}.log"), 'w', encoding='utf-8') as f:
                f.write(f"ATTEMPT: {attempt+1}\nSUCCESS: {success}\nERROR: {error_message}\n\nDETAILED OUTPUT:\n{detailed_output}")
        
        if success:
            print(f"Item {item_index}: Test SUCCESSFUL on attempt {attempt+1}")
            break
        
        print(f"Item {item_index}: Test FAILED on attempt {attempt+1}: {error_message}")
        
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


def process_dataset_concurrent(input_file: str, output_file: str, rust_files_dir: str = None, max_workers: int = 4) -> None:
    """Process the dataset with concurrent workers while maintaining order."""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_items = len(data)
    results = [None] * total_items  # Pre-allocate result list to maintain order
    
    # Create progress bar
    progress_bar = tqdm(total=total_items, desc="Processing Rust code samples")
    
    # Counter for tracking progress
    completed_count = 0
    incorrect_code_count = 0
    incorrect_test_count = 0
    
    # Process items concurrently while preserving order
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and store futures with their original indices
        future_to_idx = {
            executor.submit(process_item, item, idx, rust_files_dir): idx 
            for idx, item in enumerate(data)
        }
        
        # Process completed futures as they finish
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                results[idx] = result
                
                # Update counters
                completed_count += 1
                if result.get("test_program") == "INCORRECT CODE":
                    incorrect_code_count += 1
                elif result.get("test_program") == "INCORRECT TEST":
                    incorrect_test_count += 1
                
                # Update progress bar
                progress_bar.update(1)
                
                # Save checkpoint every 40 items
                if completed_count % 40 == 0:
                    # Copy over processed results to data
                    for i, res in enumerate(results):
                        if res is not None:
                            data[i] = res
                    
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    print(f"\nCheckpoint saved at {completed_count}/{total_items} items")
                
            except Exception as e:
                print(f"\nError processing item {idx}: {e}")
                # In case of error, keep the original item
                results[idx] = data[idx]
                results[idx]["test_program"] = "ERROR: " + str(e)
                progress_bar.update(1)
    
    progress_bar.close()
    
    # Update data with all results
    for i, result in enumerate(results):
        if result is not None:
            data[i] = result
    
    # Calculate success rate
    valid_items = total_items - incorrect_code_count
    success_rate = 0
    if valid_items > 0:
        success_rate = (valid_items - incorrect_test_count) / valid_items * 100
    
    print(f"\nTest Results:")
    print(f"Total items: {total_items}")
    print(f"Incorrect code (function not found): {incorrect_code_count}")
    print(f"Tests that failed after retries: {incorrect_test_count}")
    print(f"Success rate: {success_rate:.2f}%")
    
    # Save the final results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Final results saved to {output_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Rust code using OpenAI-generated test cases")
    parser.add_argument("--input_file", help="Input JSON file with Rust code samples")
    parser.add_argument("--output_file", help="Output JSON file for results")
    parser.add_argument("--rust_files_dir", help="Directory to save Rust test files (optional)")
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum number of concurrent workers")
    
    args = parser.parse_args()
    
    process_dataset_concurrent(args.input_file, args.output_file, args.rust_files_dir, args.max_workers)