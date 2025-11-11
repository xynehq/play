from openai import OpenAI
import os
import warnings
import tempfile
import subprocess
import re
def call_LLM(prompt_text, model, API_KEY, BASE_URL) -> str:
    """
    Call ChatGPT (o1-mini) with a prompt that requests generating assert tests
    for the given function_code based on the user's query.

    The function returns the generated test code (str), or None upon failure.
    """
    print(prompt_text)
    client = OpenAI(
        api_key=API_KEY,
        # base_url="https://xiaoai.plus/v1"
        # base_url="https://api.agicto.cn/v1"
        # base_url="https://api.deepseek.com"
        # base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        base_url=BASE_URL
    )

    while True:
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt_text,
                    }
                ],
                model=model,
                temperature=0.1
            )
        except Exception as e:
            warnings.warn(f"API call failed with error: {e}")
            continue  # Retry the API call

        # Basic checks to ensure a valid response from the API
        if not chat_completion:
            warnings.warn("Received no response from the API.")
            continue  # Retry instead of returning None

        if not hasattr(chat_completion, 'choices') or chat_completion.choices is None:
            warnings.warn("The 'choices' field is missing in the API response.")
            warnings.warn(f"Full response: {chat_completion}")
            continue  # Retry
        if len(chat_completion.choices) == 0:
            warnings.warn("The 'choices' list is empty.")
            warnings.warn(f"Full response: {chat_completion}")
            continue  # Retry

        first_choice = chat_completion.choices[0]
        if not hasattr(first_choice, 'message') or first_choice.message is None:
            warnings.warn("The 'message' field is missing in the first choice.")
            warnings.warn(f"First choice: {first_choice}")
            continue  # Retry
        if not hasattr(first_choice.message, 'content') or first_choice.message.content is None:
            warnings.warn("The 'content' field is missing in the message.")
            warnings.warn(f"Message: {first_choice.message}")
            continue  # Retry

        # Remove potential Python code fences
        response = first_choice.message.content
        # print("response: ", response)
        return response


def run_rust_code(code: str) -> str:
    with tempfile.TemporaryDirectory() as temp_dir:
        rust_file_path = os.path.join(temp_dir, 'main.rs')
        with open(rust_file_path, 'w', encoding='utf-8') as f:
            f.write(code)

        # 编译Rust代码
        compile_proc = subprocess.run(
            ['rustc', 'main.rs'],
            cwd=temp_dir,
            capture_output=True,
            text=True
        )

        if compile_proc.returncode != 0:
            return f"Compile Error:\n{compile_proc.stderr.strip()}"

        # 确定可执行文件路径
        executable = 'main.exe' if os.name == 'nt' else './main'
        exec_path = os.path.join(temp_dir, executable)

        # 执行编译后的程序
        run_proc = subprocess.run(
            [exec_path],
            cwd=temp_dir,
            capture_output=True,
            text=True
        )

        output = []
        if run_proc.stdout:
            output.append(run_proc.stdout.strip())
        if run_proc.stderr:
            output.append(f"Runtime Error:\n{run_proc.stderr.strip()}")

        return '\n'.join(output) if any(output) else "No Output"

def get_code_solution_prompt(query, signature, api_info):
    prompt=f"""
    Please complete the following code generation task. The function signature is provided. 
    Follow the requirements strictly and implement the functionality as specified.
    Note: Focus on implementing the core functionality using the specified APIs and best practices.

    ### Task Description ###
    {query}

    ### Required Function Signature ###
    {signature}

    ### Available API Information ###
    {api_info}

    Please directly output the code. Do NOT add any comment.
    """
    return prompt

def get_test_program_prompt(query,code_solution,api_name):
    prompt=f"""
     ### Task Description ###
    You are a code testing expert who needs to automatically generate a testing program based on the user's natural language query (Query) and the provided code (Code). The goal is to check if the code uses specific APIs or satisfies certain constraints. The test should include two parts:

    Static Analysis: Check if the code contains/excludes required APIs (e.g., check if requests.get() is used or disallow the use of numpy).
    Dynamic Test Cases: Provide input-output examples to verify if the code's functionality meets the requirements (even if the API usage is correct, a functional error should result in a failure).

    ### Input ###
    Query: {query}
    Code to test: {code_solution}
    API required: {api_name}

 
    Please directly output the complete and executable test program. Do NOT add any comment."""

    return prompt

def extract_rust_code(text: str) -> str:

    matches = re.findall(r"```rust\s*(.*?)```", text, re.DOTALL)

    if not matches:
        return text.strip()

    return "\n\n".join(match.strip() for match in matches)