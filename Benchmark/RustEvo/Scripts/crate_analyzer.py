"""
Rust Third-Party Library API Evolution Tracker

This script:
1. Downloads specific versions of Rust crates
2. Extracts API information from different versions
3. Detects changes between versions
4. Outputs the results to JSON files

Change types:
- Stabilized APIs: New APIs or APIs that became stable
- Signature Changes: APIs with changed signatures
- Implicit Functional Changes: Implementation changes with same signature
- Deprecated/Replaced APIs: APIs marked as deprecated or removed

how to run:
python scripts/crate_analyzer.py --crates_num 50 --start_date 2024-11-01 --end_date 2025-02-21 --version_num 4
"""

import os
import re
import json
import requests
import tarfile
import tempfile
import shutil
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime
import subprocess
from tqdm import tqdm
import time


os.makedirs("crates", exist_ok=True)
os.makedirs("reports", exist_ok=True)

def download_crate(crate: str, version: str) -> str:
    """Download a specific version of a crate directly from crates.io"""
    output_dir = f"crates/{crate}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/{version}.tar.gz"
    
    # Check if already downloaded
    if os.path.exists(output_file):
        # print(f"Already downloaded: {crate}-{version}")
        return output_file
    
    # print(f"Downloading {crate} version {version}...")
    url = f"https://static.crates.io/crates/{crate}/{crate}-{version}.crate"
    print(f"URL: {url}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(output_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Successfully downloaded to {output_file}")
        return output_file
    except Exception as e:
        print(f"Error: {e}")
        return None

def extract_crate(tar_path: str) -> str:
    """Extract the crate and return the path to the src directory"""
    if not tar_path or not os.path.exists(tar_path):
        return None
    
    temp_dir = tempfile.mkdtemp()
    # print(f"Extracting to {temp_dir}...")
    
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=temp_dir)
        
        # Find the src directory
        for root, dirs, _ in os.walk(temp_dir):
            if "src" in dirs:
                src_dir = os.path.join(root, "src")
                # print(f"Found src directory: {src_dir}")
                return src_dir
    except Exception as e:
        print(f"Error extracting {tar_path}: {e}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None
    
    return temp_dir

def collect_rust_files(dir_path: str) -> List[str]:
    """Collect all .rs files in a directory"""
    if not dir_path:
        return []
    
    rust_files = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".rs"):
                rust_files.append(os.path.join(root, file))
    return rust_files

def extract_module_path(file_path: str, src_dir: str) -> str:
    """Extract module path from file path"""
    rel_path = os.path.relpath(file_path, src_dir)
    if rel_path == "lib.rs":
        return ""
    elif rel_path == "main.rs":
        return "main"
    else:
        # Convert path to module notation
        module_path = os.path.splitext(rel_path)[0]
        if module_path.endswith("/mod") or module_path.endswith("\\mod"):
            module_path = module_path[:-4]
        return re.sub(r'[/\\]', '::', module_path)

# def parse_rust_file(file_path: str, module_path: str) -> List[Dict[str, Any]]:
#     """Parse a Rust file and extract API information using regex"""
#     with open(file_path, "r", encoding="utf-8", errors="replace") as f:
#         content = f.read()
    
#     apis = []

#     def clean_doc_comment(doc_str: str) -> str:
#         """
#         Clean documentation comments by removing /// prefixes while preserving line structure.
#         Also handles and cleans up incomplete/empty code blocks.
#         """
#         if not doc_str:
#             return ""
        
#         # Split by lines, strip each line's '///' prefix, and rejoin
#         lines = doc_str.split('\n')
#         cleaned_lines = []
        
#         for line in lines:
#             # Remove '///' prefix and one space after it (if present)
#             cleaned = re.sub(r'^///\s?', '', line)
#             cleaned_lines.append(cleaned)
        
#         cleaned_text = '\n'.join(cleaned_lines).strip()
        
#         # Handle incomplete or problematic code blocks
#         # Count the number of code block markers
#         code_block_markers = cleaned_text.count('```')
        
#         # If there's an odd number of markers, add a closing one
#         if code_block_markers % 2 != 0:
#             cleaned_text += '\n```'
        
#         # Remove empty code blocks (```)
#         cleaned_text = re.sub(r'```\s*```', '', cleaned_text)
        
#         # If all that's left is just code block markers with no content between them, return empty string
#         if re.match(r'^\s*```.*```\s*$', cleaned_text, re.DOTALL) and not re.search(r'```\s*[^\s```]+\s*```', cleaned_text, re.DOTALL):
#             return ""
        
#         # If the entire documentation is just code markers, return empty string
#         if cleaned_text.strip() in ['```', '``` ```']:
#             return ""
            
#         return cleaned_text
    

    
#     # Extract all items with public visibility
#     # Functions - captures complete function with body
#     func_pattern = r'(?:(///[^\n]*\n)*)(?:(#\[[^\]]+\]\n)*)\s*pub\s+fn\s+([a-zA-Z0-9_]+)(?:<[^>]*>)?\s*\(([^)]*)\)(?:\s*->\s*([^{]*))?(?:\s*where\s+[^{]*)?(?:\s*\{([^}]*(?:\{[^}]*\}[^}]*)*)\})'
#     for match in re.finditer(func_pattern, content, re.DOTALL):
#         docs = match.group(1) or ""
#         attrs = match.group(2) or ""
#         name = match.group(3)
#         params = match.group(4) or ""
#         return_type = match.group(5) or ""
#         function_body = match.group(6) or ""
        
#         # Clean up docs - use the new helper function
#         docs = clean_doc_comment(docs)
        
#         # Extract example code
#         examples = []
#         example_matches = re.finditer(r'# Examples(.*?)(?:# |$)', docs, re.DOTALL)
#         for ex_match in example_matches:
#             example_text = ex_match.group(1)
#             code_blocks = re.finditer(r'```(?:rust)?\n(.*?)```', example_text, re.DOTALL)
#             for code_match in code_blocks:
#                 examples.append(code_match.group(1).strip())
        
#         # Determine stability and deprecation
#         stable_match = re.search(r'#\[stable\(.*?since\s*=\s*"([^"]+)"', attrs)
#         deprecated_match = re.search(r'#\[deprecated', attrs)
        
#         # Create API entity
#         api = {
#             "name": name,
#             "module": module_path,
#             "type": "function",
#             "signature": f"fn {name}({params}){' -> ' + return_type.strip() if return_type else ''}",
#             "documentation": docs,
#             # "examples": "\n".join(examples),
#             "source_code": match.group(0),
#             "function_body": function_body.strip(),
#             "attributes": {}
#         }
        
#         if stable_match:
#             api["attributes"]["stable"] = stable_match.group(1)
#         if deprecated_match:
#             api["attributes"]["deprecated"] = True
            
#             # Check for replacement suggestion
#             replace_match = re.search(r'replace_with\s*=\s*"([^"]+)"', attrs)
#             if replace_match:
#                 api["attributes"]["replacement"] = replace_match.group(1)
        
#         apis.append(api)
    
#     # Structs
#     struct_pattern = r'(?:(///[^\n]*\n)*)(?:(#\[[^\]]+\]\n)*)\s*pub\s+struct\s+([a-zA-Z0-9_]+)(?:<[^>]*>)?(?:\s*\{([^}]*)\}|\s*\([^)]*\)|\s*;)'
#     for match in re.finditer(struct_pattern, content, re.DOTALL):
#         docs = match.group(1) or ""
#         attrs = match.group(2) or ""
#         name = match.group(3)
#         struct_body = match.group(4) or ""
        
#         # Clean up docs - use the new helper function
#         docs = clean_doc_comment(docs)
        
#         # Determine stability and deprecation
#         stable_match = re.search(r'#\[stable\(.*?since\s*=\s*"([^"]+)"', attrs)
#         deprecated_match = re.search(r'#\[deprecated', attrs)
        
#         api = {
#             "name": name,
#             "module": module_path,
#             "type": "struct",
#             "signature": f"struct {name}",
#             "documentation": docs,
#             "source_code": match.group(0),
#             "struct_body": struct_body.strip(),
#             "attributes": {}
#         }
        
#         if stable_match:
#             api["attributes"]["stable"] = stable_match.group(1)
#         if deprecated_match:
#             api["attributes"]["deprecated"] = True
        
#         apis.append(api)
    
#     # Enums
#     enum_pattern = r'(?:(///[^\n]*\n)*)(?:(#\[[^\]]+\]\n)*)\s*pub\s+enum\s+([a-zA-Z0-9_]+)(?:<[^>]*>)?\s*\{([^}]*)\}'
#     for match in re.finditer(enum_pattern, content, re.DOTALL):
#         docs = match.group(1) or ""
#         attrs = match.group(2) or ""
#         name = match.group(3)
#         enum_body = match.group(4) or ""
        
#         # Clean up docs - use the new helper function
#         docs = clean_doc_comment(docs)
        
#         # Determine stability and deprecation
#         stable_match = re.search(r'#\[stable\(.*?since\s*=\s*"([^"]+)"', attrs)
#         deprecated_match = re.search(r'#\[deprecated', attrs)
        
#         # Extract enum variants
#         variants = []
#         variant_pattern = r'(?:(///[^\n]*\n)*)(?:(#\[[^\]]+\]\n)*)\s*([a-zA-Z0-9_]+)(?:\s*\{[^}]*\}|\s*\([^)]*\))?(?:\s*=\s*[^,]*)?'
#         for variant_match in re.finditer(variant_pattern, enum_body, re.DOTALL):
#             variant_docs = variant_match.group(1) or ""
#             variant_attrs = variant_match.group(2) or ""
#             variant_name = variant_match.group(3)
            
#             # Clean up variant docs - use the new helper function
#             variant_docs = clean_doc_comment(variant_docs)
            
#             variants.append({
#                 "name": variant_name,
#                 "documentation": variant_docs,
#                 "source_code": variant_match.group(0)
#             })
        
#         api = {
#             "name": name,
#             "module": module_path,
#             "type": "enum",
#             "signature": f"enum {name}",
#             "documentation": docs,
#             "source_code": match.group(0),
#             "enum_body": enum_body.strip(),
#             "variants": variants,
#             "attributes": {}
#         }
        
#         if stable_match:
#             api["attributes"]["stable"] = stable_match.group(1)
#         if deprecated_match:
#             api["attributes"]["deprecated"] = True
        
#         apis.append(api)
    
#     # Traits
#     trait_pattern = r'(?:(///[^\n]*\n)*)(?:(#\[[^\]]+\]\n)*)\s*pub\s+trait\s+([a-zA-Z0-9_]+)(?:<[^>]*>)?(?:\s*:\s*[^{]*)?(?:\s*where\s+[^{]*)?\s*\{([^}]*(?:\{[^}]*\}[^}]*)*)\}'
#     for match in re.finditer(trait_pattern, content, re.DOTALL):
#         docs = match.group(1) or ""
#         attrs = match.group(2) or ""
#         name = match.group(3)
#         trait_body = match.group(4) or ""
        
#         # Clean up docs - use the new helper function
#         docs = clean_doc_comment(docs)
        
#         # Determine stability and deprecation
#         stable_match = re.search(r'#\[stable\(.*?since\s*=\s*"([^"]+)"', attrs)
#         deprecated_match = re.search(r'#\[deprecated', attrs)
        
#         # Extract trait methods (both required and provided)
#         methods = []
        
#         # Required methods (function signatures without bodies)
#         req_method_pattern = r'(?:(///[^\n]*\n)*)(?:(#\[[^\]]+\]\n)*)\s*fn\s+([a-zA-Z0-9_]+)(?:<[^>]*>)?\s*\(([^)]*)\)(?:\s*->\s*([^;]*))?(?:\s*where\s+[^;]*)?;'
#         for method_match in re.finditer(req_method_pattern, trait_body, re.DOTALL):
#             method_docs = method_match.group(1) or ""
#             method_attrs = method_match.group(2) or ""
#             method_name = method_match.group(3)
#             method_params = method_match.group(4) or ""
#             method_return = method_match.group(5) or ""
            
#             # Clean up method docs - use the new helper function
#             method_docs = clean_doc_comment(method_docs)
            
#             methods.append({
#                 "name": method_name,
#                 "type": "required",
#                 "signature": f"fn {method_name}({method_params}){' -> ' + method_return.strip() if method_return else ''}",
#                 "documentation": method_docs,
#                 "source_code": method_match.group(0)
#             })
        
#         # Provided methods (with default implementations)
#         provided_method_pattern = r'(?:(///[^\n]*\n)*)(?:(#\[[^\]]+\]\n)*)\s*fn\s+([a-zA-Z0-9_]+)(?:<[^>]*>)?\s*\(([^)]*)\)(?:\s*->\s*([^{]*))?(?:\s*where\s+[^{]*)?\s*\{([^}]*(?:\{[^}]*\}[^}]*)*)\}'
#         for method_match in re.finditer(provided_method_pattern, trait_body, re.DOTALL):
#             method_docs = method_match.group(1) or ""
#             method_attrs = method_match.group(2) or ""
#             method_name = method_match.group(3)
#             method_params = method_match.group(4) or ""
#             method_return = method_match.group(5) or ""
#             method_body = method_match.group(6) or ""
            
#             # Clean up method docs - use the new helper function
#             method_docs = clean_doc_comment(method_docs)
            
#             methods.append({
#                 "name": method_name,
#                 "type": "provided",
#                 "signature": f"fn {method_name}({method_params}){' -> ' + method_return.strip() if method_return else ''}",
#                 "documentation": method_docs,
#                 "source_code": method_match.group(0),
#                 "body": method_body.strip()
#             })
        
#         api = {
#             "name": name,
#             "module": module_path,
#             "type": "trait",
#             "signature": f"trait {name}",
#             "documentation": docs,
#             "source_code": match.group(0),
#             "trait_body": trait_body.strip(),
#             "methods": methods,
#             "attributes": {}
#         }
        
#         if stable_match:
#             api["attributes"]["stable"] = stable_match.group(1)
#         if deprecated_match:
#             api["attributes"]["deprecated"] = True
        
#         apis.append(api)
    
#     # Constants
#     const_pattern = r'(?:(///[^\n]*\n)*)(?:(#\[[^\]]+\]\n)*)\s*pub\s+const\s+([a-zA-Z0-9_]+)\s*:\s*([^=]*)=\s*([^;]*);'
#     for match in re.finditer(const_pattern, content, re.DOTALL):
#         docs = match.group(1) or ""
#         attrs = match.group(2) or ""
#         name = match.group(3)
#         const_type = match.group(4) or ""
#         const_value = match.group(5) or ""
        
#         # Clean up docs - use the new helper function
#         docs = clean_doc_comment(docs)
        
#         # Determine stability and deprecation
#         stable_match = re.search(r'#\[stable\(.*?since\s*=\s*"([^"]+)"', attrs)
#         deprecated_match = re.search(r'#\[deprecated', attrs)
        
#         api = {
#             "name": name,
#             "module": module_path,
#             "type": "constant",
#             "signature": f"const {name}: {const_type} = {const_value}",
#             "documentation": docs,
#             # "examples": "",
#             "source_code": match.group(0),
#             "const_type": const_type.strip(),
#             "const_value": const_value.strip(),
#             "attributes": {}
#         }
        
#         if stable_match:
#             api["attributes"]["stable"] = stable_match.group(1)
#         if deprecated_match:
#             api["attributes"]["deprecated"] = True
        
#         apis.append(api)
    
#     return apis




def parse_rust_file(file_path: str, module_path: str) -> List[Dict[str, Any]]:
    """Parse a Rust file and extract API information using regex"""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    
    apis = []
    
    def clean_doc_comment(doc_str: str) -> str:
        """
        Clean documentation comments by removing /// prefixes while preserving line structure.
        """
        if not doc_str:
            return ""
        
        # Split by lines, strip each line's '///' prefix, and rejoin
        lines = doc_str.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Remove '///' prefix and one space after it (if present)
            cleaned = re.sub(r'^///\s?', '', line)
            cleaned_lines.append(cleaned)
        
        return '\n'.join(cleaned_lines).strip()

    # Functions - captures complete function with body
    func_pattern = r'((?:///[^\n]*\n)*)(?:(#\[[^\]]+\]\n)*)(pub\s+fn\s+[a-zA-Z0-9_]+(?:<[^>]*>)?\s*\([^)]*\)(?:\s*->\s*[^{]*)?(?:\s*where\s+[^{]*)?\s*\{[^}]*(?:\{[^}]*\}[^}]*)*\})'
    for match in re.finditer(func_pattern, content, re.DOTALL):
        doc_comments = match.group(1) or ""
        attributes = match.group(2) or ""
        actual_code = match.group(3) or ""
        
        # Extract function name, params, return type from the code
        func_details = re.match(r'pub\s+fn\s+([a-zA-Z0-9_]+)(?:<[^>]*>)?\s*\(([^)]*)\)(?:\s*->\s*([^{]*))?', actual_code)
        if not func_details:
            continue
            
        name = func_details.group(1)
        params = func_details.group(2) or ""
        return_type = func_details.group(3) or ""
        
        # Extract function body
        body_match = re.search(r'\{([^}]*(?:\{[^}]*\}[^}]*)*)\}', actual_code)
        function_body = body_match.group(1) if body_match else ""
        
        # Clean up docs
        cleaned_docs = clean_doc_comment(doc_comments)
        
        # Determine stability and deprecation
        stable_match = re.search(r'#\[stable\(.*?since\s*=\s*"([^"]+)"', attributes)
        deprecated_match = re.search(r'#\[deprecated', attributes)
        
        # Create API entity
        api = {
            "name": name,
            "module": module_path,
            "type": "function",
            "signature": f"fn {name}({params}){' -> ' + return_type.strip() if return_type else ''}",
            "documentation": cleaned_docs,
            "source_code": actual_code,
            "function_body": function_body.strip(),
            "attributes": {}
        }
        
        if stable_match:
            api["attributes"]["stable"] = stable_match.group(1)
        if deprecated_match:
            api["attributes"]["deprecated"] = True
            
            # Check for replacement suggestion
            replace_match = re.search(r'replace_with\s*=\s*"([^"]+)"', attributes)
            if replace_match:
                api["attributes"]["replacement"] = replace_match.group(1)
        
        apis.append(api)
    
    # Structs
    struct_pattern = r'((?:///[^\n]*\n)*)(?:(#\[[^\]]+\]\n)*)(pub\s+struct\s+[a-zA-Z0-9_]+(?:<[^>]*>)?(?:\s*\{[^}]*\}|\s*\([^)]*\)|\s*;))'
    for match in re.finditer(struct_pattern, content, re.DOTALL):
        doc_comments = match.group(1) or ""
        attributes = match.group(2) or ""
        actual_code = match.group(3) or ""
        
        # Extract struct name and body
        struct_details = re.match(r'pub\s+struct\s+([a-zA-Z0-9_]+)(?:<[^>]*>)?', actual_code)
        if not struct_details:
            continue
            
        name = struct_details.group(1)
        
        # Extract struct body if present
        body_match = re.search(r'\{([^}]*)\}', actual_code)
        struct_body = body_match.group(1) if body_match else ""
        
        # Clean up docs
        cleaned_docs = clean_doc_comment(doc_comments)
        
        # Determine stability and deprecation
        stable_match = re.search(r'#\[stable\(.*?since\s*=\s*"([^"]+)"', attributes)
        deprecated_match = re.search(r'#\[deprecated', attributes)
        
        api = {
            "name": name,
            "module": module_path,
            "type": "struct",
            "signature": f"struct {name}",
            "documentation": cleaned_docs,
            "source_code": actual_code,
            "struct_body": struct_body.strip(),
            "attributes": {}
        }
        
        if stable_match:
            api["attributes"]["stable"] = stable_match.group(1)
        if deprecated_match:
            api["attributes"]["deprecated"] = True
        
        apis.append(api)
    
    # Enums
    enum_pattern = r'((?:///[^\n]*\n)*)(?:(#\[[^\]]+\]\n)*)(pub\s+enum\s+[a-zA-Z0-9_]+(?:<[^>]*>)?\s*\{[^}]*\})'
    for match in re.finditer(enum_pattern, content, re.DOTALL):
        doc_comments = match.group(1) or ""
        attributes = match.group(2) or ""
        actual_code = match.group(3) or ""
        
        # Extract enum name and body
        enum_details = re.match(r'pub\s+enum\s+([a-zA-Z0-9_]+)(?:<[^>]*>)?', actual_code)
        if not enum_details:
            continue
            
        name = enum_details.group(1)
        
        # Extract enum body
        body_match = re.search(r'\{([^}]*)\}', actual_code)
        enum_body = body_match.group(1) if body_match else ""
        
        # Clean up docs
        cleaned_docs = clean_doc_comment(doc_comments)
        
        # Determine stability and deprecation
        stable_match = re.search(r'#\[stable\(.*?since\s*=\s*"([^"]+)"', attributes)
        deprecated_match = re.search(r'#\[deprecated', attributes)
        
        # Extract enum variants
        variants = []
        variant_pattern = r'((?:///[^\n]*\n)*)(?:(#\[[^\]]+\]\n)*)((?:[a-zA-Z0-9_]+)(?:\s*\{[^}]*\}|\s*\([^)]*\))?(?:\s*=\s*[^,]*)?)'
        for variant_match in re.finditer(variant_pattern, enum_body, re.DOTALL):
            variant_doc_comments = variant_match.group(1) or ""
            variant_attributes = variant_match.group(2) or ""
            variant_actual_code = variant_match.group(3) or ""
            
            # Extract variant name
            variant_name_match = re.match(r'([a-zA-Z0-9_]+)', variant_actual_code)
            if not variant_name_match:
                continue
                
            variant_name = variant_name_match.group(1)
            
            # Clean up variant docs
            variant_cleaned_docs = clean_doc_comment(variant_doc_comments)
            
            variants.append({
                "name": variant_name,
                "documentation": variant_cleaned_docs,
                "source_code": variant_actual_code
            })
        
        api = {
            "name": name,
            "module": module_path,
            "type": "enum",
            "signature": f"enum {name}",
            "documentation": cleaned_docs,
            "source_code": actual_code,
            "enum_body": enum_body.strip(),
            "variants": variants,
            "attributes": {}
        }
        
        if stable_match:
            api["attributes"]["stable"] = stable_match.group(1)
        if deprecated_match:
            api["attributes"]["deprecated"] = True
        
        apis.append(api)
    
    # Traits
    trait_pattern = r'((?:///[^\n]*\n)*)(?:(#\[[^\]]+\]\n)*)(pub\s+trait\s+[a-zA-Z0-9_]+(?:<[^>]*>)?(?:\s*:\s*[^{]*)?(?:\s*where\s+[^{]*)?\s*\{[^}]*(?:\{[^}]*\}[^}]*)*\})'
    for match in re.finditer(trait_pattern, content, re.DOTALL):
        doc_comments = match.group(1) or ""
        attributes = match.group(2) or ""
        actual_code = match.group(3) or ""
        
        # Extract trait name and body
        trait_details = re.match(r'pub\s+trait\s+([a-zA-Z0-9_]+)(?:<[^>]*>)?', actual_code)
        if not trait_details:
            continue
            
        name = trait_details.group(1)
        
        # Extract trait body
        body_match = re.search(r'\{([^}]*(?:\{[^}]*\}[^}]*)*)\}', actual_code)
        trait_body = body_match.group(1) if body_match else ""
        
        # Clean up docs
        cleaned_docs = clean_doc_comment(doc_comments)
        
        # Determine stability and deprecation
        stable_match = re.search(r'#\[stable\(.*?since\s*=\s*"([^"]+)"', attributes)
        deprecated_match = re.search(r'#\[deprecated', attributes)
        
        # Extract trait methods (both required and provided)
        methods = []
        
        # Required methods (function signatures without bodies)
        req_method_pattern = r'((?:///[^\n]*\n)*)(?:(#\[[^\]]+\]\n)*)(fn\s+[a-zA-Z0-9_]+(?:<[^>]*>)?\s*\([^)]*\)(?:\s*->\s*[^;]*)?(?:\s*where\s+[^;]*)?;)'
        for method_match in re.finditer(req_method_pattern, trait_body, re.DOTALL):
            method_doc_comments = method_match.group(1) or ""
            method_attributes = method_match.group(2) or ""
            method_actual_code = method_match.group(3) or ""
            
            # Extract method name, params, return type
            method_details = re.match(r'fn\s+([a-zA-Z0-9_]+)(?:<[^>]*>)?\s*\(([^)]*)\)(?:\s*->\s*([^;]*))?', method_actual_code)
            if not method_details:
                continue
                
            method_name = method_details.group(1)
            method_params = method_details.group(2) or ""
            method_return = method_details.group(3) or ""
            
            # Clean up method docs
            method_cleaned_docs = clean_doc_comment(method_doc_comments)
            
            methods.append({
                "name": method_name,
                "type": "required",
                "signature": f"fn {method_name}({method_params}){' -> ' + method_return.strip() if method_return else ''}",
                "documentation": method_cleaned_docs,
                "source_code": method_actual_code
            })
        
        # Provided methods (with default implementations)
        provided_method_pattern = r'((?:///[^\n]*\n)*)(?:(#\[[^\]]+\]\n)*)(fn\s+[a-zA-Z0-9_]+(?:<[^>]*>)?\s*\([^)]*\)(?:\s*->\s*[^{]*)?(?:\s*where\s+[^{]*)?\s*\{[^}]*(?:\{[^}]*\}[^}]*)*\})'
        for method_match in re.finditer(provided_method_pattern, trait_body, re.DOTALL):
            method_doc_comments = method_match.group(1) or ""
            method_attributes = method_match.group(2) or ""
            method_actual_code = method_match.group(3) or ""
            
            # Extract method name, params, return type
            method_details = re.match(r'fn\s+([a-zA-Z0-9_]+)(?:<[^>]*>)?\s*\(([^)]*)\)(?:\s*->\s*([^{]*))?', method_actual_code)
            if not method_details:
                continue
                
            method_name = method_details.group(1)
            method_params = method_details.group(2) or ""
            method_return = method_details.group(3) or ""
            
            # Extract method body
            method_body_match = re.search(r'\{([^}]*(?:\{[^}]*\}[^}]*)*)\}', method_actual_code)
            method_body = method_body_match.group(1) if method_body_match else ""
            
            # Clean up method docs
            method_cleaned_docs = clean_doc_comment(method_doc_comments)
            
            methods.append({
                "name": method_name,
                "type": "provided",
                "signature": f"fn {method_name}({method_params}){' -> ' + method_return.strip() if method_return else ''}",
                "documentation": method_cleaned_docs,
                "source_code": method_actual_code,
                "body": method_body.strip()
            })
        
        api = {
            "name": name,
            "module": module_path,
            "type": "trait",
            "signature": f"trait {name}",
            "documentation": cleaned_docs,
            "source_code": actual_code,
            "trait_body": trait_body.strip(),
            "methods": methods,
            "attributes": {}
        }
        
        if stable_match:
            api["attributes"]["stable"] = stable_match.group(1)
        if deprecated_match:
            api["attributes"]["deprecated"] = True
        
        apis.append(api)
    
    # Constants
    const_pattern = r'((?:///[^\n]*\n)*)(?:(#\[[^\]]+\]\n)*)(pub\s+const\s+[a-zA-Z0-9_]+\s*:\s*[^=]*=\s*[^;]*;)'
    for match in re.finditer(const_pattern, content, re.DOTALL):
        doc_comments = match.group(1) or ""
        attributes = match.group(2) or ""
        actual_code = match.group(3) or ""
        
        # Extract constant name, type, and value
        const_details = re.match(r'pub\s+const\s+([a-zA-Z0-9_]+)\s*:\s*([^=]*)=\s*([^;]*);', actual_code)
        if not const_details:
            continue
            
        name = const_details.group(1)
        const_type = const_details.group(2) or ""
        const_value = const_details.group(3) or ""
        
        # Clean up docs
        cleaned_docs = clean_doc_comment(doc_comments)
        
        # Determine stability and deprecation
        stable_match = re.search(r'#\[stable\(.*?since\s*=\s*"([^"]+)"', attributes)
        deprecated_match = re.search(r'#\[deprecated', attributes)
        
        api = {
            "name": name,
            "module": module_path,
            "type": "constant",
            "signature": f"const {name}: {const_type} = {const_value}",
            "documentation": cleaned_docs,
            "source_code": actual_code,
            "const_type": const_type.strip(),
            "const_value": const_value.strip(),
            "attributes": {}
        }
        
        if stable_match:
            api["attributes"]["stable"] = stable_match.group(1)
        if deprecated_match:
            api["attributes"]["deprecated"] = True
        
        apis.append(api)
    
    return apis




def filter_apis(apis: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    filtered_apis = {}
    
    for key, api in apis.items():
        has_empty_field = False
        

        required_fields = ["name", "module", "type", "signature", "documentation"]
        for field in required_fields:
            if field not in api or api[field] is None or api[field] == "":
                has_empty_field = True
                break
        
        if not has_empty_field:
            if api["type"] == "function" and ("function_body" not in api or not api["function_body"]):
                has_empty_field = True
            elif api["type"] == "struct" and ("struct_body" not in api or not api["struct_body"]):
                has_empty_field = True
            elif api["type"] == "enum" and ("enum_body" not in api or not api["enum_body"]):
                has_empty_field = True
            elif api["type"] == "trait" and ("trait_body" not in api or not api["trait_body"]):
                has_empty_field = True
            elif api["type"] == "constant" and ("const_value" not in api or not api["const_value"]):
                has_empty_field = True
        
        if not has_empty_field:
            filtered_apis[key] = api
    
    # print(f"过滤前API数量: {len(apis)}, 过滤后API数量: {len(filtered_apis)}")
    return filtered_apis



def extract_apis(src_dir: str) -> Dict[str, Dict[str, Any]]:
    """Extract all APIs from a crate"""
    if not src_dir or not os.path.exists(src_dir):
        return {}
    
    apis = {}
    rust_files = collect_rust_files(src_dir)
    
    for file_path in rust_files:
        module_path = extract_module_path(file_path, src_dir)
        file_apis = parse_rust_file(file_path, module_path)
        
        for api in file_apis:
            key = f"{api['module']}::{api['name']}" if api['module'] else api['name']
            apis[key] = api

    apis = filter_apis(apis)
    return apis

def detect_changes(old_apis: Dict, new_apis: Dict, crate: str, old_ver: str, new_ver: str) -> List[Dict]:
    """Detect API changes between two versions"""
    changes = []
    
    # Find added APIs
    for key in set(new_apis.keys()) - set(old_apis.keys()):
        api = new_apis[key]
        changes.append({
            "crate": crate,
            "name": api["name"],
            "from_version": old_ver,
            "to_version": new_ver,
            "module": api["module"],
            "type": api["type"],
            "signature": api["signature"],
            "documentation": api["documentation"],
            # "examples": api["examples"],
            "changetype": "stabilized",
            "changenote": f"A new stable API has been introduced since version {new_ver}.",
            "source_code": api["source_code"]
        })
    
    # Find removed APIs
    for key in set(old_apis.keys()) - set(new_apis.keys()):
        api = old_apis[key]
        changes.append({
            "crate": crate,
            "name": api["name"],
            "from_version": old_ver,
            "to_version": new_ver,
            "module": api["module"],
            "type": api["type"],
            "signature": api["signature"],
            "documentation": api["documentation"],
            # "examples": api["examples"],
            "changetype": "deprecated",
            "changenote": f"API removed in version {new_ver}.",
            "old_source_code": api["source_code"]
        })
    
    # Find modified APIs
    for key in set(old_apis.keys()) & set(new_apis.keys()):
        old_api = old_apis[key]
        new_api = new_apis[key]
        
        # Signature changes
        if old_api["signature"] != new_api["signature"]:
            changes.append({
                "crate": crate,
                "name": new_api["name"],
                "from_version": old_ver,
                "to_version": new_ver,
                "module": new_api["module"],
                "type": new_api["type"],
                "old_signature": old_api["signature"],
                "signature": new_api["signature"],
                "documentation": new_api["documentation"],
                # "examples": new_api["examples"],
                "changetype": "signature",
                "changenote": f"Signature changed from '{old_api['signature']}' to '{new_api['signature']}'.",
                "source_code": new_api["source_code"]
            })

        # Newly deprecated（思路不对）
        # elif ("deprecated" in new_api["attributes"] and 
        #       "deprecated" not in old_api["attributes"]):
        #     changenote = f"API marked as deprecated in version {new_ver}."
        #     if "replacement" in new_api["attributes"]:
        #         changenote += f" Use {new_api['attributes']['replacement']} instead."
            
        #     changes.append({
        #         "crate": crate,
        #         "name": new_api["name"],
        #         "from_version": old_ver,
        #         "to_version": new_ver,
        #         "module": new_api["module"],
        #         "type": new_api["type"],
        #         "signature": new_api["signature"],
        #         "documentation": new_api["documentation"],
        #         # "examples": new_api["examples"],
        #         "changetype": "deprecated",
        #         "changenote": changenote,
        #         "replace_source_code": new_api["source_code"],
        #         "old_source_code": old_api["source_code"]
        #     })


        # Implicit changes (documentation changes as indicator)
        elif old_api["source_code"] != new_api["source_code"] and old_api["signature"] == new_api["signature"]:
            # Look for keywords indicating behavior changes
            # change_keywords = ["now", "change", "behavior", "different", "warning"]
            # doc_diff = new_api["documentation"].lower()
            
            # if any(keyword in doc_diff for keyword in change_keywords):
            changes.append({
                "crate": crate,
                "name": new_api["name"],
                "from_version": old_ver,
                "to_version": new_ver,
                "module": new_api["module"],
                "type": new_api["type"],
                "signature": new_api["signature"],
                "documentation": new_api["documentation"],
                # "examples": new_api["examples"],
                "changetype": "implicit",
                "changenote": "Implementation has changed while maintaining the same signature.",
                "old_source_code": old_api["source_code"],
                "source_code": new_api["source_code"]
            })
    
    return changes


def get_crate_versions(crate: str, start_date=None, end_date=None, version_num: int = 4) -> List[str]:
    """
    获取指定时间范围内发布的crate版本
    
    Args:
        crate: crate名称
        start_date: 开始日期 (格式: 'YYYY-MM-DD')，只返回此日期之后的版本
        end_date: 结束日期 (格式: 'YYYY-MM-DD')，只返回此日期之前的版本
        version_num: 要返回的版本数量，从最新版本开始计算
        
    Returns:
        按版本号升序排列的版本列表，最多包含version_num个版本，全部在指定的日期范围内
    """
    try:
        response = requests.get(f"https://crates.io/api/v1/crates/{crate}/versions")
        response.raise_for_status()
        versions_data = response.json()['versions']
        
        # 将日期字符串转换为datetime对象
        start_datetime = None
        if start_date:
            start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
        
        end_datetime = None
        if end_date:
            end_datetime = datetime.strptime(end_date, '%Y-%m-%d')
            # 设置为当天结束时间
            end_datetime = end_datetime.replace(hour=23, minute=59, second=59)
        
        # 按日期过滤版本并移除预发布版本
        filtered_versions = []
        for version in versions_data:
            # 跳过预发布版本
            version_num_str = version['num']
            if ('alpha' in version_num_str or 'beta' in version_num_str or 
                'rc' in version_num_str or '-' in version_num_str):
                continue
            
            # 解析创建时间
            created_at = datetime.strptime(version['created_at'], "%Y-%m-%dT%H:%M:%S.%f%z")
            created_at = created_at.replace(tzinfo=None)  # 移除时区信息以便比较
            
            # 检查是否在指定日期范围内
            if start_datetime and created_at < start_datetime:
                continue
            if end_datetime and created_at > end_datetime:
                continue
            
            filtered_versions.append((version_num_str, created_at))
        
        # 按版本号对版本进行语义化排序
        def version_key(v):
            parts = v[0].split('.')
            return [int(p) if p.isdigit() else 0 for p in parts]
        
        filtered_versions.sort(key=lambda x: version_key(x))
        
        # 返回最新的version_num个版本，按版本号升序排列
        # 如果可用版本少于version_num，返回所有可用版本
        if len(filtered_versions) <= version_num:
            return [v[0] for v in filtered_versions]
        else:
            # 返回最新的version_num个版本（列表中的最后version_num个元素）
            return [v[0] for v in filtered_versions[-version_num:]]
        
    except Exception as e:
        print(f"Error fetching versions for {crate}: {e}")
        return []




def get_top_crates(limit: int = 20) -> List[str]:
    """
    Get top crates from crates.io based on downloads and recent updates
    Returns crates with high download counts that have been updated in the last two months
    """
    try:
        # Use a smaller per_page value
        per_page = 50  # Reduced from 250 to a more reasonable value
        larger_pool = min(200, limit * 5)  # Cap at 200 to avoid excessive requests
        
        # Add proper headers
        headers = {
            'User-Agent': 'Rust-API-Evolution-Analyzer/1.0',
            'Accept': 'application/json'
        }
        
        # Make multiple paginated requests if needed
        all_crates = []
        page = 1
        
        while len(all_crates) < larger_pool:
            url = f"https://crates.io/api/v1/crates?sort=downloads&per_page={per_page}&page={page}"
            print(f"Requesting: {url}")
            
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            crates_batch = response.json().get('crates', [])
            if not crates_batch:
                break  # No more crates
                
            all_crates.extend(crates_batch)
            page += 1
            
            # Add a small delay to avoid rate limiting
            time.sleep(0.5)
        
        # Get current date for comparison
        from datetime import datetime, timedelta
        current_date = datetime.now()
        two_months_ago = current_date - timedelta(days=60)
        
        # Filter crates by checking their last update time
        recent_top_crates = []
        
        for crate in all_crates:
            crate_id = crate['id']
            
            # Add a retry mechanism
            max_retries = 3
            retry_count = 0
            success = False
            
            while retry_count < max_retries and not success:
                try:
                    versions_url = f"https://crates.io/api/v1/crates/{crate_id}/versions"
                    print(f"Checking versions for: {crate_id}")
                    
                    versions_response = requests.get(versions_url, headers=headers)
                    versions_response.raise_for_status()
                    
                    # Check if any version was released in the last two months
                    recent_version = False
                    
                    for version in versions_response.json()['versions'][:5]:  # Check only the 5 most recent versions
                        # Skip pre-releases
                        if 'alpha' in version['num'] or 'beta' in version['num'] or 'rc' in version['num'] or '-' in version['num']:
                            continue
                            
                        updated_at = datetime.strptime(version['updated_at'], "%Y-%m-%dT%H:%M:%S.%f%z")
                        if updated_at.replace(tzinfo=None) > two_months_ago:
                            recent_version = True
                            break
                    
                    if recent_version:
                        recent_top_crates.append(crate_id)
                        print(f"Found recently updated crate with high downloads: {crate_id}")

                        if len(recent_top_crates) >= limit:
                            return recent_top_crates  # Early return if we have enough crates
                    
                    success = True  # Successfully processed this crate
                    
                    # Add a delay to avoid rate limiting
                    time.sleep(1)
                    
                except Exception as e:
                    retry_count += 1
                    print(f"Error checking versions for {crate_id} (attempt {retry_count}/{max_retries}): {e}")
                    time.sleep(2 * retry_count)  # Exponential backoff
        
        # If we couldn't find enough recently updated crates, add more popular crates
        if len(recent_top_crates) < limit:
            print(f"Only found {len(recent_top_crates)} crates with recent updates, adding more popular crates")
            for crate in all_crates:
                if crate['id'] not in recent_top_crates:
                    recent_top_crates.append(crate['id'])
                    if len(recent_top_crates) >= limit:
                        break
        
        return recent_top_crates[:limit]
        
        
    except Exception as e:
        print(f"Error fetching top crates: {e}")
        return []

def save_to_json(changes: List[Dict], filename: str):
    """Save changes to a JSON file"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(changes, f, indent=2, ensure_ascii=False)
    print(f"Saved results to {filename}")



def main(crates_num=30, start_date=None, end_date=None, version_num=4):
    """
    主函数，分析crates并生成API变更报告
    
    Args:
        crates_num: 要分析的顶级crate数量
        start_date: 开始日期，只分析此日期之后的版本
        end_date: 结束日期，只分析此日期之前的版本
        version_num: 每个crate要分析的版本数量
    """
    # 获取top crates列表
    top_crates = get_top_crates(int(crates_num))
    
    all_changes = []
    
    for crate in tqdm(top_crates, desc="Analyzing crates"):
        print(f"\nAnalyzing crate: {crate}")
        
        # 获取符合日期范围的指定数量版本
        versions = get_crate_versions(crate, start_date, end_date, int(version_num))
        
        if len(versions) < 2:
            print(f"  Not enough versions for {crate} in the specified date range, skipping")
            continue
        
        print(f"  Versions to analyze: {versions}")
        
        # 分析版本之间的API变化
        for i in range(len(versions) - 1):
            old_ver = versions[i]
            new_ver = versions[i + 1]
            
            print(f"  Comparing {old_ver} → {new_ver}")
            
            # 下载并提取两个版本
            old_tar = download_crate(crate, old_ver)
            new_tar = download_crate(crate, new_ver)
            
            if not old_tar or not new_tar:
                print(f"  Failed to download {crate} {old_ver} or {new_ver}, skipping")
                continue
            
            old_src = extract_crate(old_tar)
            new_src = extract_crate(new_tar)
            
            if not old_src or not new_src:
                print(f"  Failed to extract {crate} {old_ver} or {new_ver}, skipping")
                continue
            
            try:
                # 提取API并检测变化
                print(f"  Extracting APIs from {old_ver}...")
                old_apis = extract_apis(old_src)
                print(f"  Found {len(old_apis)} APIs in {old_ver}")
                
                print(f"  Extracting APIs from {new_ver}...")
                new_apis = extract_apis(new_src)
                print(f"  Found {len(new_apis)} APIs in {new_ver}")
                
                print(f"  Detecting changes...")
                changes = detect_changes(old_apis, new_apis, crate, old_ver, new_ver)
                
                print(f"  Found {len(changes)} changes between {old_ver} and {new_ver}")
                
                # 添加至所有变更列表
                all_changes.extend(changes)
                
                # 清理临时目录
                shutil.rmtree(os.path.dirname(old_src), ignore_errors=True)
                shutil.rmtree(os.path.dirname(new_src), ignore_errors=True)
                
            except Exception as e:
                print(f"  Error analyzing {crate} {old_ver} to {new_ver}: {e}")
    
    # 保存结果
    if all_changes:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        date_range = ""
        if start_date:
            date_range += f"_from_{start_date.replace('-', '')}"
        if end_date:
            date_range += f"_to_{end_date.replace('-', '')}"
        
        filename = f"reports/crates_api_changes{date_range}_{timestamp}.json"
        save_to_json(all_changes, filename)
    else:
        print("No changes detected in any crate")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="分析Rust crate API在特定时间段的变化")
    parser.add_argument("--crates_num", type=int, default=30, help="要分析的顶级crate数量")
    parser.add_argument("--start_date", help="开始日期 (YYYY-MM-DD)，只分析此日期之后的版本")
    parser.add_argument("--end_date", help="结束日期 (YYYY-MM-DD)，只分析此日期之前的版本")
    parser.add_argument("--version_num", type=int, default=4, help="每个crate要分析的版本数量")
    args = parser.parse_args()
    
    main(args.crates_num, args.start_date, args.end_date, args.version_num)