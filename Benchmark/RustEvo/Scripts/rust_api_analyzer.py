"""
Rust官方仓库API变更检测

执行流程：
1. 下载或更新rust官方仓库
2. 解析不同版本的API元数据
3. 检测API变更（稳定化/废弃/签名变更/隐式变更）
4. 生成结构化的变更报告

需要预装:
- git
- tree-sitter (pip install tree-sitter)
- tree-sitter-rust (需从GitHub编译)

How to run:
python scripts/rust_api_analyzer.py --repo ./rust-repo --output ./reports --start 1.72.0 --end 1.84.0  
"""

import os
import re
import git
import json
import hashlib
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from tqdm import tqdm
import semver


# ==== 核心数据结构 ====
@dataclass
class APIEntity:
    name: str
    module: str
    type: str  # function/struct/enum/trait/method/associated_function
    signature: str
    documentation: str  
    examples: str  
    source_code: str
    attributes: Dict[str, str]  # 如 #[stable], #[deprecated]
    version: str

@dataclass
class APIChange:
    api: APIEntity
    change_type: str  # stabilized/deprecated/signature/implicit
    from_version: str
    to_version: str
    details: str
    old_source_code: str = "" 


# ==== 模块路径提取器 ====
class ModulePathExtractor:
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self.release_notes_cache = {}
        self._load_release_notes()
    
    def _load_release_notes(self):
        """加载发布说明，用于提取API模块路径"""
        release_notes_dir = self.repo_path / "RELEASES.md"
        if not release_notes_dir.exists():
            print(f"警告: 无法找到发布说明目录: {release_notes_dir}")
            return
        
        # 加载所有版本的release notes文件
        for version_file in release_notes_dir.glob("*.md"):
            try:
                version_match = re.search(r'(\d+\.\d+\.\d+)\.md', str(version_file))
                if version_match:
                    version = version_match.group(1)
                    with open(version_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        self.release_notes_cache[version] = content
            except Exception as e:
                print(f"无法解析发布说明 {version_file}: {e}")
    
    def get_module_path(self, file_path: Path, examples: List[str], api_name: str, version: str) -> str:
        """
        提取API的模块路径:
        1. 从示例代码中查找use语句
        2. 从发布说明中查找
        3. 从文件路径推导
        """
        # 从示例代码中查找use语句
        module_from_examples = self._extract_from_examples(examples, api_name)
        if module_from_examples:
            return module_from_examples
        
        # 从发布说明中查找
        module_from_release = self._extract_from_release_notes(api_name, version)
        if module_from_release:
            return module_from_release
        
        # 从文件路径推导
        return self._extract_from_file_path(file_path)
    
    def _extract_from_examples(self, examples: List[str], api_name: str) -> str:
        """从示例代码中提取模块路径"""
        if not examples:
            return ""
        
        joined_examples = "\n".join(examples)
        # 查找引用该API的use语句
        use_matches = re.finditer(r'use\s+([^;]+);', joined_examples)
        for match in use_matches:
            module_path = match.group(1).strip()
            # 检查模块路径是否包含API名称
            if api_name in module_path.split("::"):
                # 提取API名称前的路径部分
                parts = module_path.split("::")
                if api_name in parts:
                    idx = parts.index(api_name)
                    return "::".join(parts[:idx])
            # 如果找到以标准库为开头的路径，返回该路径
            if module_path.startswith(("std::", "core::", "alloc::")):
                return module_path
        
        return ""
    
    def _extract_from_release_notes(self, api_name: str, version: str) -> str:
        """从发布说明中提取模块路径"""
        if version not in self.release_notes_cache:
            return ""
        
        content = self.release_notes_cache[version]
        # 查找提到该API的行
        api_regex = fr'[`:]?\b{re.escape(api_name)}\b[`:]?'
        matches = re.finditer(api_regex, content)
        
        for match in matches:
            # 查找该行上下文中的标准库路径
            line_start = content.rfind('\n', 0, match.start()) + 1
            line_end = content.find('\n', match.end())
            line = content[line_start:line_end]
            
            # 提取形如std::module::submodule格式的路径
            path_matches = re.finditer(r'\b(std|core|alloc)::[a-zA-Z0-9_:]+', line)
            for path_match in path_matches:
                path = path_match.group(0)
                # 如果路径末尾是API名称，返回前面的部分
                if path.endswith(f"::{api_name}"):
                    return path[:-len(api_name) - 2]
                # 否则返回完整路径
                return path
        
        return ""
    
    def _extract_from_file_path(self, file_path: Path) -> str:
        """从文件路径推导模块路径"""
        try:
            rel_path = file_path.relative_to(self.repo_path)
            parts = list(rel_path.parts)
            
            # 识别标准库部分
            lib_indices = [i for i, part in enumerate(parts) if part in ["std", "core", "alloc"]]
            if not lib_indices:
                return ""
            
            lib_idx = lib_indices[0]
            lib_type = parts[lib_idx]
            
            # 找到src目录后的部分
            try:
                src_idx = parts.index("src", lib_idx)
                module_parts = [lib_type] + parts[src_idx + 1:]
                
                # 处理文件名
                if module_parts[-1].endswith('.rs'):
                    module_parts[-1] = module_parts[-1][:-3]
                if module_parts[-1] == 'mod':
                    module_parts.pop()
                
                return "::".join(module_parts)
            except ValueError:
                return lib_type
        except Exception:
            return ""


# ==== Rust语法解析器 ====
from tree_sitter import Language, Parser

class RustASTParser:
    def __init__(self):
        """初始化Tree-sitter解析器"""
        try:
            # 尝试加载已编译的语言库
            rust_lib_path = 'build/rust.so'
            if not os.path.exists(rust_lib_path):
                print("正在编译tree-sitter-rust...")
                self._build_language()
            
            self.RUST_LANG = Language(rust_lib_path, 'rust')
            self.parser = Parser()
            self.parser.set_language(self.RUST_LANG)
        except Exception as e:
            print(f"初始化Tree-sitter解析器失败: {e}")
            raise
    
    def _build_language(self):
        """编译tree-sitter-rust语言库"""
        # 创建构建目录
        os.makedirs("build", exist_ok=True)
        
        # 检查tree-sitter-rust是否已克隆
        if not os.path.exists("tree-sitter-rust"):
            subprocess.run([
                "git", "clone", "https://github.com/tree-sitter/tree-sitter-rust.git"
            ], check=True)
        
        # 编译语言库
        from tree_sitter import Language
        Language.build_library(
            'build/rust.so',
            ['tree-sitter-rust']
        )
    
    def parse_file(self, code: str) -> dict:
        """解析Rust源代码文件"""
        tree = self.parser.parse(bytes(code, "utf-8"))
        return self._walk_tree(tree.root_node)
    
    def _walk_tree(self, node, current_module="") -> dict:
        """遍历AST，提取API定义"""
        result = {
            "module": current_module,
            "functions": [],
            "structs": [],
            "enums": [],
            "traits": [],
            "impls": [],
            "macros": []
        }
        
        self._process_node(node, result, current_module)
        return result
    
    def _process_node(self, node, result, current_module):
        """处理AST节点"""
        if node is None:
            return
        
        if node.type == "module_declaration":
            # 处理模块声明
            name_node = node.child_by_field_name("name")
            if name_node:
                mod_name = name_node.text.decode('utf-8')
                new_module = f"{current_module}::{mod_name}" if current_module else mod_name
                
                # 处理内联模块
                body_node = node.child_by_field_name("body")
                if body_node:
                    for child in body_node.children:
                        self._process_node(child, result, new_module)
        
        elif node.type == "function_item":
            # 处理函数定义
            function = self._parse_function(node, current_module)
            if function:
                result["functions"].append(function)
        
        elif node.type == "struct_item":
            # 处理结构体定义
            struct = self._parse_struct(node, current_module)
            if struct:
                result["structs"].append(struct)
        
        elif node.type == "enum_item":
            # 处理枚举定义
            enum = self._parse_enum(node, current_module)
            if enum:
                result["enums"].append(enum)
        
        elif node.type == "trait_item":
            # 处理trait定义
            trait = self._parse_trait(node, current_module)
            if trait:
                result["traits"].append(trait)
        
        elif node.type == "impl_item":
            # 处理impl块
            impl = self._parse_impl(node, current_module)
            if impl:
                result["impls"].append(impl)
        
        elif node.type == "macro_definition":
            # 处理宏定义
            macro = self._parse_macro(node, current_module)
            if macro:
                result["macros"].append(macro)
        
        # 处理子节点
        if node.type not in ["module_declaration"]:  # 避免重复处理内联模块
            for child in node.children:
                self._process_node(child, result, current_module)
    
    def _parse_function(self, node, module_path):
        """解析函数定义"""
        try:
            name_node = node.child_by_field_name("name")
            if not name_node:
                return None
            
            name = name_node.text.decode('utf-8')
            
            # 处理属性
            attributes = self._parse_attributes(node)
            
            # 提取文档注释
            documentation, examples = self._parse_docs(node)
            
            # 构建函数签名
            signature = f"fn {name}"
            param_list = node.child_by_field_name("parameters")
            if param_list:
                params = []
                for param in param_list.children:
                    if param.type == "parameter":
                        param_text = param.text.decode('utf-8')
                        params.append(param_text)
                signature += f"({', '.join(params)})"
            else:
                signature += "()"
            
            # 提取返回类型
            return_type = node.child_by_field_name("return_type")
            if return_type:
                signature += f" -> {return_type.text.decode('utf-8')}"
            
            # 提取源代码
            source_code = node.text.decode('utf-8')
            
            return {
                "name": name,
                "module": module_path,
                "type": "function",
                "signature": signature,
                "documentation": documentation,
                "examples": examples,
                "source_code": source_code,
                # "attributes": attributes
            }
        except Exception as e:
            print(f"解析函数失败: {e}")
            return None
    
    def _parse_struct(self, node, module_path):
        """解析结构体定义"""
        try:
            name_node = node.child_by_field_name("name")
            if not name_node:
                return None
            
            name = name_node.text.decode('utf-8')
            
            # 处理属性
            attributes = self._parse_attributes(node)
            
            # 提取文档注释
            documentation, examples = self._parse_docs(node)
            
            # 提取源代码
            source_code = node.text.decode('utf-8')
            
            return {
                "name": name,
                "module": module_path,
                "type": "struct",
                "signature": f"struct {name}",
                "documentation": documentation,
                "examples": examples,
                "source_code": source_code,
                # "attributes": attributes
            }
        except Exception as e:
            print(f"解析结构体失败: {e}")
            return None
    
    def _parse_enum(self, node, module_path):
        """解析枚举定义"""
        try:
            name_node = node.child_by_field_name("name")
            if not name_node:
                return None
            
            name = name_node.text.decode('utf-8')
            
            # 处理属性
            attributes = self._parse_attributes(node)
            
            # 提取文档注释
            documentation, examples = self._parse_docs(node)
            
            # 提取源代码
            source_code = node.text.decode('utf-8')
            
            return {
                "name": name,
                "module": module_path,
                "type": "enum",
                "signature": f"enum {name}",
                "documentation": documentation,
                "examples": examples,
                "source_code": source_code,
                # "attributes": attributes
            }
        except Exception as e:
            print(f"解析枚举失败: {e}")
            return None
    
    def _parse_trait(self, node, module_path):
        """解析trait定义"""
        try:
            name_node = node.child_by_field_name("name")
            if not name_node:
                return None
            
            name = name_node.text.decode('utf-8')
            
            # 处理属性
            attributes = self._parse_attributes(node)
            
            # 提取文档注释
            documentation, examples = self._parse_docs(node)
            
            # 提取源代码
            source_code = node.text.decode('utf-8')
            
            # 处理trait内的方法
            methods = []
            body_node = node.child_by_field_name("body")
            if body_node:
                for child in body_node.children:
                    if child.type == "function_item":
                        method = self._parse_trait_method(child, name)
                        if method:
                            methods.append(method)
            
            return {
                "name": name,
                "module": module_path,
                "type": "trait",
                "signature": f"trait {name}",
                "documentation": documentation,
                "examples": examples,
                "source_code": source_code,
                # "attributes": attributes,
                "methods": methods
            }
        except Exception as e:
            print(f"解析trait失败: {e}")
            return None
    
    def _parse_trait_method(self, node, trait_name):
        """解析trait中的方法"""
        try:
            name_node = node.child_by_field_name("name")
            if not name_node:
                return None
            
            name = name_node.text.decode('utf-8')
            
            # 处理属性
            attributes = self._parse_attributes(node)
            
            # 提取文档注释
            documentation, examples = self._parse_docs(node)
            
            # 构建方法签名
            signature = f"fn {name}"
            param_list = node.child_by_field_name("parameters")
            if param_list:
                params = []
                for param in param_list.children:
                    if param.type == "parameter":
                        param_text = param.text.decode('utf-8')
                        params.append(param_text)
                signature += f"({', '.join(params)})"
            else:
                signature += "()"
            
            # 提取返回类型
            return_type = node.child_by_field_name("return_type")
            if return_type:
                signature += f" -> {return_type.text.decode('utf-8')}"
            
            # 提取源代码
            source_code = node.text.decode('utf-8')
            
            # 判断方法类型 (是否有self参数)
            is_method = False
            if param_list:
                for param in param_list.children:
                    if param.type == "self_parameter":
                        is_method = True
                        break
            
            return {
                "name": name,
                "trait": trait_name,
                "type": "trait_method" if is_method else "trait_associated_fn",
                "signature": signature,
                "documentation": documentation,
                "examples": examples,
                "source_code": source_code,
                # "attributes": attributes
            }
        except Exception as e:
            print(f"解析trait方法失败: {e}")
            return None
    
    def _parse_impl(self, node, module_path):
        """解析impl块"""
        try:
            # 提取实现的类型名称
            type_node = node.child_by_field_name("type")
            if not type_node:
                return None
            
            type_name = type_node.text.decode('utf-8')
            
            # 处理impl块内的方法
            methods = []
            for child in node.children:
                if child.type == "function_item":
                    method = self._parse_impl_method(child, type_name)
                    if method:
                        methods.append(method)
            
            return {
                "type_name": type_name,
                "module": module_path,
                "methods": methods
            }
        except Exception as e:
            print(f"解析impl块失败: {e}")
            return None
    
    def _parse_impl_method(self, node, type_name):
        """解析impl块中的方法"""
        try:
            name_node = node.child_by_field_name("name")
            if not name_node:
                return None
            
            name = name_node.text.decode('utf-8')
            
            # 处理属性
            attributes = self._parse_attributes(node)
            
            # 提取文档注释
            documentation, examples = self._parse_docs(node)
            
            # 构建方法签名
            signature = f"fn {name}"
            param_list = node.child_by_field_name("parameters")
            
            # 判断方法类型 (是否有self参数)
            is_method = False
            params = []
            if param_list:
                for param in param_list.children:
                    if param.type == "self_parameter":
                        is_method = True
                        params.append(param.text.decode('utf-8'))
                    elif param.type == "parameter":
                        param_text = param.text.decode('utf-8')
                        params.append(param_text)
                signature += f"({', '.join(params)})"
            else:
                signature += "()"
            
            # 提取返回类型
            return_type = node.child_by_field_name("return_type")
            if return_type:
                signature += f" -> {return_type.text.decode('utf-8')}"
            
            # 提取源代码
            source_code = node.text.decode('utf-8')
            
            return {
                "name": name,
                "impl_type": type_name,
                "type": "method" if is_method else "associated_function",
                "signature": signature,
                "documentation": documentation,
                "examples": examples,
                "source_code": source_code,
                # "attributes": attributes
            }
        except Exception as e:
            print(f"解析impl方法失败: {e}")
            return None
    
    def _parse_macro(self, node, module_path):
        """解析宏定义"""
        try:
            name_node = node.child_by_field_name("name")
            if not name_node:
                return None
            
            name = name_node.text.decode('utf-8')
            
            # 处理属性
            attributes = self._parse_attributes(node)
            
            # 提取文档注释
            documentation, examples = self._parse_docs(node)
            
            # 提取源代码
            source_code = node.text.decode('utf-8')
            
            return {
                "name": name,
                "module": module_path,
                "type": "macro",
                "signature": f"macro_rules! {name}",
                "documentation": documentation,
                "examples": examples,
                "source_code": source_code,
                # "attributes": attributes
            }
        except Exception as e:
            print(f"解析宏失败: {e}")
            return None
    
    def _parse_attributes(self, node):
        """解析节点属性（如#[stable], #[deprecated]）"""
        attributes = {}
        
        # 检查前面的属性
        prev_sibling = node.prev_sibling
        while prev_sibling:
            if prev_sibling.type == "attribute_item":
                attr_text = prev_sibling.text.decode('utf-8')
                
                # 处理stable属性
                stable_match = re.search(r'#\[\s*stable\s*\(\s*feature\s*=\s*"([^"]+)"\s*,\s*since\s*=\s*"([^"]+)"\s*\)\s*\]', attr_text)
                if stable_match:
                    feature, version = stable_match.groups()
                    attributes['stable'] = {'feature': feature, 'version': version}
                
                # 处理rustc_const_stable属性
                const_stable_match = re.search(r'#\[\s*rustc_const_stable\s*\(\s*feature\s*=\s*"([^"]+)"\s*,\s*since\s*=\s*"([^"]+)"\s*\)\s*\]', attr_text)
                if const_stable_match:
                    feature, version = const_stable_match.groups()
                    attributes['const_stable'] = {'feature': feature, 'version': version}
                    # 同时标记为stable以便于分析
                    if 'stable' not in attributes:
                        attributes['stable'] = {'feature': feature, 'version': version}
                
                # 处理unstable属性
                unstable_match = re.search(r'#\[\s*unstable\s*\(\s*feature\s*=\s*"([^"]+)"\s*,\s*issue\s*=\s*"([^"]+)"\s*(?:,\s*reason\s*=\s*"([^"]+)")?\s*\)\s*\]', attr_text)
                if unstable_match:
                    groups = unstable_match.groups()
                    feature = groups[0]
                    issue = groups[1]
                    reason = groups[2] if len(groups) > 2 else None
                    attributes['unstable'] = {'feature': feature, 'issue': issue, 'reason': reason}
                
                # 处理rustc_const_unstable属性
                const_unstable_match = re.search(r'#\[\s*rustc_const_unstable\s*\(\s*feature\s*=\s*"([^"]+)"\s*,\s*issue\s*=\s*"([^"]+)"\s*\)\s*\]', attr_text)
                if const_unstable_match:
                    feature, issue = const_unstable_match.groups()
                    attributes['const_unstable'] = {'feature': feature, 'issue': issue}
                    # 同时标记为unstable以便于分析
                    if 'unstable' not in attributes:
                        attributes['unstable'] = {'feature': feature, 'issue': issue, 'reason': None}
                
                # 处理deprecated属性 - Use a more robust pattern
                deprecated_pattern = re.compile(
                    r'#\[\s*deprecated\s*\(\s*'
                    r'(?:[\s\n]*since\s*=\s*"([^"]+)"\s*,?)?'
                    r'(?:[\s\n]*note\s*=\s*"((?:[^"]|\\")*)"\s*,?)?'
                    r'(?:[\s\n]*suggestion\s*=\s*"([^"]+)"\s*,?)?'
                    r'[\s\n]*\)\s*\]',
                    re.DOTALL
                )
                
                deprecated_match = deprecated_pattern.search(attr_text)
                if deprecated_match:
                    since = deprecated_match.group(1)
                    note = deprecated_match.group(2) if deprecated_match.group(2) else None
                    
                    # 处理转义引号
                    if note:
                        note = note.replace(r'\"', '"')
                    
                    # 确保不是allow(deprecated)
                    if not re.search(r'allow\s*\(\s*deprecated\s*\)', attr_text):
                        attributes['deprecated'] = {
                            'since': since.strip() if since else None,
                            'note': note.strip() if note else None
                        }
                
            prev_sibling = prev_sibling.prev_sibling
        
        return attributes

    def _clean_doc_block(self, lines):
        """清理文档块"""
        cleaned = []
        prev_empty = False
        for line in lines:
            if line.strip() == "":
                if not prev_empty:
                    cleaned.append("")
                    prev_empty = True
            else:
                cleaned.append(line)
                prev_empty = False
        return cleaned   
    
    def _clean_code_block(self, lines):
        """清理代码块"""
        code_lines = []
        in_code = False
        for line in lines:
            if line == "```rust":
                in_code = True
                continue
            if line == "```":
                in_code = False
                continue
            if in_code:
                code_lines.append(line)
        return code_lines


    # def _parse_docs(self, node):
    #     """解析文档注释和示例，返回合并后的字符串"""
    #     documentation = []
    #     examples = []
    #     current_block = []
    #     in_code_block = False
    #     code_lang = ""

    #     prev_sibling = node.prev_sibling
    #     while prev_sibling:
    #         if prev_sibling.type == "line_comment":
    #             line = prev_sibling.text.decode('utf-8').strip()
    #             if line.startswith("///"):
    #                 content = line[3:].strip()
                    
    #                 # 处理代码块标记
    #                 if content.startswith("```"):
    #                     lang_match = re.match(r'^```(\S*)', content)
    #                     code_lang = lang_match.group(1) if lang_match else ""
                        
    #                     if not in_code_block:
    #                         in_code_block = True
    #                         current_block = [f"```{code_lang}"]
    #                     else:
    #                         current_block.append("```")
    #                         examples.append("\n".join(current_block))
    #                         current_block = []
    #                         in_code_block = False
    #                     continue
                    
    #                 # 处理示例章节
    #                 if content.lower().startswith("# examples"):
    #                     in_examples_section = True
    #                     current_block = []
    #                     continue
                    
    #                 # 合并到当前块
    #                 current_block.append(content)
                    
    #                 # 非代码块内容添加换行
    #                 if not in_code_block and not content.startswith("#"):
    #                     current_block.append("")
    #             elif line.startswith("//!"):
    #                 # 处理模块级注释
    #                 pass
                
    #         elif prev_sibling.type == "attribute_item":
    #             pass
    #         else:
    #             break
            
        #         prev_sibling = prev_sibling.prev_sibling
    def _parse_docs(self, node):
        """解析文档注释，准确提取完整代码块"""
        documentation = []
        examples = []
        current_code_block = []  # 当前正在处理的代码块
        in_code_block = False    # 是否在代码块中
        code_lang = ""           # 代码块语言标识
        in_examples_section = False  # 新增：是否在Examples章节

        prev_sibling = node.prev_sibling
        while prev_sibling:
            if prev_sibling.type == "line_comment":
                line = prev_sibling.text.decode('utf-8').strip()
                
                # 仅处理///注释
                if line.startswith("///"):
                    content = line[3:].strip()  # 移除注释标记
                    
                    # 检测Examples章节标题
                    if re.match(r'^#+\s*examples?', content, re.IGNORECASE):
                        in_examples_section = True
                        continue
                    elif in_examples_section and content.startswith("#"):
                        # 其他章节标题，退出Examples章节
                        in_examples_section = False

                    # 代码块处理逻辑
                    if content.startswith("```"):
                        lang_match = re.match(r'^```(\S*)', content)
                        code_lang = lang_match.group(1) if lang_match else ""
                        
                        if in_code_block:
                            # 结束代码块
                            in_code_block = False
                            if current_code_block:
                                # 添加代码块闭合标记
                                current_code_block.append("```")
                                full_code = "\n".join(current_code_block)
                                if in_examples_section:
                                    examples.append(full_code)  # 完整代码块
                                else:
                                    documentation.append(full_code)
                                current_code_block = []
                        else:
                            # 开始新代码块
                            in_code_block = True
                            current_code_block.append(f"```{code_lang}")
                    elif in_code_block:
                        # 代码块内容行
                        current_code_block.append(content)
                    else:
                        # 普通文档内容
                        if in_examples_section:
                            # Examples章节的非代码内容视为描述文本
                            pass
                        else:
                            documentation.append(content)
            
            prev_sibling = prev_sibling.prev_sibling

        # 处理未闭合的代码块
        if in_code_block and current_code_block:
            current_code_block.append("```")  # 强制闭合
            examples.append("\n".join(current_code_block))

        # # 处理最后一个块
        # if in_examples_section:
        #     examples = self._clean_code_block(current_block)
        # else:
        #     documentation = self._clean_doc_block(current_block)

        return "\n".join(documentation), "\n---\n".join(examples)


# ==== 变更检测引擎 ====
class APIChangeDetector:
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self.ast_parser = RustASTParser()
        self.module_extractor = ModulePathExtractor(repo_path)
    
    def checkout_version(self, version: str):
        """检出指定版本的代码"""
        try:
            repo = git.Repo(self.repo_path)
            print(f"检出版本: {version}")
            if version in repo.tags:
                repo.git.checkout(version)
            else:
                repo.git.checkout(f'tags/{version}')
            return True
        except Exception as e:
            print(f"检出版本 {version} 失败: {e}")
            return False

    def find_api_files(self):
        """查找标准库中的API定义文件"""
        std_paths = [
            self.repo_path / "library" / "std",
            self.repo_path / "library" / "core",
            self.repo_path / "library" / "alloc"
        ]
        
        api_files = []
        for path in std_paths:
            if path.exists():
                api_files.extend(path.glob("**/*.rs"))
        
        return api_files
    

    def extract_apis_from_file(self, file_path: Path) -> List[APIEntity]:
        """从文件中提取API实体，识别稳定和废弃的API"""
        apis = []
        try:
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Use a more robust pattern for finding stable attributes
            stable_pattern = re.compile(
                r'#\[\s*stable\s*\(\s*feature\s*=\s*"([^"]+)"\s*,\s*since\s*=\s*"([^"]+)"\s*\)\s*\]',
                re.DOTALL
            )
            
            # Define the deprecated pattern at function level so it's available throughout
            deprecated_pattern = re.compile(
                r'#\[\s*deprecated\s*\(\s*'
                r'(?:[\s\n]*since\s*=\s*"([^"]+)"\s*,?)?'
                r'(?:[\s\n]*note\s*=\s*"((?:[^"]|\\")*)"\s*,?)?'
                r'(?:[\s\n]*suggestion\s*=\s*"([^"]+)"\s*,?)?'
                r'[\s\n]*\)\s*\]',
                re.DOTALL
            )
            
            # 查找所有stable属性标记的API
            stable_matches = list(stable_pattern.finditer(content))
            for stable_match in stable_matches:
                feature, version = stable_match.groups()
                match_end_pos = stable_match.end()
                
                # 提取API信息
                api_info = self._extract_api_info(content, match_end_pos, file_path, version)
                if api_info:
                    # 检查API是否被标记为废弃
                    # 在stable属性后查找是否有deprecated属性
                    chunk = content[match_end_pos:match_end_pos + 1000]  # Increase search range from 500 to 1000
                    
                    deprecated_match = deprecated_pattern.search(chunk)
                    
                    if deprecated_match:
                        # 确保这是实际的deprecated属性，而不是allow(deprecated)
                        line_start = chunk.rfind('\n', 0, deprecated_match.start()) + 1
                        line = chunk[line_start:deprecated_match.start()].strip()
                        
                        # 只有当这是独立的属性行时才将其识别为deprecated
                        if not (line and '#[allow(' in line):
                            since_version = deprecated_match.group(1) if deprecated_match.group(1) else ""
                            note = deprecated_match.group(2) if deprecated_match.group(2) else ""
                            
                            # 添加deprecated信息到API属性
                            api_info.attributes['deprecated'] = {
                                'since': since_version,
                                'note': note
                            }
                    
                    apis.append(api_info)
            
            # 也检查没有stable标记但直接使用deprecated标记的API
            # 使用相同的robust pattern
            for match in deprecated_pattern.finditer(content):
                # 确保这不是在allow语句中
                line_start = content.rfind('\n', 0, match.start()) + 1
                line = content[line_start:match.start()].strip()
                
                if '#[allow(' in line:
                    continue  # 跳过allow内部的deprecated
                
                since_version = match.group(1) if match.group(1) else ""
                note = match.group(2) if match.group(2) else ""
                
                # 检查是否已经作为stable+deprecated对处理过
                nearby_stable = False
                for api in apis:
                    if api.source_code in content[match.end():match.end()+1000]:
                        nearby_stable = True
                        break
                
                if nearby_stable:
                    continue
                
                # 提取API信息
                api_info = self._extract_api_info(content, match.end(), file_path, since_version, 
                                                deprecated=True, deprecated_note=note)
                if api_info:
                    apis.append(api_info)
                        
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
        
        return apis
    

    def _extract_api_info(self, content: str, start_pos: int, file_path: Path, version: str, 
                          deprecated=False, deprecated_note="") -> Optional[APIEntity]:
        """提取API详情"""
        try:
            # 向上搜索文档注释
            doc_start = self._find_doc_start(content, start_pos)
            doc_lines = []
            prev_lines = content[:doc_start].split('\n')
            for line in reversed(prev_lines):
                line = line.strip()
                if line.startswith('///') or line.startswith('//!'):
                    doc_lines.insert(0, line)
                elif line and not line.startswith('#['):
                    break
            
            # 解析文档和示例
            documentation = []
            examples = []
            in_examples = False
            
            for line in doc_lines:
                if line.startswith('///'):
                    content_line = line[3:].strip()
                elif line.startswith('//!'):
                    content_line = line[3:].strip()
                else:
                    continue
                
                if content_line.startswith('# Examples') or content_line == '# Example':
                    in_examples = True
                    continue
                elif content_line.startswith('# '):
                    in_examples = False
                
                if in_examples:
                    examples.append(content_line)
                else:
                    documentation.append(content_line)
            
            # 提取API代码
            api_code = self._extract_source_code(content, start_pos)
            if not api_code:
                return None
            
            # 尝试提取签名和名称
            name, api_type, signature = self._parse_signature(api_code)
            if not name:
                return None
            
            # 解析属性
            attributes = {}
            if deprecated:
                attributes['deprecated'] = {'since': version, 'note': deprecated_note}
            else:
                attributes['stable'] = {'version': version}
            
            # 提取模块路径
            module_path = self.module_extractor.get_module_path(file_path, examples, name, version)
            
            return APIEntity(
                name=name,
                module=module_path,
                type=api_type,
                signature=signature,
                documentation="\n".join(documentation),
                examples=examples,
                source_code=api_code,
                attributes=attributes,
                version=version
            )
        except Exception as e:
            print(f"提取API信息失败: {e}")
            return None
    
    def _find_doc_start(self, content: str, attr_start: int, max_lookback: int = 50) -> int:
        """查找文档注释的起始位置"""
        lines = content[:attr_start].split('\n')
        doc_start = attr_start
        
        for i in range(len(lines) - 1, max(-1, len(lines) - max_lookback - 1), -1):
            line = lines[i].strip()
            if not line or (not line.startswith('///') and not line.startswith('//!') and not line.startswith('#[')):
                doc_start = sum(len(l) + 1 for l in lines[:i+1])
                break
                
        return doc_start
    
    def _extract_source_code(self, content: str, start_pos: int) -> str:
        """提取完整的源代码，包括函数体"""
        lines = content[start_pos:].split('\n')
        source_lines = []
        bracket_count = 0
        started = False
        
        for line in lines:
            stripped = line.strip()
            
            # 跳过纯注释行和空行，直到找到实际代码
            if not started:
                if stripped and not stripped.startswith('///') and not stripped.startswith('//') and not stripped.startswith('#['):
                    started = True
                elif stripped.startswith('#['):
                    source_lines.append(line)
                    continue
                    
            if started:
                source_lines.append(line)
                bracket_count += line.count('{')
                bracket_count -= line.count('}')
                if bracket_count > 0:
                    continue
                elif bracket_count == 0 and len(source_lines) > 0:
                    # 确保至少包含了一个完整的代码块
                    if any('{' in l for l in source_lines):
                        break
                        
        return '\n'.join(source_lines).rstrip()
    
    def _parse_signature(self, source_code: str) -> Tuple[str, str, str]:
        """解析API签名，返回(名称，类型，签名)"""
        # 移除前导属性
        lines = source_code.split('\n')
        code_lines = []
        for line in lines:
            if not line.strip().startswith('#['):
                code_lines.append(line)
        
        code = '\n'.join(code_lines)
        
        # 检查是否是函数定义
        fn_match = re.search(r'(pub\s+)?(?:unsafe\s+)?fn\s+([a-zA-Z_][a-zA-Z0-9_]*)', code)
        if fn_match:
            name = fn_match.group(2)
            
            # 检查是否有self参数
            if re.search(r'\(\s*&?(?:mut\s+)?self\b', code):
                api_type = "method"
            else:
                api_type = "function"
            
            # 提取函数签名（到第一个{前）
            signature_line = code[:code.find('{')].strip() if '{' in code else code.strip()
            return name, api_type, signature_line
        
        # 检查是否是结构体定义
        struct_match = re.search(r'(pub\s+)?struct\s+([a-zA-Z_][a-zA-Z0-9_]*)', code)
        if struct_match:
            name = struct_match.group(2)
            signature_line = code[:code.find('{')].strip() if '{' in code else code.strip()
            return name, "struct", signature_line
        
        # 检查是否是枚举定义
        enum_match = re.search(r'(pub\s+)?enum\s+([a-zA-Z_][a-zA-Z0-9_]*)', code)
        if enum_match:
            name = enum_match.group(2)
            signature_line = code[:code.find('{')].strip() if '{' in code else code.strip()
            return name, "enum", signature_line
        
        # 检查是否是trait定义
        trait_match = re.search(r'(pub\s+)?trait\s+([a-zA-Z_][a-zA-Z0-9_]*)', code)
        if trait_match:
            name = trait_match.group(2)
            signature_line = code[:code.find('{')].strip() if '{' in code else code.strip()
            return name, "trait", signature_line
        
        # 检查是否是impl块方法
        impl_method_match = re.search(r'impl\s+.*?\{\s*.*?(pub\s+)?(?:unsafe\s+)?fn\s+([a-zA-Z_][a-zA-Z0-9_]*)', code, re.DOTALL)
        if impl_method_match:
            name = impl_method_match.group(2)
            
            # 检查是否有self参数
            if re.search(r'\(\s*&?(?:mut\s+)?self\b', code):
                api_type = "method"
            else:
                api_type = "associated_function"
            
            # 提取方法签名（这里简化处理）
            fn_start = code.find('fn')
            if fn_start != -1:
                signature_part = code[fn_start:]
                signature_line = signature_part[:signature_part.find('{')].strip() if '{' in signature_part else signature_part.strip()
                return name, api_type, signature_line
        
        return "", "", ""
    
    def extract_apis_from_version(self, version: str) -> Dict[str, APIEntity]:
        """从指定版本中提取所有API"""
        if not self.checkout_version(version):
            return {}
        
        apis = {}
        api_files = self.find_api_files()
        
        print(f"正在从版本 {version} 中提取API...")
        for file_path in tqdm(api_files):
            file_apis = self.extract_apis_from_file(file_path)
            for api in file_apis:
                key = f"{api.module}::{api.name}" if api.module else api.name
                apis[key] = api
        
        return apis
    


    def detect_changes(self, from_version: str, to_version: str) -> List[APIChange]:
        """检测两个版本之间的API变更，正确处理新增的deprecated标记"""
        # 提取两个版本的API
        old_apis = self.extract_apis_from_version(from_version)
        new_apis = self.extract_apis_from_version(to_version)
        
        changes = []
        
        # 检测API变更
        print("检测API变更...")
        
        # 1. 稳定化API (Stabilized): 新增加的稳定API或从unstable转为stable的API
        for key, new_api in new_apis.items():
            if 'stable' in new_api.attributes:
                stable_info = new_api.attributes['stable']
                stable_version = stable_info.get('version', '')
                
                # 只检测在我们版本范围内稳定化的API
                if self._is_version_in_range(stable_version, from_version, to_version):
                    # 新增的API
                    if key not in old_apis:
                        changes.append(APIChange(
                            api=new_api,
                            change_type="stabilized",
                            from_version=from_version,
                            to_version=to_version,
                            details=f"New API stabilized in version {stable_version}"
                        ))
                    # 从unstable转为stable的API
                    elif 'stable' not in old_apis[key].attributes and 'unstable' in old_apis[key].attributes:
                        changes.append(APIChange(
                            api=new_api,
                            change_type="stabilized",
                            from_version=from_version,
                            to_version=to_version,
                            details=f"API stabilized in version {stable_version}, previously unstable"
                        ))
        
        # 2. 废弃API (Deprecated): 只检测在from_version到to_version之间新增deprecated标记的API
        for key, new_api in new_apis.items():
            if 'deprecated' in new_api.attributes:
                # 查看旧版本是否存在该API
                if key not in old_apis:
                    # 如果旧版本没有这个API，但它在新版本中已被标记为deprecated，则这可能是在此区间内新增并立即废弃的API
                    deprecated_info = new_api.attributes['deprecated']
                    deprecated_version = deprecated_info.get('since', '')
                    
                    if deprecated_version and self._is_version_in_range(deprecated_version, from_version, to_version):
                        changes.append(APIChange(
                            api=new_api,
                            change_type="deprecated",
                            from_version=from_version,
                            to_version=to_version,
                            details=f"New API immediately deprecated in version {deprecated_version}: {deprecated_info.get('note', 'No reason provided')}"
                        ))
                else:
                    # 只有当旧版本中没有deprecated标记时才算作变更
                    if 'deprecated' not in old_apis[key].attributes:
                        deprecated_info = new_api.attributes['deprecated']
                        deprecated_version = deprecated_info.get('since', '')
                        
                        # 检查deprecated版本是否有效且在指定区间
                        if deprecated_version:
                            try:
                                # 如果deprecated_version就是to_version，或者在from_version和to_version之间，才记录变更
                                if (deprecated_version == to_version or 
                                    deprecated_version == from_version or
                                    self._is_version_in_range(deprecated_version, from_version, to_version)):
                                    changes.append(APIChange(
                                        api=new_api,
                                        change_type="deprecated",
                                        from_version=from_version,
                                        to_version=to_version,
                                        details=f"API deprecated in version {deprecated_version}: {deprecated_info.get('note', 'No reason provided')}",
                                        old_source_code=old_apis[key].source_code
                                    ))
                            except ValueError:
                                # 如果版本号无效，记录但不添加到变更中
                                print(f"警告: API {key} 有无效的废弃版本号: {deprecated_version}")
                                continue
        
        # 3. 签名变更 (Signature): 使用标准化后的签名比较
        for key in set(old_apis.keys()) & set(new_apis.keys()):
            # 跳过已经检测到的变更
            if any(c.api.name == new_apis[key].name and c.api.module == new_apis[key].module for c in changes):
                continue
                
            old_api = old_apis[key]
            new_api = new_apis[key]
            
            # 标准化签名后再比较
            old_normalized = self._normalize_signature(old_api.signature)
            new_normalized = self._normalize_signature(new_api.signature)
            
            if old_normalized != new_normalized:
                changes.append(APIChange(
                    api=new_api,
                    change_type="signature",
                    from_version=from_version,
                    to_version=to_version,
                    details=f"Signature changed from `{old_api.signature}` to `{new_api.signature}`",
                    old_source_code=old_api.source_code
                ))
            # 4. 隐式变更 (Implicit): 签名未变但行为可能变化
            elif self._detect_implicit_change(old_api, new_api):
                changes.append(APIChange(
                    api=new_api,
                    change_type="implicit",
                    from_version=from_version,
                    to_version=to_version,
                    details="API behavior may have changed (implementation or documentation has significant changes)",
                    old_source_code=old_api.source_code
                ))
        
        return changes


    def _is_version_in_range(self, version: str, from_version: str, to_version: str) -> bool:
        """检查版本是否在指定范围内，包括边界值"""
        try:
            ver = semver.VersionInfo.parse(version)
            from_ver = semver.VersionInfo.parse(from_version)
            to_ver = semver.VersionInfo.parse(to_version)
            
            return from_ver <= ver <= to_ver
        except ValueError:
            # 版本格式无效
            return False
    
    def _normalize_signature(self, signature: str) -> str:
        """标准化签名，消除格式差异"""
        # 移除注释
        signature = re.sub(r'//.*$', '', signature, flags=re.MULTILINE)
        # 将多个空白字符替换为单个空格
        signature = re.sub(r'\s+', ' ', signature)
        # 移除括号、逗号、冒号周围的空格
        signature = re.sub(r'\s*([(),:])\s*', r'\1', signature)
        # 移除首尾空白
        signature = signature.strip()
        return signature
    
    def _detect_implicit_change(self, old_api: APIEntity, new_api: APIEntity) -> bool:
        """检测API是否有隐式功能变更"""
        # 提取并标准化函数体
        old_body = self._extract_function_body(old_api.source_code)
        new_body = self._extract_function_body(new_api.source_code)
        
        if old_body != new_body:
            # 标准化代码后比较
            old_normalized = self._normalize_code(old_body)
            new_normalized = self._normalize_code(new_body)
            
            if old_normalized != new_normalized:
                # 计算代码相似度
                similarity = self._code_similarity(old_normalized, new_normalized)
                # 相似度阈值提高到0.85，减少误报
                if similarity < 0.85:
                    return True
        
        # 检查文档变化是否暗示行为变更
        if old_api.documentation != new_api.documentation:
            # 关注更明确的行为变更关键词
            behavior_phrases = [
                'breaking change', 'behavior change', 'now returns', 
                'now behaves', 'changed behavior', 'panic', 'differently'
            ]
            
            old_doc = old_api.documentation.lower()
            new_doc = new_api.documentation.lower()
            
            # 只有新文档包含而旧文档不包含的关键词才考虑
            for phrase in behavior_phrases:
                if phrase in new_doc and phrase not in old_doc:
                    return True
        
        return False

    def _normalize_code(self, code: str) -> str:
        """标准化代码，消除格式差异"""
        # 移除注释
        code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
        # 移除多余空白
        code = re.sub(r'\s+', ' ', code)
        # 移除首尾空白
        code = code.strip()
        return code

    def _extract_function_body(self, code: str) -> str:
        """提取函数体"""
        # 查找第一个大括号
        open_brace = code.find('{')
        if open_brace == -1:
            return code
        
        # 计算括号嵌套来找到匹配的闭括号
        count = 1
        for i in range(open_brace + 1, len(code)):
            if code[i] == '{':
                count += 1
            elif code[i] == '}':
                count -= 1
                if count == 0:
                    return code[open_brace:i+1]
        
        return code[open_brace:]

    def _code_similarity(self, code1: str, code2: str) -> float:
        """计算代码相似度"""
        # 对代码进行分词
        tokens1 = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*|[+\-*/=<>!&|^~%]+|[\(\){}\[\];,]|\d+', code1)
        tokens2 = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*|[+\-*/=<>!&|^~%]+|[\(\){}\[\];,]|\d+', code2)
        
        # 计算Jaccard相似度
        set1 = set(tokens1)
        set2 = set(tokens2)
        
        if not set1 and not set2:
            return 1.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0

# ==== 数据采集执行器 ====
class DataCollector:
    def __init__(self, repo_url="https://github.com/rust-lang/rust.git", repo_path="./rust-repo"):
        self.repo_url = repo_url
        self.repo_path = Path(repo_path)
        self.detector = None  # 延迟初始化检测器
    
    def setup_repo(self):
        """设置或更新Rust仓库"""
        # After cloning in setup_repo method:
        if not self.repo_path.exists():
            print(f"克隆Rust仓库到 {self.repo_path}...")
            git.Repo.clone_from(self.repo_url, self.repo_path, depth=1)
            repo = git.Repo(self.repo_path)
            repo.git.fetch('--tags')
            
        else:
            print("更新Rust仓库...")
            repo = git.Repo(self.repo_path)
            repo.git.fetch('--tags')
        
        # 初始化API检测器
        self.detector = APIChangeDetector(self.repo_path)
    
    def get_version_tags(self) -> List[str]:
        """获取所有版本标签"""
        repo = git.Repo(self.repo_path)
        version_tags = []
        
        # Debug - print all raw tags
        # print(f"All tags: {[str(tag) for tag in repo.tags]}")
        
        for tag in repo.tags:
            tag_name = str(tag.name)
            # print(f"Processing tag: {tag_name}")
            # 只保留稳定版本标签（格式如1.50.0）
            if re.match(r'^1\.\d+\.\d+$', tag_name):
                # print(f"  Matched version tag: {tag_name}")
                version_tags.append(tag_name)
        
        # print(f"Found version tags: {version_tags}")
        # 按照版本号排序
        version_tags.sort(key=lambda v: semver.VersionInfo.parse(v))
        return version_tags
    
    
    def get_version_pairs(self, start_version=None, end_version=None, interval=1) -> List[Tuple[str, str]]:
        """获取需要比较的版本对"""
        tags = self.get_version_tags()
        # print(f"All version tags (sorted): {tags}")
        
        # 过滤版本范围
        if start_version:
            # Convert to VersionInfo objects for proper comparison
            start_ver = semver.VersionInfo.parse(start_version)
            start_idx = next((i for i, v in enumerate(tags) if semver.VersionInfo.parse(v) >= start_ver), 0)
            print(f"Start index for {start_version}: {start_idx}")
            tags = tags[start_idx:]
        
        if end_version:
            # Convert to VersionInfo objects for proper comparison
            end_ver = semver.VersionInfo.parse(end_version)
            end_idx = next((i+1 for i, v in enumerate(tags) if semver.VersionInfo.parse(v) > end_ver), len(tags))
            print(f"End index for {end_version}: {end_idx}")
            tags = tags[:end_idx]
        
        # print(f"Filtered tags: {tags}")
        
        # 生成版本对
        pairs = []
        for i in range(0, len(tags) - 1, interval):
            if i+interval < len(tags):
                pairs.append((tags[i], tags[i + interval]))
        
        # print(f"Generated pairs: {pairs}")
        return pairs
    
    
    def collect_api_changes(self, version_pairs) -> List[APIChange]:
        """收集指定版本对之间的API变更"""
        all_changes = []
        
        for from_ver, to_ver in version_pairs:
            print(f"分析版本 {from_ver} -> {to_ver} 的变更...")
            changes = self.detector.detect_changes(from_ver, to_ver)
            for change in changes:
                change.from_version = from_ver
                change.to_version = to_ver
            
            all_changes.extend(changes)
        
        return all_changes


# ==== 报告生成器 ====
class ReportGenerator:
    def __init__(self, output_dir="./reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def generate_json_report(self, changes: List[APIChange], filename=None):
        """生成JSON格式的变更报告"""
        if not filename:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rust_api_changes_{timestamp}.json"
        
        output_path = self.output_dir / filename
        
        # 将APIChange对象转换为可JSON序列化的字典
        report_data = []
        for change in changes:
            # 将APIEntity转换为字典
            api_dict = {
                "name": change.api.name,
                "from_version": change.from_version,
                "to_version": change.to_version,
                "module": change.api.module,
                "type": change.api.type,
                "signature": change.api.signature,
                "change_type": change.change_type,
                # "attributes": change.api.attributes,
                "documentation": change.api.documentation,
                "examples": change.api.examples,
                "source_code": change.api.source_code,
                "old_source_code": getattr(change, 'old_source_code', '') 
            }
            
            # # 将APIChange转换为字典
            # change_dict = {
            #     "api": api_dict,
            #     "change_type": change.change_type,
            #     "from_version": change.from_version,
            #     "to_version": change.to_version,
            #     "details": change.details
            # }
            
            report_data.append(api_dict)
        
        # 写入JSON文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"报告已生成: {output_path}")
        return output_path
    
    def generate_markdown_report(self, changes: List[APIChange], filename=None):
        """生成Markdown格式的变更报告"""
        if not filename:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rust_api_changes_{timestamp}.md"
        
        output_path = self.output_dir / filename
        
        # 按变更类型分组
        changes_by_type = {}
        for change in changes:
            if change.change_type not in changes_by_type:
                changes_by_type[change.change_type] = []
            changes_by_type[change.change_type].append(change)
        
        # 生成Markdown内容
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Rust API 变更报告\n\n")
            
            # 变更统计
            f.write("## 变更统计\n\n")
            for change_type, type_changes in changes_by_type.items():
                type_name = {
                    "stabilized": "stabilized",
                    "deprecated": "deprecated",
                    "signature": "signature_changed",
                    "implicit": "implicit_changed",
                }.get(change_type, change_type)
                
                f.write(f"- {type_name}: {len(type_changes)}个API\n")
            
            f.write("\n")
            
            # 按类型生成详细报告
            for change_type, type_changes in changes_by_type.items():
                type_name = {
                    "stabilized": "stabilized API",
                    "deprecated": "deprecated API",
                    "signature": "signature changed API",
                    "implicit": "implicit changed API",
                }.get(change_type, change_type)
                
                f.write(f"## {type_name}\n\n")
                
                for change in type_changes:
                    api = change.api
                    module_path = f"{api.module}::" if api.module else ""
                    f.write(f"### {module_path}{api.name}\n\n")
                    
                    f.write(f"**version**: {change.from_version} → {change.to_version}\n\n")
                    f.write(f"**type**: {api.type}\n\n")
                    f.write(f"**signature**: `{api.signature}`\n\n")
                    f.write(f"**change_details**: {change.details}\n\n")
                    
                    if api.documentation:
                        f.write("**doc**:\n\n")
                        f.write(f"```\n{api.documentation}\n```\n\n")
                    
                    if api.examples:
                        f.write("**example**:\n\n")
                        f.write("```rust\n")
                        for example in api.examples:
                            f.write(f"{example}\n")
                        f.write("```\n\n")
                    
                    f.write("---\n\n")
        
        print(f"Markdown报告已生成: {output_path}")
        return output_path


# ==== 主执行流程 ====
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Rust官方仓库API变更检测工具")
    parser.add_argument("--repo", default="./rust-repo", help="Rust仓库本地路径")
    parser.add_argument("--output", default="./reports", help="输出报告路径")
    parser.add_argument("--start", default=None, help="起始版本，如1.50.0")
    parser.add_argument("--end", default=None, help="结束版本，如1.55.0")
    parser.add_argument("--interval", type=int, default=1, help="版本间隔，默认为1")
    parser.add_argument("--format", choices=["json", "markdown", "both"], default="both", help="报告格式")
    
    args = parser.parse_args()
    
    # 初始化数据采集器
    collector = DataCollector(repo_path=args.repo)
    collector.setup_repo()
    
    # 获取要分析的版本对
    version_pairs = collector.get_version_pairs(args.start, args.end, args.interval)
    # print(f"将分析以下版本对: {version_pairs}")
    
    # 收集API变更
    changes = collector.collect_api_changes(version_pairs)
    print(f"共检测到 {len(changes)} 个API变更")
    
    # 生成报告
    reporter = ReportGenerator(args.output)
    
    if args.format in ["json", "both"]:
        json_path = reporter.generate_json_report(changes)
    
    if args.format in ["markdown", "both"]:
        md_path = reporter.generate_markdown_report(changes)


if __name__ == "__main__":
    main()