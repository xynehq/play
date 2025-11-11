"""
python scripts/search_usage.py --input ./reports/input.json --output ./reports/output.json
"""

import json
import os
import requests
import time
import asyncio
import aiohttp
from urllib.parse import quote
from collections import defaultdict
from tqdm import tqdm
import random
import traceback
from typing import List, Dict, Any

# GitHub API configuration
GITHUB_API_URL = "https://api.github.com"
SEARCH_ENDPOINT = "/search/code"

# Rate limit configuration
REQUEST_TIMEOUT = 60  # Request timeout (seconds)
RETRY_BACKOFF = 10  # Base retry interval (seconds)
MAX_RETRIES = 7  # Maximum retry attempts

# Multi-token configuration
GITHUB_TOKENS = os.getenv('GITHUB_TOKENS', '').split(',')
if not GITHUB_TOKENS or GITHUB_TOKENS[0] == '':
    single_token = os.getenv('GITHUB_TOKEN')
    if single_token:
        GITHUB_TOKENS = [single_token]
    else:
        GITHUB_TOKENS = []

class RustApiUsageFinder:
    """Helper class for analyzing Rust code to find actual API usages"""
    
    @staticmethod
    def extract_crate_name(module_path):
        """Extract the crate name from a module path"""
        if not module_path:
            return None
        # Most module paths start with the crate name
        return module_path.split('::')[0]
    
    @staticmethod
    def parse_cargo_toml(content):
        """Parse Cargo.toml content to extract dependencies"""
        dependencies = {}
        in_dependencies_section = False
        
        for line in content:
            line = line.strip()
            
            # Check for dependencies section headers (handles multiple dependency section types)
            if line == '[dependencies]' or line == '[dev-dependencies]' or line == '[build-dependencies]':
                in_dependencies_section = True
                continue
            elif line.startswith('[') and line.endswith(']'):
                in_dependencies_section = False
                continue
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Extract dependency information if we're in a dependencies section
            if in_dependencies_section and '=' in line:
                parts = line.split('=', 1)
                crate_name = parts[0].strip().strip('"\'')
                version_info = parts[1].strip().strip(',"\'')
                
                dependencies[crate_name] = version_info
        
        return dependencies
    
    @staticmethod
    def analyze_imports(lines, api_entry):
        """Analyze import statements to verify API usage"""
        module_path = api_entry.get('module', '')
        api_name = api_entry['name']
        
        # Track all imports
        imports = []
        
        # Look for import patterns
        for line in lines:
            line = line.strip()
            
            if line.startswith("use ") and ";" in line:
                import_path = line[4:line.find(";")].strip()
                imports.append(import_path)
                
                # Direct import: use std::fs::File;
                if import_path == f"{module_path}::{api_name}":
                    return 1.0  # 100% confidence
                
                # Import with braces: use std::fs::{File, write};
                if module_path in import_path and "{" in import_path and "}" in import_path:
                    brace_content = import_path[import_path.find("{")+1:import_path.find("}")]
                    items = [item.strip() for item in brace_content.split(",")]
                    if api_name in items:
                        return 1.0  # 100% confidence
                    
                # Import with alias: use std::fs::File as StdFile;
                if f"{module_path}::{api_name} as " in import_path:
                    return 1.0  # 100% confidence
                
                # Import the entire module: use std::fs;
                if import_path == module_path:
                    return 0.7  # 70% confidence - need to check usage with qualified path
        
        # No exact import found, check partial matches
        if module_path:
            crate_name = RustApiUsageFinder.extract_crate_name(module_path)
            if crate_name:
                for imp in imports:
                    if imp.startswith(crate_name):
                        return 0.3  # 30% confidence - at least the crate is imported
        
        return 0.0  # No relevant imports found
    
    @staticmethod
    def analyze_usage(lines, api_entry):
        """Analyze API usage in the code"""
        api_name = api_entry['name']
        module_path = api_entry.get('module', '')
        confidence = 0.0
        
        # First check for imports
        import_confidence = RustApiUsageFinder.analyze_imports(lines, api_entry)
        if import_confidence == 1.0:
            # If we have a direct import, look for usage
            for line in lines:
                line = line.strip()
                # Look for function calls: api_name(...)
                if f"{api_name}(" in line and not line.startswith("//"):
                    return 0.9  # 90% confidence - direct call after import
                # Look for struct/type usage: Type { ... } or Type::method
                if (f"{api_name} {{" in line or f"{api_name}::" in line) and not line.startswith("//"):
                    return 0.9  # 90% confidence - direct usage after import
            
            # Import found but no clear usage
            return 0.5
        
        # If module was imported but not the specific item, look for qualified usage
        if import_confidence == 0.7:
            module_parts = module_path.split("::")
            for line in lines:
                line = line.strip()
                if not line.startswith("//"):
                    # Check for qualified usage: module::api_name
                    if f"{module_parts[-1]}::{api_name}" in line:
                        return 0.8  # 80% confidence - qualified usage
        
        return import_confidence  # Return based just on import analysis

class TokenWorker:
    """Independent worker coroutine for each token"""
    def __init__(self, token_id, token, queue, result_collector, progress_bar):
        self.token_id = token_id
        self.token = token
        self.queue = queue
        self.result_collector = result_collector
        self.progress_bar = progress_bar
        
        # Token status
        self.reset_time = time.time()
        self.remaining = 30  # Default search API quota
        self.error_count = 0
        self.paused = False
        
        # Create random User-Agent
        self.user_agent = f"GitHubApiClient-{token_id}-{random.randint(1000,9999)}"
        
        # Session object
        self.session = None
    
    def get_headers(self):
        """Get authentication headers"""
        return {
            "Accept": "application/vnd.github.v3+json",
            "Authorization": f"token {self.token}",
            "User-Agent": self.user_agent
        }
    
    async def initialize(self):
        """Initialize token status"""
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(timeout=timeout)
        
        try:
            print(f"Initializing token {self.token_id+1}/{len(GITHUB_TOKENS)}...")
            
            async with self.session.get(
                f"{GITHUB_API_URL}/rate_limit",
                headers=self.get_headers()
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Get search API limit info
                    search_limits = data.get('resources', {}).get('search', {})
                    self.remaining = search_limits.get('remaining', 30)
                    self.reset_time = search_limits.get('reset', int(time.time() + 60))
                    
                    if self.remaining <= 0:
                        self.paused = True
                        wait_time = max(0, self.reset_time - time.time())
                        print(f"Token {self.token_id+1}: Rate limited, will resume in {wait_time:.0f} seconds")
                    
                    return True
                else:
                    print(f"Token {self.token_id+1}: Initialization failed, status code {response.status}")
                    return False
        except Exception as e:
            print(f"Token {self.token_id+1}: Initialization error - {str(e)}")
            return False
    
    async def run(self):
        """Run worker coroutine"""
        if not await self.initialize():
            print(f"Token {self.token_id+1}: Initialization failed, not participating")
            return
        
        try:
            while True:
                # Check if we need to pause (rate limited)
                if self.paused:
                    current_time = time.time()
                    if current_time < self.reset_time:
                        wait_time = self.reset_time - current_time
                        print(f"Token {self.token_id+1}: Waiting for rate limit reset, {wait_time:.0f} seconds remaining")
                        await asyncio.sleep(min(wait_time + 2, 60))  # Wait at most 60 seconds
                        continue
                    else:
                        # Reset status
                        self.paused = False
                        self.remaining = 30  # Default search API quota
                        self.error_count = 0
                        print(f"Token {self.token_id+1}: Rate limit reset, continuing work")
                
                # Get task, non-blocking
                try:
                    task = self.queue.get_nowait()
                except asyncio.QueueEmpty:
                    # Queue is empty, all tasks completed
                    print(f"Token {self.token_id+1}: Queue empty, work complete")
                    break
                
                # Process task
                group_key, entries = task
                try:
                    result = await self.process_api_group(group_key, entries)
                    if result:
                        await self.result_collector.put(result)
                    
                    # Update progress bar
                    self.progress_bar.update(1)
                    
                    # Mark task as done
                    self.queue.task_done()
                except Exception as e:
                    print(f"Token {self.token_id+1}: Task exception - {str(e)}")
                    self.error_count += 1
                    
                    # Put failed task back in queue, but retry at most 3 times
                    if getattr(task, '_retry_count', 0) < 3:
                        task._retry_count = getattr(task, '_retry_count', 0) + 1
                        await self.queue.put(task)
                        print(f"Token {self.token_id+1}: Task {group_key} requeued, retry: {task._retry_count}")
                    else:
                        print(f"Token {self.token_id+1}: Task {group_key} failed too many times, abandoning")
                        # Update progress bar
                        self.progress_bar.update(1)
                        self.queue.task_done()
                
                # Pause if too many errors
                if self.error_count > 10:
                    print(f"Token {self.token_id+1}: Too many errors, pausing for 10 minutes")
                    self.paused = True
                    self.reset_time = time.time() + 600  # Pause for 10 minutes
                
                # Random delay to avoid too many requests
                await asyncio.sleep(random.uniform(1.0, 3.0))
        except Exception as e:
            print(f"Token {self.token_id+1}: Worker coroutine exception - {str(e)}")
            traceback.print_exc()
        finally:
            # Close session
            if self.session and not self.session.closed:
                await self.session.close()
    
    async def search_github_code(self, query, retries=0):
        """Execute GitHub code search"""
        # Check token status
        if self.remaining <= 0:
            self.paused = True
            print(f"Token {self.token_id+1}: Rate limited, waiting for reset")
            return []
        
        try:
            # Add random delay
            await asyncio.sleep(random.uniform(0.5, 1.5))
            
            # Prepare request parameters
            params = {'q': query, 'per_page': 5}  # Get up to 5 results for better chance of finding actual usage
            
            # Send request
            print(f"Token {self.token_id+1}: Searching {query[:50]}...")
            timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
            
            async with self.session.get(
                f"{GITHUB_API_URL}{SEARCH_ENDPOINT}", 
                headers=self.get_headers(),
                params=params,
                timeout=timeout
            ) as response:
                # Process response
                response_text = await response.text()
                
                # Update token status
                self.remaining = int(response.headers.get('X-RateLimit-Remaining', '0'))
                self.reset_time = int(response.headers.get('X-RateLimit-Reset', time.time() + 60))
                
                print(f"Token {self.token_id+1}: {self.remaining} requests remaining, "
                      f"reset time: {time.strftime('%H:%M:%S', time.localtime(self.reset_time))}")
                
                if response.status == 403 or "rate limit exceeded" in response_text.lower():
                    # Rate limited
                    self.paused = True
                    self.remaining = 0
                    print(f"Token {self.token_id+1}: Rate limited during search")
                    
                    if retries < MAX_RETRIES:
                        # Exponential backoff retry
                        retry_wait = RETRY_BACKOFF * (2 ** retries) + random.uniform(1, 5)
                        print(f"Token {self.token_id+1}: Waiting {retry_wait:.1f} seconds before retry...")
                        await asyncio.sleep(retry_wait)
                        return await self.search_github_code(query, retries + 1)
                    else:
                        print(f"Token {self.token_id+1}: Maximum retries reached, abandoning search")
                        return []
                
                if response.status == 200:
                    try:
                        data = json.loads(response_text)
                        items = data.get('items', [])
                        print(f"Token {self.token_id+1}: Found {len(items)} results")
                        return items
                    except json.JSONDecodeError:
                        print(f"Token {self.token_id+1}: JSON parsing failed")
                        self.error_count += 1
                        
                        if retries < MAX_RETRIES:
                            await asyncio.sleep(RETRY_BACKOFF * (retries + 1))
                            return await self.search_github_code(query, retries + 1)
                        return []
                
                # Other errors
                print(f"Token {self.token_id+1}: Search failed, status code {response.status}")
                self.error_count += 1
                
                if retries < MAX_RETRIES:
                    retry_wait = RETRY_BACKOFF * (retries + 1) + random.uniform(1, 3)
                    await asyncio.sleep(retry_wait)
                    return await self.search_github_code(query, retries + 1)
                return []
        
        except asyncio.TimeoutError:
            print(f"Token {self.token_id+1}: Search timeout")
            self.error_count += 1
            
            if retries < MAX_RETRIES:
                retry_wait = RETRY_BACKOFF * (retries + 1)
                await asyncio.sleep(retry_wait)
                return await self.search_github_code(query, retries + 1)
            return []
            
        except Exception as e:
            print(f"Token {self.token_id+1}: Search exception - {str(e)}")
            self.error_count += 1
            
            if retries < MAX_RETRIES:
                retry_wait = RETRY_BACKOFF * (retries + 1)
                await asyncio.sleep(retry_wait)
                return await self.search_github_code(query, retries + 1)
            return []
    
    async def get_file_content(self, url, retries=0):
        """Get file content"""
        try:
            # Add random delay
            await asyncio.sleep(random.uniform(0.5, 1.5))
            
            # Send request
            print(f"Token {self.token_id+1}: Getting file {url.split('/')[-1]}...")
            timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
            
            async with self.session.get(
                url,
                headers=self.get_headers(),
                timeout=timeout
            ) as response:
                # Process response
                if response.status == 403:
                    # Possibly rate limited
                    response_text = await response.text()
                    
                    if "rate limit exceeded" in response_text.lower():
                        self.paused = True
                        self.remaining = 0
                        self.reset_time = int(response.headers.get('X-RateLimit-Reset', time.time() + 3600))
                        print(f"Token {self.token_id+1}: Rate limited while getting file")
                    
                    if retries < MAX_RETRIES:
                        retry_wait = RETRY_BACKOFF * (2 ** retries) + random.uniform(1, 5)
                        print(f"Token {self.token_id+1}: Waiting {retry_wait:.1f} seconds before retry...")
                        await asyncio.sleep(retry_wait)
                        return await self.get_file_content(url, retries + 1)
                    else:
                        print(f"Token {self.token_id+1}: Maximum retries reached, abandoning file retrieval")
                        return []
                
                if response.status == 404:
                    # File doesn't exist
                    print(f"Token {self.token_id+1}: File not found")
                    return []
                
                if response.status == 200:
                    # Get file content
                    try:
                        text = await response.text()
                        print(f"Token {self.token_id+1}: File retrieved successfully, size {len(text)} bytes")
                        return text.splitlines()
                    except UnicodeDecodeError:
                        print(f"Token {self.token_id+1}: File encoding issue")
                        self.error_count += 1
                        
                        if retries < MAX_RETRIES:
                            await asyncio.sleep(RETRY_BACKOFF * (retries + 1))
                            return await self.get_file_content(url, retries + 1)
                        return []
                
                # Other errors
                print(f"Token {self.token_id+1}: File retrieval failed, status code {response.status}")
                self.error_count += 1
                
                if retries < MAX_RETRIES:
                    retry_wait = RETRY_BACKOFF * (retries + 1) + random.uniform(1, 3)
                    await asyncio.sleep(retry_wait)
                    return await self.get_file_content(url, retries + 1)
                return []
        
        except asyncio.TimeoutError:
            print(f"Token {self.token_id+1}: File retrieval timeout")
            self.error_count += 1
            
            if retries < MAX_RETRIES:
                retry_wait = RETRY_BACKOFF * (retries + 1)
                await asyncio.sleep(retry_wait)
                return await self.get_file_content(url, retries + 1)
            return []
            
        except Exception as e:
            print(f"Token {self.token_id+1}: File retrieval exception - {str(e)}")
            self.error_count += 1
            
            if retries < MAX_RETRIES:
                retry_wait = RETRY_BACKOFF * (retries + 1)
                await asyncio.sleep(retry_wait)
                return await self.get_file_content(url, retries + 1)
            return []
    
    async def get_cargo_toml(self, repo_full_name, retries=0):
        """Get Cargo.toml file from repository root"""
        cargo_url = f"https://raw.githubusercontent.com/{repo_full_name}/master/Cargo.toml"
        content = await self.get_file_content(cargo_url, retries)
        
        # Try alternative locations if not found
        if not content:
            cargo_url = f"https://raw.githubusercontent.com/{repo_full_name}/main/Cargo.toml"
            content = await self.get_file_content(cargo_url, retries)
        
        return content
    
    def extract_code_context(self, lines, api_name, context_lines=15):
        """Extract code context"""
        if not lines:
            return None
            
        contexts = []
        for idx, line in enumerate(lines):
            if api_name in line and not line.strip().startswith("//"):
                # Use more context lines
                start = max(0, idx - context_lines)
                end = min(len(lines), idx + context_lines + 1)
                context = lines[start:end]
                
                # Mark target line
                marked = []
                for i, l in enumerate(context, start=start):
                    prefix = '>>> ' if i == idx else '    '
                    marked.append(f"{prefix}{l}")
                
                contexts.append('\n'.join(marked))
                
                # Only get up to 2 contexts per file
                if len(contexts) >= 2:
                    break
        
        if contexts:
            return '\n\n...\n\n'.join(contexts)
        return None
    
    def build_search_query(self, api_entry):
        """Build GitHub code search query"""
        module_path = api_entry.get('module', '')
        api_name = api_entry['name']
        
        # Build more complex query
        query_parts = [
            f'"{api_name}"',  # Exact match for API name
            'language:Rust',
            'NOT filename:test',
            'NOT path:tests',
            'NOT path:examples',
            'NOT path:benches',
            'NOT language:Markdown',
            'size:<1000'
        ]
        
        # If we have a module path, add more specific constraints
        if module_path:
            crate_name = RustApiUsageFinder.extract_crate_name(module_path)
            
            # Add possible import patterns
            if crate_name:
                # Try to match the import statement for this specific API
                query_parts.append(f'"{crate_name}"')
                
                # Check if this is in the standard library
                if crate_name == 'std':
                    # For std, focus on the module path
                    module_parts = module_path.split('::')
                    if len(module_parts) > 1:
                        # Add the second level module (e.g., "fs" from "std::fs")
                        query_parts.append(f'"{module_parts[1]}"')
                else:
                    # For third-party crates, specifically mention Cargo.toml
                    query_parts.append('filename:Cargo.toml')
        
        return ' '.join(query_parts)
    
    async def verify_dependencies(self, repo_name, api_entry):
        """Verify repository dependencies match the API's crate"""
        module_path = api_entry.get('module', '')
        if not module_path:
            return 0.5  # No module to verify
        
        crate_name = RustApiUsageFinder.extract_crate_name(module_path)
        if not crate_name or crate_name == 'std':
            return 1.0  # Standard library is always available
        
        # Get Cargo.toml content
        cargo_content = await self.get_cargo_toml(repo_name)
        if not cargo_content:
            return 0.2  # No Cargo.toml found
        
        # Parse dependencies
        dependencies = RustApiUsageFinder.parse_cargo_toml(cargo_content)
        
        # Check if the target crate is a dependency
        if crate_name in dependencies:
            return 1.0  # Direct dependency
        
        # Check for common renamed imports
        if crate_name.startswith('tokio'):
            if 'tokio' in dependencies:
                return 0.9  # Part of Tokio ecosystem
        
        # Some APIs might come from parent crates
        for dep in dependencies:
            if crate_name.startswith(dep) or dep.startswith(crate_name):
                return 0.8  # Possible related dependency
        
        return 0.1  # No matching dependency found
    
    async def process_api_entry(self, api_entry):
        """Process a single API entry"""
        query = self.build_search_query(api_entry)
        
        # Search for API usage
        search_results = await self.search_github_code(query)
        if not search_results:
            return []  # No usage examples found
        
        # Process search results
        usage_examples = []
        
        for result in search_results[:3]:  # Limit to 3 results
            file_url = result.get('html_url', '')
            repo_name = result.get('repository', {}).get('full_name', '')
            
            # Skip if no file URL or repository
            if not file_url or not repo_name:
                continue
            
            # Convert to raw content URL
            if 'github.com' in file_url and '/blob/' in file_url:
                raw_url = file_url.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/')
            else:
                print(f"Token {self.token_id+1}: Unsupported file URL format: {file_url}")
                continue
            
            # Get file content
            lines = await self.get_file_content(raw_url)
            if not lines:
                continue
            
            # Verify dependencies
            dep_confidence = await self.verify_dependencies(repo_name, api_entry)
            if dep_confidence < 0.5:
                print(f"Token {self.token_id+1}: Repository {repo_name} likely doesn't use the target crate")
                continue
            
            # Analyze API usage using AST-like analysis
            usage_confidence = RustApiUsageFinder.analyze_usage(lines, api_entry)
            if usage_confidence < 0.3:
                print(f"Token {self.token_id+1}: Low confidence in API usage in {file_url}")
                continue
            
            # Extract code context
            context = self.extract_code_context(lines, api_entry['name'])
            if not context:
                continue
            
            # Calculate overall confidence
            overall_confidence = (dep_confidence + usage_confidence) / 2
            
            # Add to usage examples
            usage_examples.append({
                'file_url': result['html_url'],
                'code_snippet': context,
                'repository': repo_name,
                'confidence': overall_confidence
            })
            
            # If we have enough good examples, stop
            if len(usage_examples) >= 2:
                break
        
        # Sort by confidence score
        usage_examples.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        return usage_examples
    
    async def process_api_group(self, group_key, entries):
        """Process an API entry (packaged as a single-entry group)"""
        # Each entry is now processed individually
        entry = entries[0]  # There should be only one entry per task now
        
        # Get usage code examples for this specific entry
        print(f"Token {self.token_id+1}: Processing API {entry.get('module', '')}::{entry['name']}")
        usage = await self.process_api_entry(entry)
        
        # Assign usage to this entry
        entry['usage_code'] = usage
        
        return entries

async def process_apis_with_parallel_tokens(input_file, output_file, resume_from=0):
    """Process all APIs with parallel tokens"""
    # Check if there are available tokens
    if not GITHUB_TOKENS:
        raise ValueError("No GitHub tokens provided, please set GITHUB_TOKENS environment variable")
    
    print(f"Processing in parallel with {len(GITHUB_TOKENS)} GitHub tokens")
    
    # Read API data
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            api_data = json.load(f)
        print(f"Loaded {len(api_data)} API entries")
    except Exception as e:
        print(f"Error reading API data file: {e}")
        return
    
    # 为每个API创建独立任务，不进行分组
    task_items = []
    for entry in api_data:
        # 创建一个唯一标识符
        key = f"{entry.get('module', '')}::{entry['name']}:{id(entry)}"
        # 每个条目作为单独的"组"，只包含一个元素
        task_items.append((key, [entry]))
    
    total_tasks = len(task_items)
    print(f"Prepared {total_tasks} individual API tasks")
    
    # Support resuming from specific position
    if resume_from > 0:
        print(f"Resuming from API task {resume_from}...")
        if resume_from >= total_tasks:
            print(f"Specified resume position ({resume_from}) exceeds total API count ({total_tasks}), starting from beginning")
            resume_from = 0
        task_items = task_items[resume_from:]
    
    # Create progress bar
    progress_bar = tqdm(total=len(task_items), desc="Processing APIs")
    
    # Create task queue
    task_queue = asyncio.Queue()
    result_queue = asyncio.Queue()
    
    # Add tasks to queue
    for item in task_items:
        await task_queue.put(item)
    
    # Create token worker coroutines
    workers = []
    for i, token in enumerate(GITHUB_TOKENS):
        worker = TokenWorker(
            token_id=i,
            token=token,
            queue=task_queue,
            result_collector=result_queue,
            progress_bar=progress_bar
        )
        workers.append(worker.run())
    
    # Create result collection coroutine
    async def collect_results():
        processed_data = []
        temp_file_counter = 0
        
        try:
            while True:
                try:
                    # Get result with 10 second timeout
                    result = await asyncio.wait_for(result_queue.get(), timeout=10)
                    processed_data.extend(result)
                    result_queue.task_done()
                    
                    # Save temporary file every 100 results
                    if len(processed_data) >= 100:
                        temp_output = f"{output_file}.temp.{temp_file_counter}"
                        try:
                            with open(temp_output, 'w', encoding='utf-8') as f:
                                json.dump(processed_data, f, indent=2, ensure_ascii=False)
                            print(f"Saved {len(processed_data)} API results to temporary file: {temp_output}")
                            temp_file_counter += 1
                        except Exception as e:
                            print(f"Failed to save temporary file: {e}")
                        
                        # Continue collecting next batch
                        processed_data = []
                except asyncio.TimeoutError:
                    # Check queue size and task completion
                    if task_queue.empty() and all(worker.done() for worker in asyncio.all_tasks() if worker != asyncio.current_task()):
                        print("All worker coroutines completed, ending collection")
                        break
        finally:
            # Save final results
            if processed_data:
                final_temp_output = f"{output_file}.temp.final"
                try:
                    with open(final_temp_output, 'w', encoding='utf-8') as f:
                        json.dump(processed_data, f, indent=2, ensure_ascii=False)
                    print(f"Saved final {len(processed_data)} API results to temporary file: {final_temp_output}")
                except Exception as e:
                    print(f"Failed to save final temporary file: {e}")
        
        return temp_file_counter
    
    # Start all coroutines
    all_tasks = workers + [collect_results()]
    results = await asyncio.gather(*all_tasks, return_exceptions=True)
    
    # Close progress bar
    progress_bar.close()
    
    # Merge all temporary files
    temp_file_count = results[-1]  # Last result is collect_results return value
    all_processed_data = []
    
    for i in range(temp_file_count + 1):  # +1 to include final file
        temp_file = f"{output_file}.temp.{i}" if i < temp_file_count else f"{output_file}.temp.final"
        if os.path.exists(temp_file):
            try:
                with open(temp_file, 'r', encoding='utf-8') as f:
                    batch_data = json.load(f)
                    all_processed_data.extend(batch_data)
                    print(f"Loaded {len(batch_data)} APIs from temporary file: {temp_file}")
            except Exception as e:
                print(f"Failed to read temporary file: {e}")
    
    # Write final results
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_processed_data, f, indent=2, ensure_ascii=False)
        print(f"All API processing complete! Results saved to: {output_file}")
        print(f"Processed {len(all_processed_data)} APIs, {len(all_processed_data)/len(api_data)*100:.1f}% of total")
    except Exception as e:
        print(f"Failed to save final results: {e}")
        return
    
    # Clean up temporary files
    try:
        for i in range(temp_file_count + 1):
            temp_file = f"{output_file}.temp.{i}" if i < temp_file_count else f"{output_file}.temp.final"
            if os.path.exists(temp_file):
                os.remove(temp_file)
        print("Cleaned up all temporary files")
    except Exception as e:
        print(f"Error cleaning up temporary files: {e}")

def main():
    """Main function"""
    # Check if tokens are set
    if not os.getenv('GITHUB_TOKENS') and not os.getenv('GITHUB_TOKEN'):
        print("Error: GITHUB_TOKENS environment variable (comma-separated tokens) or GITHUB_TOKEN must be set")
        return
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Enhanced Rust API Usage Search Tool")
    parser.add_argument('--input', type=str, default='./reports/rust_api_changes_latest_version.json',
                       help='Input API data file path')
    parser.add_argument('--output', type=str, default='./reports/rust_api_changes_with_usage.json',
                       help='Output results file path')
    parser.add_argument('--resume-from', type=int, default=0,
                       help='Resume processing from specific API group index (default: 0)')
    
    args = parser.parse_args()
    
    # Run async main function
    asyncio.run(process_apis_with_parallel_tokens(
        args.input, 
        args.output, 
        args.resume_from
    ))

if __name__ == "__main__":
    main()