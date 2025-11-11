import requests
from bs4 import BeautifulSoup
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import os
from urllib.parse import urljoin


def fetch_rust_api_info():
    """
    Fetch comprehensive information about Rust standard library APIs.
    Collects stability information, descriptions, categorization, and more.
    """
    # Base URL for Rust documentation
    base_url = "https://doc.rust-lang.org"

    try:
        # Get the main page which lists all the stable APIs
        response = requests.get(f"{base_url}/stable/std/", timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the Rust version
        version_element = soup.select_one('meta[name="generator"]')
        rust_version = version_element['content'] if version_element else "Unknown"
        version_match = re.search(r'\d+\.\d+\.\d+', rust_version)
        rust_version = version_match.group(0) if version_match else rust_version

        # Create output directory for detailed API information
        os.makedirs("rust_api_details", exist_ok=True)

        # Find all modules in the standard library
        modules = []
        module_links = {}  # Store module name -> URL mapping

        for link in soup.select('.item-name a'):
            module_name = link.text.strip()
            if module_name and not module_name.startswith('_'):
                modules.append(module_name)
                # Store the URL for each module
                href = link.get('href')
                if href:
                    module_links[module_name] = urljoin(f"{base_url}/stable/std/", href)

        print(f"Found {len(modules)} modules to process")

        # For storing all API information
        api_info = {}

        def get_module_info(module_name):
            try:
                # Use stored URL if available, otherwise construct it
                module_url = module_links.get(module_name, f"{base_url}/stable/std/{module_name}/")
                module_response = requests.get(module_url, timeout=15)
                module_response.raise_for_status()

                module_soup = BeautifulSoup(module_response.text, 'html.parser')

                # Initialize the module information dictionary
                module_data = {
                    "name": module_name,
                    "url": module_url,
                    "stability": extract_version_info(module_soup),
                    "description": extract_description(module_soup),
                    "type": detect_api_type(module_soup),
                    "items": extract_items(module_soup, module_url),
                    "deprecated": is_deprecated(module_soup),
                    "deprecation_note": extract_deprecation_info(module_soup),
                    "examples": extract_examples(module_soup),
                    "related_modules": extract_related_modules(module_soup),
                }

                # Save detailed HTML structure to file for reference
                save_html_structure(module_name, module_soup)

                return (module_name, module_data)

            except requests.RequestException as e:
                print(f"Error fetching {module_name}: {str(e)}")
                return (module_name, {"error": f"Request failed: {str(e)}"})
            except Exception as e:
                print(f"Error processing {module_name}: {str(e)}")
                return (module_name, {"error": f"Processing error: {str(e)}"})

        def extract_version_info(soup):
            """Extract version information using multiple methods."""
            # Method 1: Look for .stab or .stability class divs
            for class_name in ['.stab', '.stability', '.since', '.portability', '.feature']:
                elements = soup.select(class_name)
                for element in elements:
                    text = element.get_text()
                    # Try different regex patterns
                    for pattern in [
                        r'Stable since (\d+\.\d+\.\d+)',
                        r'since (\d+\.\d+\.\d+)',
                        r'(\d+\.\d+\.\d+)',
                    ]:
                        match = re.search(pattern, text)
                        if match:
                            return match.group(1)

            # Method 2: Look for version information in the page footer
            footer = soup.select_one('footer')
            if footer:
                footer_text = footer.get_text()
                match = re.search(r'(\d+\.\d+\.\d+)', footer_text)
                if match:
                    return match.group(1)

            # Method 3: Look for version in #main-content section
            main_content = soup.select_one('#main-content')
            if main_content:
                # Look for a span with version information
                spans = main_content.select('span')
                for span in spans:
                    span_text = span.get_text()
                    if 'version' in span_text.lower() or 'since' in span_text.lower():
                        match = re.search(r'(\d+\.\d+\.\d+)', span_text)
                        if match:
                            return match.group(1)

            # Method 4: Check the attributes section
            attribs = soup.select('.docblock')
            for attrib in attribs:
                attrib_text = attrib.get_text()
                if 'since' in attrib_text.lower():
                    match = re.search(r'since\s+(\d+\.\d+\.\d+)', attrib_text, re.IGNORECASE)
                    if match:
                        return match.group(1)

            # Method 5: Check for any element with data-stability attribute
            elements_with_stability = soup.select('[data-stability]')
            for element in elements_with_stability:
                stability = element.get('data-stability')
                if stability and re.search(r'\d+\.\d+\.\d+', stability):
                    match = re.search(r'(\d+\.\d+\.\d+)', stability)
                    if match:
                        return match.group(1)

            # If no version found, check for any number that looks like a version
            all_text = soup.get_text()
            version_matches = re.findall(r'(?:since|version)\s+(\d+\.\d+\.\d+)', all_text, re.IGNORECASE)
            if version_matches:
                return version_matches[0]

            return "Unknown"

        def extract_description(soup):
            """Extract module description."""
            # Try to find the main description block
            for selector in ['.docblock', '.desc', '.summary', '#main-content p']:
                elements = soup.select(selector)
                if elements:
                    # Get the first paragraph that looks like a description
                    for element in elements:
                        text = element.get_text().strip()
                        if text and len(text) > 20:  # Avoid very short descriptions
                            return text[:500]  # Limit description length

            return "No description available"

        def detect_api_type(soup):
            """Detect if the API is a struct, trait, enum, etc."""
            # Look for indicators in the title or headers
            title = soup.title.string if soup.title else ""
            headers = [h.get_text() for h in soup.find_all(['h1', 'h2', 'h3'])]
            all_text = title + " " + " ".join(headers)

            if "struct" in all_text.lower():
                return "struct"
            elif "trait" in all_text.lower():
                return "trait"
            elif "enum" in all_text.lower():
                return "enum"
            elif "fn" in all_text.lower() or "function" in all_text.lower():
                return "function"
            elif "mod" in all_text.lower() or "module" in all_text.lower():
                return "module"
            elif "macro" in all_text.lower():
                return "macro"
            else:
                return "unknown"

        def extract_items(soup, module_url):
            """Extract items contained in this module (functions, structs, etc.)"""
            items = []
            # Look for item listings
            for item_selector in ['.item-table', '.item-list', '.struct-list', '.trait-list', '.enum-list']:
                item_elements = soup.select(f"{item_selector} .item")
                for item in item_elements:
                    item_name_element = item.select_one('.item-name')
                    if item_name_element:
                        item_name = item_name_element.get_text().strip()
                        item_link = item_name_element.select_one('a')
                        item_href = item_link.get('href') if item_link else None
                        item_url = urljoin(module_url, item_href) if item_href else None

                        # Determine item type
                        item_type = "unknown"
                        item_classes = item.get('class', [])
                        if isinstance(item_classes, list):
                            class_str = " ".join(item_classes)
                        else:
                            class_str = str(item_classes)

                        if "struct" in class_str:
                            item_type = "struct"
                        elif "trait" in class_str:
                            item_type = "trait"
                        elif "enum" in class_str:
                            item_type = "enum"
                        elif "fn" in class_str or "method" in class_str:
                            item_type = "function"
                        elif "mod" in class_str:
                            item_type = "module"
                        elif "macro" in class_str:
                            item_type = "macro"

                        # Get item description
                        item_desc_element = item.select_one('.desc')
                        item_desc = item_desc_element.get_text().strip() if item_desc_element else ""

                        items.append({
                            "name": item_name,
                            "type": item_type,
                            "url": item_url,
                            "description": item_desc[:200] if item_desc else ""  # Limit description length
                        })

            return items

        def is_deprecated(soup):
            """Check if the API is deprecated."""
            deprecated_indicators = [
                'deprecated',
                'will be removed',
                'use instead',
                'replaced by',
                'no longer supported'
            ]

            # Check for deprecated class or text
            deprecated_elements = soup.select('.deprecated')
            if deprecated_elements:
                return True

            # Check text for deprecation indicators
            text = soup.get_text().lower()
            for indicator in deprecated_indicators:
                if indicator in text:
                    return True

            return False

        def extract_deprecation_info(soup):
            """Extract information about deprecation if available."""
            if not is_deprecated(soup):
                return None

            # Look for deprecation notice
            deprecated_elements = soup.select('.deprecated')
            if deprecated_elements:
                return deprecated_elements[0].get_text().strip()

            # Look for text containing deprecation information
            text = soup.get_text()
            deprecation_patterns = [
                r'deprecated.*?[\.\n]',
                r'will be removed.*?[\.\n]',
                r'use .* instead.*?[\.\n]',
                r'replaced by.*?[\.\n]'
            ]

            for pattern in deprecation_patterns:
                match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if match:
                    return match.group(0).strip()

            return "Deprecated, no additional information available"

        def extract_examples(soup):
            """Extract code examples."""
            examples = []

            # Look for elements with code examples
            for code_block in soup.select('pre.rust'):
                code_text = code_block.get_text().strip()
                if code_text:
                    examples.append(code_text)

            return examples[:3]  # Limit to 3 examples

        def extract_related_modules(soup):
            """Extract related modules mentioned in the documentation."""
            related = []

            # Look for links to other modules in the standard library
            for link in soup.select('a'):
                href = link.get('href', '')
                if '/std/' in href and not href.startswith('http'):
                    module_name = link.get_text().strip()
                    if module_name and module_name not in related:
                        related.append(module_name)

            return related[:10]  # Limit to 10 related modules

        def save_html_structure(module_name, soup):
            """Save HTML structure to a file for reference and debugging."""
            try:
                # Create a simplified structural representation
                structure = {
                    "title": soup.title.string if soup.title else "No title",
                    "headers": [h.get_text().strip() for h in soup.find_all(['h1', 'h2', 'h3'])],
                    "classes": list(set(c for tag in soup.find_all(class_=True) for c in tag.get('class', []))),
                    "ids": [tag.get('id') for tag in soup.find_all(id=True)],
                }

                # Save to file
                with open(f"rust_api_details/{module_name}_structure.json", 'w') as f:
                    json.dump(structure, f, indent=2)
            except Exception as e:
                print(f"Error saving HTML structure for {module_name}: {str(e)}")

        # Process a subset of modules first to test
        sample_size = min(5, len(modules))
        print(f"Testing with {sample_size} sample modules first:")

        sample_results = {}
        for module in modules[:sample_size]:
            module_name, module_data = get_module_info(module)
            sample_results[module_name] = module_data
            print(f"Sample processed: {module_name}")

        # Save the sample results
        with open("rust_api_sample.json", 'w') as f:
            json.dump(sample_results, f, indent=2)

        print("Sample results saved to rust_api_sample.json")

        # Ask if user wants to continue with all modules
        user_continue = input("Continue processing all modules? (y/n): ")
        if user_continue.lower() != 'y':
            print("Exiting at user request")
            return {
                "current_rust_version": rust_version,
                "api_count": len(sample_results),
                "apis": sample_results
            }

        # Use multithreading for the remaining modules
        remaining_modules = modules[sample_size:]
        print(f"Processing remaining {len(remaining_modules)} modules...")

        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_module = {executor.submit(get_module_info, module): module for module in remaining_modules}

            for future in as_completed(future_to_module, timeout=600):  # 10-minute timeout
                module = future_to_module[future]
                try:
                    module_name, module_data = future.result()
                    api_info[module_name] = module_data
                    print(f"Processed {module_name}")
                except Exception as e:
                    print(f"Exception processing {module}: {str(e)}")
                    api_info[module] = {"error": str(e)}

        # Combine sample results with the rest
        api_info.update(sample_results)

        # Extract additional statistics about the APIs
        stats = calculate_api_statistics(api_info)

        return {
            "current_rust_version": rust_version,
            "api_count": len(api_info),
            "statistics": stats,
            "apis": api_info
        }

    except requests.RequestException as e:
        print(f"Error fetching main Rust documentation page: {str(e)}")
        return {"error": f"Request failed: {str(e)}"}
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return {"error": f"Unexpected error: {str(e)}"}


def calculate_api_statistics(api_info):
    """Calculate statistics about the APIs."""
    stats = {
        "total_apis": len(api_info),
        "by_type": {},
        "by_stability": {},
        "deprecated_count": 0,
        "with_examples": 0,
        "total_examples": 0,
    }

    for api, data in api_info.items():
        # Count by type
        api_type = data.get("type", "unknown")
        stats["by_type"][api_type] = stats["by_type"].get(api_type, 0) + 1

        # Count by stability version
        stability = data.get("stability", "Unknown")
        stats["by_stability"][stability] = stats["by_stability"].get(stability, 0) + 1

        # Count deprecated APIs
        if data.get("deprecated", False):
            stats["deprecated_count"] += 1

        # Count APIs with examples
        examples = data.get("examples", [])
        if examples:
            stats["with_examples"] += 1
            stats["total_examples"] += len(examples)

    return stats


def save_to_json(data, filename="rust_api_info.json"):
    """Save the data to a JSON file."""
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Data saved to {filename}")
    except Exception as e:
        print(f"Error saving data to {filename}: {str(e)}")


def print_summary(data):
    """Print a summary of the collected API information."""
    if "error" in data:
        print(f"Error: {data['error']}")
        return

    print(f"Current Rust Version: {data['current_rust_version']}")
    print(f"Total APIs analyzed: {data['api_count']}")

    if "statistics" in data:
        stats = data["statistics"]
        print("\nAPI Statistics:")
        print(f"- APIs by type:")
        for api_type, count in stats.get("by_type", {}).items():
            print(f"  - {api_type}: {count}")

        print(f"- Deprecated APIs: {stats.get('deprecated_count', 0)}")
        print(
            f"- APIs with examples: {stats.get('with_examples', 0)} (total examples: {stats.get('total_examples', 0)})")

        print("\nStability distribution:")
        for version, count in stats.get("by_stability", {}).items():
            print(f"  - {version}: {count} APIs")

    if "apis" in data:
        apis = data["apis"]
        print("\nSample of collected APIs:")
        for i, (api_name, api_data) in enumerate(list(apis.items())[:5]):
            print(f"\n{i + 1}. {api_name}:")
            print(f"   Type: {api_data.get('type', 'unknown')}")
            print(f"   Stable since: {api_data.get('stability', 'Unknown')}")
            print(f"   Description: {api_data.get('description', 'No description')[:100]}...")

            if api_data.get("deprecated", False):
                print(f"   DEPRECATED: {api_data.get('deprecation_note', 'No details')[:100]}...")

            items = api_data.get("items", [])
            if items:
                print(f"   Contains {len(items)} items ({', '.join(item['name'] for item in items[:3])}...)")

            examples = api_data.get("examples", [])
            if examples:
                print(f"   Has {len(examples)} code examples")

            related = api_data.get("related_modules", [])
            if related:
                print(f"   Related modules: {', '.join(related[:5])}" + ("..." if len(related) > 5 else ""))

    print("\nDetailed information has been saved to the output files.")


if __name__ == "__main__":
    print("Fetching comprehensive Rust API information...")

    start_time = time.time()
    api_data = fetch_rust_api_info()
    elapsed_time = time.time() - start_time
    print(f"Data collection completed in {elapsed_time:.2f} seconds")

    save_to_json(api_data)
    print_summary(api_data)