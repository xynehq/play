import requests
from bs4 import BeautifulSoup
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


def fetch_rust_stable_apis():
    """
    Fetch all stable APIs and their corresponding Rust versions from the Rust docs.
    """
    # Base URL for Rust documentation
    base_url = "https://doc.rust-lang.org"

    try:
        # Get the main page which lists all the stable APIs
        response = requests.get(f"{base_url}/stable/std/", timeout=10)
        response.raise_for_status()  # Raise an exception for 4XX/5XX responses
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the Rust version
        version_element = soup.select_one('meta[name="generator"]')
        rust_version = version_element['content'] if version_element else "Unknown"

        # Extract and clean the Rust version (e.g., "Rustdoc 1.75.0" -> "1.75.0")
        version_match = re.search(r'\d+\.\d+\.\d+', rust_version)
        rust_version = version_match.group(0) if version_match else rust_version

        # Find all modules in the standard library
        modules = []
        for link in soup.select('.item-name a'):
            module_name = link.text.strip()
            if module_name and not module_name.startswith('_'):
                modules.append(module_name)

        print(f"Found {len(modules)} modules to process")

        # Get information about each module's first stable version
        api_versions = {}

        # Get a sample module's HTML to debug structure
        sample_module = modules[0] if modules else None
        if sample_module:
            debug_module_structure(f"{base_url}/stable/std/{sample_module}/")

        def get_module_info(module_name):
            try:
                module_url = f"{base_url}/stable/std/{module_name}/"
                module_response = requests.get(module_url, timeout=10)
                module_response.raise_for_status()

                module_soup = BeautifulSoup(module_response.text, 'html.parser')

                # Try multiple selectors for stability information
                stable_since = extract_version_info(module_soup)

                return (module_name, stable_since)

            except requests.RequestException as e:
                print(f"Error fetching {module_name}: {str(e)}")
                return (module_name, "Error: Request failed")
            except Exception as e:
                print(f"Error processing {module_name}: {str(e)}")
                return (module_name, f"Error: {str(e)}")

        def extract_version_info(soup):
            """
            Extract version information using multiple methods.
            """
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

        def debug_module_structure(url):
            """Debug the structure of a sample module page"""
            try:
                response = requests.get(url, timeout=10)
                soup = BeautifulSoup(response.text, 'html.parser')

                print("\n--- DEBUG: Sample module structure ---")
                # Check what classes might contain version info
                classes = set()
                for tag in soup.find_all(class_=True):
                    classes.update(tag.get('class'))

                print(f"Available classes: {', '.join(sorted(classes))}")

                # Look for elements that might contain version information
                version_candidates = soup.find_all(string=re.compile(r'(version|since)'))
                print(f"Elements containing 'version' or 'since': {len(version_candidates)}")
                for i, elem in enumerate(version_candidates[:3]):  # Show first 3 examples
                    parent = elem.parent
                    print(f"Example {i + 1}: {parent.name} with classes {parent.get('class')}: {elem.strip()[:50]}...")

                print("--- End DEBUG ---\n")
            except Exception as e:
                print(f"Error debugging module structure: {str(e)}")

        # Use multithreading to speed up the process with a timeout
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Submit all tasks and track them
            future_to_module = {executor.submit(get_module_info, module): module for module in modules}

            for future in as_completed(future_to_module, timeout=300):  # 5-minute overall timeout
                module = future_to_module[future]
                try:
                    module_name, stable_since = future.result()
                    api_versions[module_name] = stable_since
                    print(f"Processed {module_name}: {stable_since}")
                except Exception as e:
                    print(f"Exception processing {module}: {str(e)}")
                    api_versions[module] = f"Error: {str(e)}"

        return {
            "current_rust_version": rust_version,
            "stable_apis": api_versions
        }

    except requests.RequestException as e:
        print(f"Error fetching main Rust documentation page: {str(e)}")
        return {
            "current_rust_version": "Error",
            "stable_apis": {}
        }
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return {
            "current_rust_version": "Error",
            "stable_apis": {}
        }


def save_to_json(data, filename="rust_stable_apis.json"):
    """Save the data to a JSON file"""
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Data saved to {filename}")
    except Exception as e:
        print(f"Error saving data to {filename}: {str(e)}")


def print_summary(data):
    """Print a summary of the data"""
    print(f"Current Rust Version: {data['current_rust_version']}")
    print(f"Total Stable APIs: {len(data['stable_apis'])}")

    if data['stable_apis']:
        print("\nSample of Stable APIs and their versions:")
        # Print first 10 APIs as a sample
        for i, (api, version) in enumerate(list(data['stable_apis'].items())[:10]):
            print(f"{i + 1}. {api}: Stable since {version}")

        # Count how many are still unknown
        unknown_count = sum(1 for version in data['stable_apis'].values() if version == "Unknown")
        print(f"\nUnknown versions: {unknown_count} out of {len(data['stable_apis'])}")

        # Show a sample of those with known versions
        known_apis = {api: version for api, version in data['stable_apis'].items() if version != "Unknown"}
        if known_apis:
            print("\nSample of APIs with known versions:")
            for i, (api, version) in enumerate(list(known_apis.items())[:5]):
                print(f"{i + 1}. {api}: Stable since {version}")
    else:
        print("No stable APIs found or an error occurred.")


def direct_extract_test():
    """Test direct extraction from a known module"""
    print("\n--- DIRECT EXTRACTION TEST ---")
    try:
        url = "https://doc.rust-lang.org/stable/std/option/enum.Option.html"
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Save HTML for analysis
        with open("option_debug.html", "w", encoding="utf-8") as f:
            f.write(soup.prettify())

        print(f"Downloaded page: {url}")
        print(f"Page size: {len(response.text)} bytes")
        print("HTML saved to option_debug.html for manual inspection")

        # Try all methods to extract version
        # Look for version in any attribute
        for tag in soup.find_all(True):
            for attr_name, attr_value in tag.attrs.items():
                if isinstance(attr_value, str) and re.search(r'\d+\.\d+\.\d+', attr_value):
                    print(f"Found version in {tag.name} [{attr_name}]: {attr_value}")

        # Check for specific "since" text
        version_text = soup.find_all(string=re.compile(r'since', re.IGNORECASE))
        for text in version_text:
            print(f"Found 'since' text: {text.strip()}")
            match = re.search(r'since\s+(\d+\.\d+\.\d+)', text, re.IGNORECASE)
            if match:
                print(f"  Extracted version: {match.group(1)}")

        # Check for version information in the detailed description
        detail_sections = soup.select('.docblock')
        for section in detail_sections:
            section_text = section.get_text()
            if 'since' in section_text.lower():
                print(f"Found section with 'since': {section_text[:100]}...")
                match = re.search(r'since\s+(\d+\.\d+\.\d+)', section_text, re.IGNORECASE)
                if match:
                    print(f"  Extracted version: {match.group(1)}")

        print("--- END DIRECT EXTRACTION TEST ---\n")
    except Exception as e:
        print(f"Error in direct extraction test: {str(e)}")


if __name__ == "__main__":
    print("Fetching Rust stable APIs and their versions...")

    # Run a direct test first to understand the structure
    direct_extract_test()

    start_time = time.time()
    data = fetch_rust_stable_apis()
    elapsed_time = time.time() - start_time
    print(f"Fetch completed in {elapsed_time:.2f} seconds")

    save_to_json(data)
    print_summary(data)