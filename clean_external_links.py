import json
from urllib.parse import urlparse
import requests
from tqdm import tqdm

def save_urls(file_path, urls):
    with open(file_path, 'w') as file:
        json.dump(urls, file, indent=4)

def is_pdf(url):
    try:
        response = requests.get(url, stream=True, timeout=10, allow_redirects=True)
        response.raise_for_status()  # raise an error for HTTP issues

        # check headers
        content_type = response.headers.get('Content-Type', '')
        print(f"URL: {url} | Content-Type: {content_type}")

        # check for PDF MIME type in headers
        if 'application/pdf' in content_type:
            return True

        # read magic number from the first few bytes
        first_bytes = response.raw.read(1024)  # Read the first 1KB for better accuracy
        if first_bytes.startswith(b'%PDF'):
            return True

        return False

    except requests.exceptions.RequestException as e:
        print(f"HTTP error for URL: {url} | Error: {e}")
        raise e
    except Exception as e:
        print(f"Error for URL: {url} | Error: {e}")
        raise e

if __name__ == "__main__":
    input_file = "all_urls.json"
    pdf_file = "pdf_urls.json"
    non_pdf_file = "non_pdf_urls.json"
    error_file = "error_urls.json"

    with open(input_file, "r") as f:
        urls = json.load(f)

    print(f"Found {len(urls)} external links")
    # remove duplicates by converting to a set and back to a list
    unique_urls = list(set(urls))
    print(f"Found {len(unique_urls)} unique external links")

    # remove urls that are not http or https
    valid_urls = [url for url in unique_urls if url.startswith("http")]
    print(f"Found {len(valid_urls)} external links starting with http or https.")

    # remove YouTube links
    valid_urls = [url for url in valid_urls if 'youtu.be' not in url]
    print(f"Found {len(valid_urls)} external links not from YouTube.")

    # initialize lists for categorization
    pdf_urls = []
    nonpdf_urls = []
    error_urls = []

    # iterate through URLs and categorize
    for url in tqdm(valid_urls):
        try:
            if is_pdf(url):
                pdf_urls.append(url)
            else:
                nonpdf_urls.append(url)
        except requests.exceptions.RequestException as e:
            error_urls.append({"url": url, "error": str(e)})
        except Exception as e:
            error_urls.append({"url": url, "error": str(e)})
    
    # save categorized URLs
    save_urls(pdf_file, pdf_urls)
    save_urls(non_pdf_file, nonpdf_urls)
    save_urls(error_file, error_urls)