import os
import json

if __name__ == "__main__":
    folder_path = "pages"
    all_links = []
    # iterate through files in the directory
    for filename in os.listdir(folder_path):
        if filename.startswith("page_") and filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            
            # load each JSON file and extract the "links" field
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if "links" in data and isinstance(data["links"], list):
                    all_links.extend(data["links"])

    # save the collected URLs to all_urls.json
    output_file = "all_urls.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_links, f, indent=2)

    print(f"Collected {len(all_links)} links and saved to {output_file}.")
