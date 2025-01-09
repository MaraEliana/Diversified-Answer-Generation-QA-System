import json

if __name__ == "__main__":
    # load the original JSON file
    input_file = "external_urls.json"  
    output_file = "all_urls.json"

    # read the input JSON file
    with open(input_file, "r") as file:
        data = json.load(file)

    # extract all URLs
    all_urls = []
    for item in data:
        if "links" in item:
            all_urls.extend(item["links"])

    # write the URLs to a new JSON file
    with open(output_file, "w") as file:
        json.dump(all_urls, file, indent=4)

    print(f"Extracted URLs have been saved to {output_file}. There are {len(all_urls)} URLs in total.")
