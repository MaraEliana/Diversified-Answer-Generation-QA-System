import re
import json
from datetime import date, timedelta
from opensearchpy import OpenSearch
from opensearchpy.exceptions import NotFoundError

def load_mapping(file_path):
    """
    Import a mapping of an OpenSearch index from a JSON file.
    :param file_path: Path to the JSON file.
    """
    with open(file_path, 'r') as f:
        return json.load(f)

def create_opensearch_client(host='opensearch-ds-2.ifi.uni-heidelberg.de', port=443, username=None, password=None):
    """
    Create an OpenSearch client.
    :param host: Hostname of the OpenSearch server.
    :param port: Port number of the OpenSearch server.
    :param username: Username for authentication.
    :param password: Password for authentication.
    :return: OpenSearch client object.
    """
    auth = (username, password) if username and password else None
    client = OpenSearch(
        hosts=[{'host': host, 'port': port}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        assert_hostname=False,
        ssl_show_warn=False
    )
    return client

def index_exists(client, index_name):
    return client.indices.exists(index=index_name)

def create_index(client, index_name, mapping, logger=None):
    """
    Create an index in OpenSearch if it doesn't already exist.
    :param client: OpenSearch client.
    :param index_name: Name of the index.
    :param mapping: Mapping definition for the index.
    :param field_limit: Field limit for the index (default: 10000).
    :param logger: Optional logger object.
    """
    field_limit = 100000
    # Add the field limit setting to the mapping
    settings = {
        "settings": {
            "index.mapping.total_fields.limit": field_limit
        }
    }

    # merge settings and mappings
    index_body = {**settings, **mapping}

    # check if the index exists
    if not index_exists(client, index_name):
        logger.info(f"Creating index '{index_name}' with a field limit of {field_limit}...")
        client.indices.create(index=index_name, body=index_body)
        logger.info(f"Index '{index_name}' created with a field limit of {field_limit}.")
    else:
        logger.info(f"Index '{index_name}' already exists.")


def extract_date_from_url(url, logger):
    """
    Extract a date from the URL in YYYY/MM/DD format.
    Example: https://epthinktank.eu/2024/10/24/... -> 2024-10-24
    :param url: URL string.
    :param logger: Logger object.
    :return: Date object if found, otherwise None.
    """
    match = re.search(r'/(\d{4})/(\d{2})/(\d{2})/', url)
    if match:
        year, month, day = map(int, match.groups())
        return date(year, month, day)
    else:
        logger.info(f"Date not found in URL: {url}")
        return None

def add_url_with_date(client, index_name, url, logger=None):
    """
    Add a URL to the index using its extracted date as the date field.
    :param client: OpenSearch client.
    :param index_name: Name of the index.
    :param url: URL string.
    :param logger: Optional logger object.
    """
    extracted_date = extract_date_from_url(url, logger)
    if extracted_date:
        document = {
            "url": url,
            "date": extracted_date
        }
        client.index(index=index_name, body=document)
        logger.info(f"Added URL '{url}' with date '{extracted_date}' to index '{index_name}'.")
    else:
        logger.info(f"Could not add URL '{url}' to index '{index_name}' because extracted_date is None.")

def get_latest_date(client, index_name, logger=None):
    """
    Retrieve the most recent date in the index based on the 'date' field.
    :param client: OpenSearch client.
    :param index_name: Name of the index.
    :return: The latest date as a date object, or None if the index is empty.
    """
    try:
        response = client.search(
            index=index_name,
            body={
                "size": 1,
                "sort": [{"date": {"order": "desc"}}],
                "_source": ["date"]  # Only fetch the date field
            }
        )
        hits = response.get('hits', {}).get('hits', [])
        if hits:
            logger.info(f"Latest date in index '{index_name}': {hits[0]['_source']['date']}")
            latest_date = hits[0]['_source']['date']
            return date.fromisoformat(latest_date)
        else:
            logger.info(f"No documents found in index '{index_name}'.")
    except NotFoundError:
        logger.info(f"Index '{index_name}' does not exist.")
    return None

def get_urls_by_date(client, index_name, target_date, logger=None):
    """
    Retrieve all URLs from the index for a specific date.
    :param client: OpenSearch client.
    :param index_name: Name of the index.
    :param target_date: The date to filter by (date object).
    :param logger: Optional logger object.
    :return: List of URLs matching the target date.
    """
    try:
        date_str = target_date.strftime('%Y-%m-%d')
        next_day = (target_date + timedelta(days=1)).strftime('%Y-%m-%d')

        response = client.search(
            index=index_name,
            body={
                "query": {
                    "range": {
                        "date": {
                            "gte": date_str,
                            "lt": next_day
                        }
                    }
                },
                "_source": ["url"],  # retrieve only the URL field
                "size": 1000 
            }
        )
        hits = response.get('hits', {}).get('hits', [])
        if hits:
            logger.info(f"Found {len(hits)} URLs for date '{target_date}' in index '{index_name}'.")
        return [hit['_source']['url'] for hit in hits]
    except NotFoundError:
        logger.info(f"Index '{index_name}' does not exist.")
        return []
    