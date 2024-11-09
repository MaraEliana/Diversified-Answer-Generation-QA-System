import os
from dotenv import load_dotenv
from opensearchpy import OpenSearch

# load environment variables from the .env file
load_dotenv()

# read credentials from environment variables
opensearch_user = os.getenv('OPENSEARCH_USER')
opensearch_password = os.getenv('OPENSEARCH_PASSWORD')

# OpenSearch client
client = OpenSearch(
    hosts=[{"host": "opensearch-ds-2.ifi.uni-heidelberg.de", "port": 443}],
    http_auth=(opensearch_user, opensearch_password),
    use_ssl=True,
    verify_certs=True,
    assert_hostname=False,
    ssl_show_warn=False
)
print(client)
print(client.indices.exists(index="eur-lex-diversified-test"))
# info = client.info()
# print(f"Welcome to {info['version']['distribution']} {info['version']['number']}!")