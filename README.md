# Diversified Answer Generation QA System

## Completed Tasks:
- wrote the script `urls_scraper.py` to scrape the urls of the individual blog posts by Ask EP, save them and easily update the stored urls with the newest ones.
- wrote the script `content_scraper.py` to scrape the relevant information for the newest blog posts and store them temporarily in a .json file.
- created cron jobs for running the two scrapers periodically and logging the changes.

## Next Tasks:
- change the structure of the stored information from each content (there is not a 1-to-1 correspondence between section titles and section texts)
- add the possibility to store the scraped content directly in an OpenSearch index.