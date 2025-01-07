# Diversified Answer Generation QA System

## Completed Tasks:
- wrote the script `urls_scraper.py` to scrape the urls of the individual blog posts by Ask EP, save them and easily update the stored urls with the newest ones.
- wrote the script `content_scraper.py` to scrape the relevant information for the newest blog posts and store them temporarily in a .json file.
- created cron jobs for running the two scrapers periodically and logging the changes.

## Next Tasks:
- change the structure of the stored information from each content (there is not a 1-to-1 correspondence between section titles and section texts)
- add the possibility to store the scraped content directly in an OpenSearch index.


PROBLEMS:
- 
- languages
- strong tag was not detected (look at page_3.json)

NOTES FOR SCRAPING EXTERNAL DOCS:
- Some of them are links to EUR-Lex pages and those contains links to simplified versions with the same content. Would be very helpful.
- Use MarkItDown.
- Beware of the PDFs.
- First do not take care of text hierarchy.[- ]