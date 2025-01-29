# Diversified Answer Generation QA System

## Completed Tasks:
- QA index:
    - Added an explicit ID for each document (the url is the unique identifier).
    - Added a field for marking that the same content is present in multiple languages.
    - All web pages can be indexed now. The problem with the 23 documents was automatically solved after changing the code.
    - Removed non-English content.
    - Used LLM to decide if the title or the first paragraph should be the question.


## Pending Taks:
- QA index:
    - Take the answer and ask an LLM to formulate a question for it and store this question in a separate field.
    - Compute text analytics.
- Knowledge base:
    - Add explicit ID for each document.
    - Keep track of the domains from which the URLs come (helpful for the final presentation).
    - Compute text analytics.
    - Analyse the linked PDFs to see how many include tabular data. This then determines if the conversion to markdown using MarkItDown is justified. See if LlamaParse  can convert to MarkDown while also preserving text hierarchy.
    - If the web page/PDF has some internal hierarchy, then keep it (store for each chunk of a paragraph the corresponding section title; this is helpful for a preliminary filtering stage when given a query).
    - Experiment with different LangChain text splitters (LangChain Text Splitters).
    - Since the content of most linked web pages is from the legal domain, try using a legal domain embedding model.
    - Reindex the documents after applying the above changes.

- Explore crawl4ai. 
- upload presentation on Gitlab in a separate folder.
- update the remote repository.
- Look into papers regarding diversified question/text generation and see what metrics they use.
- Metric for diversified retrieval: alpha-nDCG.
- Consider three approaches:
        - first: naive RAG.
        - second: use the MMRRetriever from LangChain to encourage diversified text retrieval.
        - third approach: add clustering (take retrieved documents, cluster them and generate part of the answer for each cluster)
- For clustering check out the paper Improving RAG Systems via Sentence Clustering and Reordering. 
- Filtering through metadata can be easily achieved with retrievers from LangChain. Make sure to check documentation for that.


## Next Tasks:

## Test pages:
- for QA index:
        - https://epthinktank.eu/2024/05/22/ revision-of-the-eus-common-agricultural-policy-answering-citizens-concerns/ (check if only the English text is crawled)
        - https://epthinktank.eu/2024/02/22/rail-passenger-rights-how-does-the-european-union-protect-your-rights-when-travelling-by-train/ (example of page where the title is the better choice for question)
        - https://epthinktank.eu/2024/01/31/planned-eu-wide-recognition-of-parenthood-answering-citizens-concerns/ (better to take the first paragraph as the question)
- for knowledge base index:



## PROBLEMS:
- Have to eliminate the strong tag used for subtitles because it is also being used for normal words.
- Some of the external documents which are linked are not in English. Most of the time they are in French.
- There are 134 pdfs linked (actually more than that because some do not have the .pdf extension in the url like "http://register.consilium.europa.eu/doc/srv?l=EN&t=PDF&gc=true&sc=false&f=ST%207029%202014%20REV%201")
- The pages like:
    - "https://epthinktank.eu//?s=syria"
    - "http://epthinktank.eu/?s=TTIP",
    - "http://epthinktank.eu/?s=migration",
    - "http://epthinktank.eu/author/epanswers/",
    - "http://epthinktank.eu/iraqikurds/",
    - "http://epthinktank.eu/tag/development-policy/",
    - "http://epthinktank.eu/tag/youth-employment/",
only show the titles of blog posts, but do not have much content themselves.


NOTES FOR SCRAPING EXTERNAL DOCS:
- Some of them are links to EUR-Lex pages and those contains links to simplified versions with the same content. Would be very helpful.
- Use MarkItDown.
- Beware of the PDFs.
- First do not take care of text hierarchy.

PLAN: ✅ or ❌
- create a .json file containing all external links. ✅
- analyse the existing links and create special cases if needed:
    1. group urls according to domains and count them. ✅
    2. delete all mailto urls and #_msocom_1". ✅
    3. deleted duplicates. ✅
    4. eliminate Youtube url. ✅
    5. separate the pdfs from the rest. ✅
    NOTE: URLs that end with .pdf are not necessarily pdfs.
    There are 134 pdfs, 1651 non-pdfs and 359 urls which lead to errors (the ones below were eliminated). The following urls were problematic.
        5.1 The following pages could not be found:
        - http://eacea.ec.europa.eu/bilateral_cooperation/index_en.php
        - http://eacea.ec.europa.eu/erasmus_mundus/
        - http://eacea.ec.europa.eu/tempus/
        Eliminated those urls. ✅
        5.2 The following urls cannot be accessed:
        - http://www.elections2014.eu/en/.
        - http://www.elections2014.eu/en/in-the-member-states.
        - http://www.elections2014.eu/en/in-the-member-states/european-union.
        - http://www.elections2014.eu/en/new-commission/hearings/by-committee.
        - http://www.elections2014.eu/en/new-commission/portfolios-and-candidates.
        - http://www.elections2014.eu/en/new-parliament.
        - http://www.elections2014.eu/en/news-room/content/20140918IFG65303/html/Infographic-how-the-European-Commission-will-get-elected.
        - http://www.elections2014.eu/en/press-kit/content/20131112PKH24411/html/Overview-of-Parliament-and-the-2014-elections.
        - https://europa.eu/eyd2015/
        Eliminated those urls.✅
        5.3 Insecure:
        - https://www.avrupa.info.tr/en
        Eliminated url.✅
       
- create a mapping for the Opensearch index of the knowledge base.
NOTE: adjust the embedding dimension if necessary

- experiment with multiple retrieval strategies and possibly multiple embedding models and configurations
- create a script that scrapes the contents of the web pages from the external links:
    1. use Beautiful Soup. ✅
    2. use MarkItDown (https://github.com/microsoft/markitdown) and then process the markdown files. ✅
- compare and improve.

-Crawl4AI does not support pdfs