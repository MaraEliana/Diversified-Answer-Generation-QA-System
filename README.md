# Diversified Answer Generation QA System

## Completed Tasks:
- QA index:
    - Added an explicit ID for each document (the url is the unique identifier).
    - Added a field for marking that the same content is present in multiple languages.
    - All web pages can be indexed now. The problem with the 23 documents was automatically solved after changing the code.
    - Removed non-English content.
    - Used LLM to decide if the title or the first paragraph should be the question.
- Knowledge base index:
    - Added custom identifier for each document (url + chunk id).
    - I keep track of the domains from which the URLs come (helpful for the final presentation).
    - Used crawl4ai for scraping the content of non_pdfs.
    - Analysed pdf urls and determined that MarkItDown seems a good fit given that most pdfs have tabular data.
    - Indexed all non-pdf documents.
- RAG pipeline:
- Evaluation:
    - Looked into alpha-NDCG. Cannot implement it without further discussions.


## Pending Taks:
- QA index:
    - Take the answer and ask an LLM to formulate a question for it and store this question in a separate field.
- Compute text analytics.
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

NOTE: adjust the embedding dimension if necessary

- experiment with multiple retrieval strategies and possibly multiple embedding models and configurations
- create a script that scrapes the contents of the web pages from the external links:
    1. use Beautiful Soup. ✅
    2. use MarkItDown (https://github.com/microsoft/markitdown) and then process the markdown files. ✅
- compare and improve.

-Crawl4AI does not support pdfs

# NOTE:
- Current index for knowledge-base is: "eur-lex-diversified-knowledge-base-3" (vector search does not work on 2)
- Text splitter: RecursiveCharacterTextSplitter with chunk_size=6000 characters and chunk_overlap=200 characters.
- Embedding model: "text-embedding-3-small" with embedding dimension 1536.
- Total cost for embedding all documents: 0.44$.
- Mention that 99% of the documents could be indexed before an error occurred. Say what that error means and how I was able to fix it.

# Reasons for selecting the above:
- for the text splitter
- for the chunk size
- for scraping strategy
- why not use a free embedding model? what are the disadvantages?

# Good data for presentation
- Indexing non-pdf documents: start: 13:32:02,346 - end: 15:25:13,766 (for 99% of the urls), start: 17:43:03,185 - end: 17:43:56,806 (for the last 20 urls)
- There are 11417 documents from non-pdf urls.
- Indexing pdf documents: start: 19:02:55,085 - end: 19:35:24,141, start: 20:05:11,074 - end: 20:10:02,596
- There are 3497 documents from pdf urls.
- There are 135 pdf urls. 87 of them contain tables.
- Now the final knowledge base has 14914 documents.