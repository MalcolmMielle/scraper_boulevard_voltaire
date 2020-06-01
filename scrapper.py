from bs4 import BeautifulSoup
from newspaper import Article
import requests
import pandas as pd


def scrap_archive_page(page: int) -> int:
    url = "https://www.bvoltaire.fr/archives/page/" + str(page) + "/"
    print("Scrapping page ", url)
    # headers = requests.utils.default_headers()
    # headers.update({'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0'})
    req = requests.get(url)
    soup = BeautifulSoup(req.text, 'html.parser')
    links = set()
    for link in soup.find_all('h3'):
        class_ret = link.get('class')
        if class_ret is not None and class_ret[0] == 'article-title':
            print(link.a.get('href'))
            links.add(link.a.get('href'))
    articles_text = list()
    articles_title = list()
    articles_tag = list()
    articles_publish_date = list()
    articles_urls = list()
    for i, link in enumerate(links):
        print("art ", i, " out of", len(links))
        article = Article(link)
        article.download()
        article.parse()
        articles_text.append(article.text)
        articles_title.append(article.title)
        articles_tag.append(article.tags)
        articles_publish_date.append(article.publish_date)
        articles_urls.append(article.url)

    column = ["title", "text", "tags", "publish_date", "url"]
    data_list = [articles_title, articles_text, articles_tag, articles_publish_date, articles_urls]
    print(column)
    sample_data = pd.DataFrame(data_list, column).T
    sample_data.to_csv("data/bvoltaire_archive_page_" + str(page) + ".csv", index=False)
    return len(articles_text)


def scrap_archives():
    nb_of_pages = 500
    start_page = 51
    print("Scrap")
    total_articles = 0
    for page in range(start_page, nb_of_pages + 1):
        num_art = scrap_archive_page(page)
        total_articles += num_art
    print("Saved ", total_articles, " articles :D")


def main():
    scrap_archives()


if __name__ == "__main__":
    main()
