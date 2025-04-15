import feedparser


def fetch_clean_google_news(company_name, limit=20):
    rss_url = f'https://news.google.com/rss/search?q={company_name}'
    feed = feedparser.parse(rss_url)
    articles = []

    for entry in feed.entries[:limit]:
        title = str(entry.title).strip()

        articles.append({
            'title': title,
            'published': entry.published,
            # 'snippet': snippet
        })

    return articles

# Example usage
company_name = 'Tesla'
articles = fetch_clean_google_news(company_name)

for i, article in enumerate(articles, 1):
    print(f"{i}. Title: {article['title']}")
    print(f"   Published: {article['published']}")
    print('-' * 80)
