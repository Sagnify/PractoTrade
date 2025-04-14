import requests

def fetch_reddit_posts(company_name, subreddit='stocks', limit=20):
    url = f'https://www.reddit.com/r/{subreddit}/search.json?q={company_name}&restrict_sr=1&sort=new&limit={limit}'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()

        posts = []
        for post in data['data']['children']:
            post_title = post['data']['title']
            post_url = post['data']['url']
            post_content = post['data'].get('selftext', 'No content')  # Some posts may not have text content

            # Add the post details to the posts list
            posts.append({
                'title': post_title,
                'url': post_url,
                'content': post_content
            })
        
        return posts
    else:
        print(f"Error fetching posts: {response.status_code}")
        return []

def analyze_sentiment(content):
    # Basic sentiment analysis: Check if the content has words like "good", "positive", "strong" or "bad", "weak", "decline"
    positive_keywords = ['good', 'positive', 'strong', 'up', 'improve', 'growth']
    negative_keywords = ['bad', 'negative', 'weak', 'down', 'decline', 'fall']

    content = content.lower()

    positive_score = sum(word in content for word in positive_keywords)
    negative_score = sum(word in content for word in negative_keywords)

    if positive_score > negative_score:
        return 'Positive'
    elif negative_score > positive_score:
        return 'Negative'
    else:
        return 'Neutral'

# Example Usage
company_name = 'Tesla'
posts = fetch_reddit_posts(company_name, limit=20)

# Print the post titles and analyze sentiment
for post in posts:
    print(f"Title: {post['title']}")
    print(f"URL: {post['url']}")
    print(f"Content: {post['content'][:200]}...")  # Show a snippet of the content (first 200 characters)
    sentiment = analyze_sentiment(post['content'])
    print(f"Sentiment: {sentiment}")
    print('-' * 50)
