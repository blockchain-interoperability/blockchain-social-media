def extract_hashtags(
    tweet: str
) -> list[str]:
    return [w[1:] for w in tweet.split() if w.startswith("#")]
