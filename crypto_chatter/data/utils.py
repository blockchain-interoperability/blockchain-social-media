def preprocess_text(text:str) -> str:
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        t = '' if t.startswith('0x') else t
        new_text += [t]
    return " ".join(new_text)

def is_spam(text:str) -> bool:
    if 'i wish i discovered this earlier' in text.lower():
        return True
    if 'uniswap is being exploited by this dude' in text.lower():
        return True
    if 'more than $200k so far' in text.lower():
        return True

    return False
