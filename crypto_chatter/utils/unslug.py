def unslug(
    slug_text:str,
    capitalize=True,
    delim="_",
):
    return " ".join([
        (w if not capitalize else w.capitalize()) 
        for w in slug_text.split(delim)
    ])
