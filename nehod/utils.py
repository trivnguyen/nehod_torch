
import numpy as np
try:
    from wonderwords import RandomWord
except:
    RandomWord = None
    continue

def get_random_name():
    """ Generate a random name """
    if RandomWord is None:
        return "default-run-0"
    w = RandomWord()
    word1 = " "
    while "-" in word1 or " " in word1:
        word1 = w.word(
            include_categories=["adjective"], word_min_length=3, word_max_length=6)
    word2 = " "
    while "-" in word2 or " " in word2:
        word2 = w.word(
            include_categories=["noun"], word_min_length=3, word_max_length=6)
    number = np.random.randint(0, 100)
    return f"{word1}-{word2}-{number}"
