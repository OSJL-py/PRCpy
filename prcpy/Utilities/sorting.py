import re

def natural_sort_key(s):

    _nsre = re.compile('([0-9]+)')

    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]
