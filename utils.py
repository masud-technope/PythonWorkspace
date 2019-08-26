def get_max(items):
    max = items[0]
    for item in items:
        if item > max:
            max = item
    return max
