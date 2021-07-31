
def bucketized_age(age):
    age_threshold = [18, 25, 30, 35, 40, 45, 50, 55, 60, 65]
    index = 0
    for t in age_threshold:
        if age < t:
            return index
        index += 1
    return index