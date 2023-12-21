import math

def cosine_similarity(dict1,dict2, max_size):
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"

    
    v1 = [0] * max_size
    v2 = [0] * max_size
    
    for key, value in dict1.items():
        v1[key] += value
    for key, value in dict2.items():
        v2[key] += value

    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(max_size):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
            
    if sumxx == 0 or sumyy == 0:
        return 0.0
    print(sumxy/math.sqrt(sumxx*sumyy))
    return sumxy/math.sqrt(sumxx*sumyy)

def cosine_similarity2(v1,v2, max_size):
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"

    # v1 = [0] * max_size
    # v2 = [0] * max_size
    
    # for key, value in dict1.items():
    #     v1[key] += value
    # for key, value in dict2.items():
    #     v2[key] += value

    dot_product = sum(v1.get(term, 0) * v2.get(term, 0) for term in set(v1) & set(v2))
    magnitude1 = math.sqrt(sum(value**2 for value in v1.values()))
    magnitude2 = math.sqrt(sum(value**2 for value in v2.values()))

    if magnitude1 == 0 or magnitude2 == 0:
        return 0

    return dot_product / (magnitude1 * magnitude2)