import math

def cosine_similarity(v1,v2):
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"

    # if len(v1) != len(v2):
    #     raise ValueError("Input vectors must have the same length")
    
    sumxx, sumxy, sumyy = 0, 0, 0
    for key, value in v1.items():
        if key in v2:
            x = v1[key]; y = v2[key]
            sumxx += x*x
            sumyy += y*y
            sumxy += x*y
            
    if sumxx == 0 or sumyy == 0:
        return 0.0
    
    return sumxy/math.sqrt(sumxx*sumyy)