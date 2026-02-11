"""


Reverse the words in a sentence. For example, “I work for Microsoft” becomes “Microsoft for work I.”
"""

input_word = "I work for Microsoft"

mutable_word = []
for cur in input_word:
    mutable_word.append(cur)

"""
2 pass

tfos....  I



"""


result = ""
cur_char = ""
for i in range(len(mutable_word)//2):
    cur_char = mutable_word[i]
    mutable_word[i] =  mutable_word[-i-1]
    mutable_word[-i - 1] = cur_char



cur_word = []
start_id = 0
for i in range(len(mutable_word)):
    if mutable_word[i] == " ":
        cur_word.reverse()
        cur_word.append(" ")
        for k, j in enumerate(range(start_id, start_id+len(cur_word))):
            mutable_word[j] = cur_word[k]
        start_id = i+1
        cur_word = []
    else:
        cur_word.append(mutable_word[i])

for cur_char in mutable_word:
    print(cur_char, end="")



# print(result)