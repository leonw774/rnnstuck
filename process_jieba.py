import os
import re
import sys
import jieba_zhtw as jb

PROC_PATH = "processed_posts/"
CUT_PATH = "cut_posts/"

def sorting_file_name(element) :
    if not element.endswith(".txt") :
        print(element, " <-- This is not a txt file. wtf did you mess out?")
        sys.exit(0)
    return int(element[0 : len(element) - 4])

print("\nprocessing path names...\n")
POSTNAME_LIST = []
for filename in os.listdir(PROC_PATH) :
    POSTNAME_LIST.append(filename)
POSTNAME_LIST = sorted(POSTNAME_LIST, key = sorting_file_name)

### JIEBA ###
print("CREATE_NEW_JIEBA...")
jb.dt.cache_file = 'jieba.cache.zhtw'
jb.load_userdict('hs_dict.dict')
for postname in POSTNAME_LIST :
    jieba_in_string = open(PROC_PATH + postname, 'r', encoding = "utf-8-sig").read() 
    jieba_in_string = re.sub(r" +", " ", jieba_in_string)
    cut_post = jb.cut(jieba_in_string, cut_all = False)
    open(CUT_PATH + postname, 'w+', encoding = 'utf-8-sig').write(" ".join(cut_post))
### END JIEBA ###