import os
import re
import jieba_zhtw as jb

POST_PATH = "./_posts/"
PROC_PATH = "./processed_posts/"
CUT_PATH = "./cut_posts/"
WRITE_BLOG_RAW = False

filename_list = []
for file in os.listdir(POST_PATH) :
    filename_list.append(file)
filename_list = sorted(filename_list)

def tag_remover(post_string) :
    temp = ""
    result = ""
    is_ignored = False
    
    for c in post_string :
        if (c == '\t' or c == '\n' or c == '\r') :
            continue
        elif (c == ' ') :
            if (len(temp) == 0) :
                if (result.endswith(' ') or result.endswith('\n')) :
                    continue
            elif temp.endswith(' ') :
                continue                
        
        if not is_ignored :
            if (c == '<') :
                if ("AC_FL" not in temp
                and "Show" not in temp
                and "Hide" not in temp
                and "\'/>" not in temp
                and "title" not in temp
                and ";}" not in temp
                and "document" not in temp
                ) :
                    result += temp
                is_ignored = True
                temp = ""
            else :
                temp += c;
                if (temp.find("&lt;") != -1) :
                    # this could be in context, so is treated more carefully
                    # nepeta your quirk is bothering me
                    # also, if &lt; comes before nbsp, equa, amp or |, do not ignore. 
                    if (":33 &lt;" in temp
                        or "&lt; " in temp
                        or "&lt;=" in temp
                        or "&lt;&" in temp
                        or "&lt;|" in temp
                    ) :
                        temp = re.sub("\&lt;", "<", temp)
                        result += temp
                        is_ignored = False
                    else :
                        if ("AC_FL" not in temp
                        and "Show" not in temp
                        and "Hide" not in temp
                        and "\'/>" not in temp
                        and "title" not in temp
                        and ";}" not in temp
                        and "document" not in temp
                        ) :
                            temp = re.sub("\&lt;", "", temp)
                            result += temp
                        is_ignored = True
                    temp = ""
        #end if not is_ignored
        else : #is_ignored
            temp += c
            if (c == '>') :
                there_is_br = temp.endswith("br />") or temp.endswith("br/>") or temp.endswith("br>")
                # 3 diffent types of br tag :/
                if (there_is_br and not result.endswith('\n')) :
                    result += "\n"
                is_ignored = False
                temp = ""
            elif (len(temp) >= 5 and "&gt;" in temp[len(temp) - 5:]) :
                if (there_is_br and not result.endswith('\n')) :
                    result += "\n"
                is_ignored = False
                temp = ""
        #end else (is_ignored)
    # end for c in post_string
    if not result.endswith('\n') : result += '\n'
    return result
# end def tag_remover

blog_raw = open("blog-raw.txt",  'w+',  encoding = 'utf-8-sig')
post_string = ""
for file_i, filename in enumerate(filename_list) :
    if file_i % 1000 == 0 : print(file_i)
    old_file_lines = open(POST_PATH + filename, 'r', encoding = 'utf-8-sig').readlines()
    
    # get title
    title = old_file_lines[2][7:]
    if title.startswith('"') :
        title = title[1 : len(title) - 2] + '\n' # because it ends with '\"\n'
    # get post begin line num
    begin_line = 0
    for n, line in enumerate(old_file_lines[1:]):
        if line == "---":
            begin_line = n
            break
            
    new_lines = [title]
    new_lines += old_file_lines[begin_line+2:]
    for new_l in new_lines :
        post_string += new_l
    post_string = tag_remover(post_string)
    post_string = re.sub("&nbsp;", " ", post_string)
    post_string = re.sub("&gt;", ">", post_string)
    post_string = re.sub("&#42;", "*", post_string)
    post_string = re.sub("&#124;", "|", post_string)
    post_string = re.sub("&amp;", "&", post_string)
    new_file = open(PROC_PATH + str(file_i) + ".txt", 'w', encoding = 'utf-8-sig')
    new_file.write(post_string)
    if WRITE_BLOG_RAW :
        blog_raw.write(post_string)
        blog_raw.write("\n")        
    post_string = ""

if WRITE_BLOG_RAW :
    blog_raw.write(open("blog_nonposts.txt", 'r', encoding = 'utf-8-sig').read())
blog_raw.close()

def sorting_file_name(element) :
    if not element.endswith(".txt") :
        print(element, " <-- This is not a txt file. wtf did you mess out?")
        exit(0)
    return int(element[0 : len(element) - 4])

proc_post_list = []
for filename in os.listdir(PROC_PATH) :
    proc_post_list.append(filename)
proc_post_list = sorted(proc_post_list, key = sorting_file_name)

### JIEBA ###
print("CREATE_NEW_JIEBA...")
jb.dt.cache_file = 'jieba.cache.zhtw'
jb.load_userdict('hs_dict.dict')
print("cutting posts...")
for postname in proc_post_list :
    jieba_in_string = open(PROC_PATH + postname, 'r', encoding = "utf-8-sig").read() 
    jieba_in_string = re.sub(r" +", " ", jieba_in_string)
    cut_post_list = jb.cut(jieba_in_string, cut_all = False)
    cut_post_string = " ".join(cut_post_list)
    cut_post_string = re.sub("= = = = = = >", "======>", cut_post_string)
    cut_post_string = re.sub("= = >", "==>", cut_post_string)
    cut_post_string = re.sub(r" +", " ", cut_post_string)
    open(CUT_PATH + postname, 'w+', encoding = 'utf-8-sig').write(cut_post_string)
### END JIEBA ###

