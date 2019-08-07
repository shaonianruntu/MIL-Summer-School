# 数据处理
import sys
import os  # 系统命令
import json
import re

def parseRawData(author = None, constrain = None):
    rst = []

    def sentenceParse(para):
        result, number = re.subn("（.*）", "", para)  # 字符串替换
        result, number = re.subn("（.*）", "", para)
        result, number = re.subn("{.*}", "", result)
        result, number = re.subn("《.*》", "", result)
        result, number = re.subn("《.*》", "", result)
        result, number = re.subn("[\]\[]", "", result)
        r = ""
        for s in result:
            if s not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-']:
                r += s;
        r, number = re.subn("。。", "。", r)
        return r

    def handleJson(file):
        # print file
        rst = []
        # with open(file, 'r', encoding='UTF-8') as f:
        #     data = json.loads(f)
        # #     fp.read('labels.json, 'r'', encoding ='UTF - 8')
        data = json.loads(open(file,'rb').read())
        for poetry in data:
            pdata = ""
            if (author!=None and poetry.get("author")!=author):
                continue
            p = poetry.get("paragraphs")
            flag = False
            for s in p:
                sp = re.split("[，！。]", s)
                for tr in sp:
                    if constrain != None and len(tr) != constrain and len(tr)!=0:
                        flag = True
                        break
                    if flag:
                        break
            if flag:
                continue
            for sentence in poetry.get("paragraphs"):
                pdata += sentence
            pdata = sentenceParse(pdata)
            if pdata!="":
                rst.append(pdata)
        return rst
    # print sentenceParse("")
    data = []
    src = './chinese-poetry/json/'
    for filename in os.listdir(src):
        if filename.startswith("poet.tang"):
            data.extend(handleJson(src+filename))
    return data



# if __name__=='__main__':
#     print parseRawData.sentenceParse("熱暖將來賓鐵文，暫時不動聚白雲。撥卻白雲見青天，掇頭裏許便乘仙。（見影宋蜀刻本《李太白文集》卷二十三。）（以上繆氏本《太白集》）-362-。")
#