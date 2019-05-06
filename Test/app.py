# encoding=utf-8
from flask import Flask
from flask import request
from flask import render_template
import ex
import sys
import importlib
#from flask_bootstrap import Bootstrap
importlib.reload(sys)

#app = Flask(__name__, template_folder='./templates',static_folder="",static_url_path="")
app = Flask(__name__)
#bootstrap = Bootstrap(app)

@app.route('/',methods=[ 'GET'])
#@app.route('/index',methods=['GET'])
#get从服务器获取数据
def home_form():
    return render_template("query.html", query = ex.docxPath)
    #先引入dou模版，然后传参数。
@app.route('/signin',methods=['POST'])
#一般登陆时用post
def signin():
    queryPath = request.form['file']
    textflag = queryPath.split(".")
    queryText = textflag[0] + ".txt"
    targetText = ex.getTextFromDocx(queryPath, queryText)
    target = [targetText]
    targetLib_texts = ex.pre_process_cn(target)
    # 选择一个基准数据
    ml_txt = ex.lib_texts[0]
    # 词袋处理
    targetBow = ex.dictionary.doc2bow(ml_txt)
    # 在上面选择的模型数据 lsi 中，计算其他数据与其的相似度
    targetLsi = ex.lsi[targetBow]
    sims = ex.index[targetLsi]
    sort_sims = sorted(enumerate(sims), key=lambda item: -item[1])
    simpath = ex.docxPath[sort_sims[1][0]]
    simtext = ex.text[sort_sims[1][0]]
    # print "最相似文本：
    # print sort_sims[1] ??
    # print text[sort_sims[1][0]]??
    textsort = []
    simsort = []
    for i in range(1, 10):
        textsort.append(ex.docxPath[sort_sims[i][0]])
        simsort.append(sort_sims[i][1])
    evertsim = zip(textsort, simsort)
    result = ex.sentenceCompare(targetText, simtext)
    return render_template('home.html', querypath=queryPath, targettext=targetText, simpath=simpath, simtext=simtext,
                           sim=sort_sims[1][1], result=result, everysim=evertsim)


if __name__ == '__main__':
    app.run()

