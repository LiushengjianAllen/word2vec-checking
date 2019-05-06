import jieba
import jieba.analyse
txt='那些你很冒险的梦，我陪你去疯，折纸飞机碰到雨天终究会坠落，伤人的话我直说，因为你会懂，冒险不冒险你不清楚，折纸飞机也不会回来，做梦的人睡不醒！'
Key=jieba.analyse.extract_tags(txt,3)
print(Key)