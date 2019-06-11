import sys
import os
import jieba
"""
文本分类例子，数据预处理
"""
"""
需要完成：
1. 分词[中文一般都需要分词]
2. 词语数字化：
    词表
    matrix -> [|V|, embed_size]
    词语A -> id(5)
3. label -> id
"""
# input file
train_file = './data/cnews.train.txt'
val_file = './data/cnews.val.txt'
test_file = './data/cnews.test.txt'
# output file
seg_train_file = './data/cnews.train.seg.txt'
seg_val_file = './data/cnews.val.seg.txt'
seg_test_file = './data/cnews.test.seg.txt'
# 词表映射
vocab_file = './data/cnews.vocab.txt'
category_file = './data/cnews.category.txt'


def generate_seg_file(input_file, output_seg_file):
    """Segment the sentences in each line in input_file"""
    with open(input_file, encoding='utf-8') as f:
        lines = f.readlines()
    with open(output_seg_file, 'w', encoding='utf-8') as f:
        for line in lines:
            label, content = line.strip('\r\n').split('\t')
            word_iter = jieba.cut(content)
            word_content = ''
            # jieba分词对空格处理不好，需要特殊处理
            for word in word_iter:
                word = word.strip(' ')
                if word != '':
                    word_content += word + ' '
            out_line = '%s\t%s\n' % (label, word_content.strip(' '))
            f.write(out_line)


if not os.path.exists(seg_train_file):
    generate_seg_file(train_file, seg_train_file)
if not os.path.exists(seg_val_file):
    generate_seg_file(val_file, seg_val_file)
if not os.path.exists(seg_test_file):
    generate_seg_file(test_file, seg_test_file)


def generate_vocab_file(input_seg_file, out_vocab_file):
    with open(input_seg_file, encoding='utf-8') as f:
        lines = f.readlines()
    word_dict = {}
    for line in lines:
        label, content = line.strip('\r\n').split('\t')
        for word in content.split():
            word_dict.setdefault(word, 0)
            word_dict[word] += 1
    # [(word, frequrncy), …, ()]
    sorted_word_dict = sorted(
        word_dict.items(),
        key=lambda d:d[1],
        reverse=True)
    with open(out_vocab_file, 'w', encoding='utf-8') as f:
        f.write('<UNK>\t1000000\n')
        for item in sorted_word_dict:
            f.write('%s\t%d\n' % (item[0], item[1]))


def generate_category_file(input_file, out_category_file):
    with open(input_file, encoding='utf-8') as f:
        lines = f.readlines()
    category_dict = {}
    for line in lines:
        label, content = line.strip('\r\n').split('\t')
        category_dict.setdefault(label, 0)
        category_dict[label] += 1
    # category_number = len(category_dict)
    with open(out_category_file, 'w', encoding='utf-8') as f:
        for category in category_dict:
            line = '%s\n' % category
            print('%s\t%d' % (category, category_dict[category]))
            f.write(line)


if not os.path.exists(vocab_file):
    generate_vocab_file(seg_train_file, vocab_file)
if not os.path.exists(category_file):
    generate_category_file(seg_train_file, category_file)




