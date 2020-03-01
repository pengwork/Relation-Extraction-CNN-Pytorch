# coding=utf-8

'''
Preprocess original Sem-eval task8 data
'''

import json

class2label = {'Other': 0,
               'Message-Topic(e1,e2)': 1, 'Message-Topic(e2,e1)': 2,
               'Product-Producer(e1,e2)': 3, 'Product-Producer(e2,e1)': 4,
               'Instrument-Agency(e1,e2)': 5, 'Instrument-Agency(e2,e1)': 6,
               'Entity-Destination(e1,e2)': 7, 'Entity-Destination(e2,e1)': 8,
               'Cause-Effect(e1,e2)': 9, 'Cause-Effect(e2,e1)': 10,
               'Component-Whole(e1,e2)': 11, 'Component-Whole(e2,e1)': 12,
               'Entity-Origin(e1,e2)': 13, 'Entity-Origin(e2,e1)': 14,
               'Member-Collection(e1,e2)': 15, 'Member-Collection(e2,e1)': 16,
               'Content-Container(e1,e2)': 17, 'Content-Container(e2,e1)': 18}

label2class = {0: 'Other',
               1: 'Message-Topic(e1,e2)', 2: 'Message-Topic(e2,e1)',
               3: 'Product-Producer(e1,e2)', 4: 'Product-Producer(e2,e1)',
               5: 'Instrument-Agency(e1,e2)', 6: 'Instrument-Agency(e2,e1)',
               7: 'Entity-Destination(e1,e2)', 8: 'Entity-Destination(e2,e1)',
               9: 'Cause-Effect(e1,e2)', 10: 'Cause-Effect(e2,e1)',
               11: 'Component-Whole(e1,e2)', 12: 'Component-Whole(e2,e1)',
               13: 'Entity-Origin(e1,e2)', 14: 'Entity-Origin(e2,e1)',
               15: 'Member-Collection(e1,e2)', 16: 'Member-Collection(e2,e1)',
               17: 'Content-Container(e1,e2)', 18: 'Content-Container(e2,e1)'}


def handleposition(entity, sentence_length):
    res = [_pos(i - entity) for i in range(sentence_length)]
    return res


def handlelabel(y):
    return class2label.get(y, 0)


def _pos(x):
    '''
    map the relative distance between [0, 123)
    '''
    if x < -49:
        return 0
    if 49 >= x >= -49:
        return x + 50
    if x > 49:
        return 0


def process_question(question):
    question = question.lower()
    question = question.replace("'", " '")
    question = question.replace(",", " ,")
    question = question.replace(".", " .")
    question = question.split(' ')
    e1_begin = e1_end = e2_begin = e2_end = 0
    for i, item in enumerate(question):
        if item.startswith('<e1>'):
            e1_begin = i
        if item.endswith('</e1>'):
            e1_end = i
        if item.startswith('<e2>'):
            e2_begin = i
        if item.endswith('</e2>'):
            e2_end = i

    def remove_tag(x):
        x = x.replace('<e1>', '')
        x = x.replace('</e1>', '')
        x = x.replace('<e2>', '')
        x = x.replace('</e2>', '')
        return x

    question = list(map(remove_tag, question))
    return question, e1_begin, e1_end, e2_begin, e2_end


def process_file(in_filename, out_filename):
    max_len = 0
    max_distance = 0
    with open(in_filename, 'r') as f:
        lines = f.readlines()
    new_lines = []
    for i in range(0, len(lines), 4):
        relation = lines[i + 1].strip()
        question = lines[i].strip().split('\t')[1][1:-1]
        question, e1_begin, e1_end, e2_begin, e2_end = process_question(question)
        max_len = max(max_len, len(question))
        max_distance = max(max_distance, e1_end)
        max_distance = max(max_distance, len(question) - e1_end)
        max_distance = max(max_distance, e2_end)
        max_distance = max(max_distance, len(question) - e2_end)

        new_lines.append({'sentence': ' '.join(question[:96]),
                          'label': class2label.get(relation, 0),
                          "e1": ' '.join([str(_pos(i - e1_begin)) for i in range(len(question))]),
                          'e1_begin': e1_begin,
                          'e2': ' '.join([str(_pos(i - e2_begin)) for i in range(len(question))]),
                          'e2_begin': e2_begin})

    with open(out_filename, 'w') as f:
        for dic in new_lines:
            f.writelines(json.dumps(dic) + '\n')

    print("Max length: {}".format(max_len))
    print("Max distance: {}".format(max_distance))


if __name__ == '__main__':
    train_in_file = "../data/TRAIN_FILE.TXT"
    train_out_file = "../data/train.txt"
    test_in_file = '../data/TEST_FILE_FULL.TXT'
    test_out_file = '../data/test.txt'

    process_file(train_in_file, train_out_file)
    process_file(test_in_file, test_out_file)
