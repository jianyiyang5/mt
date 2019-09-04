import random
from sklearn.model_selection import train_test_split
import numpy

lines = open('data/%s-%s.bpe' % ('eng', 'fra'), encoding='utf-8').\
        read().strip().split('\n')
data = numpy.array(lines)
x_train, x_test = train_test_split(data, test_size=0.1)

with open("data/eng-fra.txt", "w", encoding='utf-8') as f:
    f.writelines('\n'.join(x_train))

f_en = open("test/test.en", "w", encoding='utf-8')
f_fr = open("test/test.fr", "w", encoding='utf-8')
f_all = open('test/%s-%s.txt' % ('eng', 'fra'), 'w', encoding='utf-8')
for row in x_test:
    cols = row.split('\t')
    f_en.write(cols[0]+'\n')
    f_fr.write(cols[1]+'\n')
    f_all.write(row+'\n')

f_en.close()
f_fr.close()
f_all.close()