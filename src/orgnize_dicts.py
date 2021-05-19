"""read"""
import os
fr = '../embeddings/filtered_dictionaries'
to = 'data/crosslingual/dictionaries'
langs = ['hin', 'kor', 'rus', 'fin', 'jpn']
langs2 = ['hi', 'ko', 'ru', 'fi', 'ja']
for l11, l12 in zip(langs, langs2):
    for l21, l22 in zip(langs, langs2):
        if l11 == l21: continue
        path = os.path.join(fr, '{}-{}.0-5000.txt'.format(l11, l21))
        dicts = []
        with open(path) as f:
            lines = f.readlines()
            for line in lines:
                try:
                    a = line.split()[0]
                    b = line.split()[1]
                    dicts.append('{} {}'.format(a, b))
                except:
                    pass
        path = os.path.join(to, '{}-{}.0-5000.txt'.format(l12, l22))
        with open(path, 'w') as f:
            f.write('\n'.join(dicts))

        path = os.path.join(fr, '{}-{}.5000-6500.txt'.format(l11, l21))
        dicts = []
        with open(path) as f:
            lines = f.readlines()
            for line in lines:
                try:
                    a = line.split()[0]
                    b = line.split()[1]
                    dicts.append('{} {}'.format(a, b))
                except:
                    pass
        path = os.path.join(to, '{}-{}.5000-6500.txt'.format(l12, l22))
        with open(path, 'w') as f:
            f.write('\n'.join(dicts))
