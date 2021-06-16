"""
Prepare crosslingual dictionaries of language pairs not including English.
"""
import os
fr = '../embeddings/filtered_dictionaries'
fr2 = '../embeddings/bridged_dictionaries'
to = 'data/crosslingual/dictionaries'
# https://ja.wikipedia.org/wiki/ISO_639-1%E3%82%B3%E3%83%BC%E3%83%89%E4%B8%80%E8%A6%A7
# ISO639-2/T
langs = ['hin', 'kor', 'rus', 'fin', 'jpn', 'zho', 'tha', 'swe', 'nld', 'ara', 'ind', 'tur']
# ISO639-1
langs2 = ['hi', 'ko', 'ru', 'fi', 'ja', 'zh', 'th', 'sv', 'nl', 'ar', 'id', 'tr']
for l11, l12 in zip(langs, langs2):
    for l21, l22 in zip(langs, langs2):
        if l11 == l21: continue
        try:
            path = os.path.join(fr, '{}-{}.0-5000.txt'.format(l11, l21))
            dicts = []
            with open(path) as f:
                lines = f.readlines()
                for line in lines:
                    try:
                        a = line.split()[0]
                        b = line.split()[1]
                        if ';' in a or ';' in b: continue
                        dicts.append('{} {}'.format(a, b))
                    except:
                        pass
        except:
            path = os.path.join(fr2, '{}-{}.0-5000.txt'.format(l11, l21))
            dicts = []
            with open(path) as f:
                lines = f.readlines()
                for line in lines:
                    try:
                        a = line.split()[0]
                        b = line.split()[1]
                        if ';' in a or ';' in b: continue
                        dicts.append('{} {}'.format(a, b))
                    except:
                        pass
        path = os.path.join(to, '{}-{}.0-5000.txt'.format(l12, l22))
        with open(path, 'w') as f:
            f.write('\n'.join(dicts))
        try:
            path = os.path.join(fr, '{}-{}.5000-6500.txt'.format(l11, l21))
            dicts = []
            with open(path) as f:
                lines = f.readlines()
                for line in lines:
                    try:
                        a = line.split()[0]
                        b = line.split()[1]
                        if ';' in a or ';' in b: continue
                        dicts.append('{} {}'.format(a, b))
                    except:
                        pass
        except:
            path = os.path.join(fr2, '{}-{}.5000-6500.txt'.format(l11, l21))
            dicts = []
            with open(path) as f:
                lines = f.readlines()
                for line in lines:
                    try:
                        a = line.split()[0]
                        b = line.split()[1]
                        if ';' in a or ';' in b: continue
                        dicts.append('{} {}'.format(a, b))
                    except:
                        pass
        path = os.path.join(to, '{}-{}.5000-6500.txt'.format(l12, l22))
        with open(path, 'w') as f:
            f.write('\n'.join(dicts))
