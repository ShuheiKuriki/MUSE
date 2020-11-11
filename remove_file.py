import glob, os
lis = glob.glob('dumped/learning/*-unsup/*.txt')
for path in lis:
    os.remove(path)