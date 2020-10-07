""" get dammy embeddings """
import io
import numpy as np
emb_path = "data/wiki.en.vec"
emb_path2 = "data/wiki.en_dammy.vec"
lines = []
with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
  for i, line in enumerate(f):
    if i == 0:
      split = line.split()
      assert len(split) == 2
      lines.append(line)
    else:
      word, vect = line.rstrip().split(' ', 1)
      vect = np.random.randn(300)
      new_line = [word]
      for j in range(300):
        new_line.append(str(vect[j]))
      new_line = ' '.join(new_line)
      lines.append(new_line)
    if i % 10000 == 0:
      print(i)

with io.open(emb_path2, 'w', encoding='utf-8') as f:
  f.writelines('\n'.join(lines))
