from g2p_en import G2p
import re
import numpy as np
from joblib import Memory
memory = Memory(location='/users/u220110626/.cache', verbose=0, mmap_mode='r')

g2p = G2p()
phonemes = g2p.phonemes
phonemes = [re.sub(r'\d+$', '', p) for p in phonemes]
phonemes = list(dict.fromkeys(phonemes))
phonemes = sorted(phonemes)[4:]
print(f"Use {len(phonemes)} phonemes {phonemes}")

# (39,)
@memory.cache()
def getMatrix(word : str):
    result = np.zeros(39)
    for phoneme in g2p(word):
        # phoneme is already a string, clean it by removing digits
        clean_phoneme = re.sub(r'\d+$', '', phoneme)
        if clean_phoneme in phonemes:
            result[phonemes.index(clean_phoneme)] = 1.0
    return result

print(getMatrix("word"))