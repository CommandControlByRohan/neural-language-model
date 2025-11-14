# inspect_data.py 
import json, torch, collections, re
from pathlib import Path
p = Path(".")
# 1) sizes of tensors
for name in ("train.pt","val.pt","test.pt"):
    if (p/name).exists():
        t = torch.load(p/name)
        print(name, {k: getattr(v,'shape',None) for k,v in t.items()})
    else:
        print("Missing", name)

# 2) load vocab (word2id) and invert
vocab = json.loads((p/"vocab.json").read_text(encoding="utf8"))
w2i = vocab.get("word2id", vocab.get("token2id", {}))
i2w = {int(i):w for w,i in w2i.items()} if all(isinstance(k,str) for k in w2i.keys()) else {v:k for k,v in w2i.items()}

# 3) sample decoded sequence from train.pt (first batch)
train = torch.load("train.pt")
x = train["x"][:1]  # first example (seq_len)
print("\nSample encoded x shape:", x.shape)
seq = x[0].tolist()
decoded = " ".join(i2w.get(int(tok), "<UNK>") for tok in seq[:60])
print("\nDecoded sample (first 60 tokens):\n", decoded)

# 4) top 50 words from raw text (safe)
if (p/"Pride_and_Prejudice-Jane_Austen.txt").exists():
    txt = (p/"Pride_and_Prejudice-Jane_Austen.txt").read_text(encoding="utf8").lower()
    tokens = re.findall(r"[a-z']+", txt)
    ctr = collections.Counter(tokens)
    print("\nTop 50 words (raw counts):")
    for w,c in ctr.most_common(50):
        print(f"{w:>12} {c}")
else:
    print("Raw text not found for frequency counts")
