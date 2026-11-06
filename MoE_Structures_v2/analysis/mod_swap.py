"""Held-out mod pairs split by whether the swapped pair (b, a) is in the training half."""
import json, sys, torch
sys.path.insert(0, ".")
from backbone.model import EncoderModel
from backbone.data import Dataset, load_vocab
from data_gen.vocab import ANS, VOCAB
from data_gen.tasks import mod_split, SYM0
torch.set_num_threads(1)
split = mod_split()
res = {}
for run in sys.argv[1:]:
    cfg = json.load(open(f"runs/{run}/config.json"))
    m = EncoderModel(cfg, len(load_vocab())); m.load_state_dict(torch.load(f"runs/{run}/model.pt", map_location="cpu")); m.eval()
    c = {}
    for tok, tgt, pad in Dataset("mod", "test").eval_batches(500, 4):
        sel = tok == ANS
        with torch.no_grad():
            lg, _ = m(tok, pad, select=sel)
        ok = (lg.argmax(-1) == tgt[sel]).tolist()
        bi, ti = sel.nonzero(as_tuple=True)
        for j in range(len(ok)):
            a, o, b = int(tok[bi[j], ti[j] - 4]) - SYM0, VOCAB[int(tok[bi[j], ti[j] - 3])], int(tok[bi[j], ti[j] - 2]) - SYM0
            key = (o, "swap in train" if split[o][b, a] else "swap not in train")
            c.setdefault(key, [0, 0]); c[key][0] += ok[j]; c[key][1] += 1
    res[run] = {f"{o} | {k}": (round(x / n, 3), n) for (o, k), (x, n) in sorted(c.items())}
    print(run); [print("   ", k, v) for k, v in res[run].items()]
json.dump(res, open("/tmp/claude-0/-home-claude/e857238f-5029-57e7-91a7-11f63f106500/scratchpad/mod_swap.json", "w"), indent=1)
