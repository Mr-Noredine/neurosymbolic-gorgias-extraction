"""Quelles règles de référence manquent, et à quoi ressemble leur surface ?"""
import sys, glob, os
BASE = '/home/noureddine/neurosymbolic-gorgias-extraction'
sys.path.insert(0, os.environ.get('GORGIAS_SRC', BASE + '/src'))
sys.path.insert(0, BASE + '/bench')
os.chdir(BASE)
from gorgias import app
import scorer
app._ask_stage = lambda *a, **k: (_ for _ in ()).throw(app.StageError("muet"))
corpus = sys.argv[1] if len(sys.argv) > 1 else 'data/cas'
manques, faux = [], []
for txt in sorted(glob.glob(corpus + '/*.txt')):
    texte = open(txt).read()
    try:
        sortie = app.annotate(texte, model=object(), output_format='brat')
    except Exception:
        sortie = ''
    ann = txt[:-4] + '.ann'
    if not os.path.exists(ann): continue
    reference = open(ann).read()
    ref, pro = scorer.analyser(reference), scorer.analyser(sortie)
    liaison = scorer.apparier_entites(ref, pro)
    sref = scorer._signatures(ref, lambda t: t)
    spro = list(scorer._signatures(pro, lambda t: liaison.get(t)).values())
    def montre(t):
        ty, d, f = ref.entites[t]; return texte[d:f]
    for s in sref.values():
        if s[0] != 'rule': continue
        if s in spro: continue
        produit_effet = any(p[0]=='rule' and p[1]==s[1] for p in spro)
        manques.append((os.path.basename(txt)[:-4],
                        'CONDITION' if produit_effet else 'ABSENTE',
                        montre(s[1]), [montre(c) for c in s[2]]))
print(f"=== {len(manques)} règle(s) de référence manquée(s) ===")
for m in manques:
    print(f"  [{m[1]}] {m[0][:26]:<26} « {m[2][:46]} »  <=  {[c[:40] for c in m[3]]}")
