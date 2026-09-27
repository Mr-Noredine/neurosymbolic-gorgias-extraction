"""Les règles PRODUITES qui ne s'apparient à aucune règle de référence."""
import sys, glob, os
BASE = '/home/noureddine/neurosymbolic-gorgias-extraction'
sys.path.insert(0, os.environ.get('GORGIAS_SRC', BASE + '/src'))
sys.path.insert(0, BASE + '/bench')
os.chdir(BASE)
from gorgias import app
import scorer
app._ask_stage = lambda *a, **k: (_ for _ in ()).throw(app.StageError("muet"))
corpus = sys.argv[1]
faux = []
for txt in sorted(glob.glob(corpus + '/*.txt')):
    texte = open(txt).read()
    try:
        sortie = app.annotate(texte, model=object(), output_format='brat')
    except Exception:
        sortie = ''
    ann = txt[:-4] + '.ann'
    if not os.path.exists(ann): continue
    ref, pro = scorer.analyser(open(ann).read()), scorer.analyser(sortie)
    liaison = scorer.apparier_entites(ref, pro)
    sref = list(scorer._signatures(ref, lambda t: t).values())
    spro = scorer._signatures(pro, lambda t: liaison.get(t))
    for ident, s in spro.items():
        if s[0] != 'rule' or s in sref: continue
        genre, effet, conds = pro.evenements[ident][0], None, []
        roles = pro.evenements[ident][1]
        def m(t):
            ty, d, f = pro.entites[t]
            return f"{'✓' if t in liaison else '✗'}{texte[d:f][:40]!r}"
        faux.append((os.path.basename(txt)[:-4],
                     m(roles['Effect'][0]),
                     [m(c) for c in roles.get('Condition', [])]))
print(f"=== {len(faux)} règle(s) produite(s) sans appariement ===")
for f in faux: print(f"  {f[0][:26]:<26} {f[1]}  <=  {f[2]}")
