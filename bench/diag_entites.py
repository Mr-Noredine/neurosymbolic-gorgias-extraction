"""Les entités PRODUITES qui ne s'apparient à aucune entité de référence."""
import sys, glob, os, collections
BASE = '/home/noureddine/neurosymbolic-gorgias-extraction'
sys.path.insert(0, os.environ.get('GORGIAS_SRC', BASE + '/src'))
sys.path.insert(0, BASE + '/bench')
os.chdir(BASE)
from gorgias import app
import scorer
app._ask_stage = lambda *a, **k: (_ for _ in ()).throw(app.StageError("muet"))
faux = collections.Counter(); ou = collections.defaultdict(list)
for txt in sorted(glob.glob(sys.argv[1] + '/*.txt')):
    texte = open(txt).read()
    try:
        sortie = app.annotate(texte, model=object(), output_format='brat')
    except Exception:
        sortie = ''
    ann = txt[:-4] + '.ann'
    if not os.path.exists(ann): continue
    ref, pro = scorer.analyser(open(ann).read()), scorer.analyser(sortie)
    liaison = scorer.apparier_entites(ref, pro)
    for ident, (ty, d, f) in pro.entites.items():
        if ident in liaison: continue
        faux[(ty, texte[d:f][:44])] += 1
        ou[(ty, texte[d:f][:44])].append(os.path.basename(txt)[:-4])
print(f"=== {sum(faux.values())} entité(s) produite(s) sans appariement ===")
for (ty, s), n in faux.most_common(30):
    print(f"  {n:>3}  {ty:<8} {s!r}   ex. {ou[(ty,s)][0][:26]}")
