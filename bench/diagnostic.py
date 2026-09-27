"""Pourquoi une priorité de référence n'est-elle pas produite ?"""
import sys, glob, os
BASE = '/home/noureddine/neurosymbolic-gorgias-extraction'
sys.path.insert(0, os.environ.get('GORGIAS_SRC', BASE + '/src'))
sys.path.insert(0, BASE + '/bench')
os.chdir(BASE)
from gorgias import app, priorites
import scorer
MUET = len(sys.argv) < 3 or sys.argv[2] != 'avec-modele'
if MUET:
    app._ask_stage = lambda *a, **k: (_ for _ in ()).throw(app.StageError("muet"))
corpus = sys.argv[1] if len(sys.argv) > 1 else 'data/cas'
for txt in sorted(glob.glob(corpus + '/*.txt')):
    texte = open(txt).read()
    try:
        sortie = app.annotate(texte, model=object(), output_format='brat')
    except Exception:
        sortie = ''
    ann = txt[:-4] + '.ann'
    if not os.path.exists(ann):
        continue
    reference = open(ann).read()
    sc = scorer.scorer_document(reference, sortie)
    if sc['priorite']['justes'] == sc['priorite']['attendus'] and \
       sc['priorite']['produits'] == sc['priorite']['attendus']:
        continue
    ref, pro = scorer.analyser(reference), scorer.analyser(sortie)
    liaison = scorer.apparier_entites(ref, pro)
    inverse = {v: k for k, v in liaison.items()}
    sref = scorer._signatures(ref, lambda t: t)
    spro = scorer._signatures(pro, lambda t: liaison.get(t))
    prio_ref = [s for s in sref.values() if s[0] != 'rule']
    prio_pro = [s for s in spro.values() if s[0] != 'rule']
    manques = [s for s in prio_ref if s not in prio_pro]
    faux = [s for s in prio_pro if s not in prio_ref]
    print(f"\n=== {os.path.basename(txt)[:-4]} "
          f"(prio {sc['priorite']['justes']}/{sc['priorite']['produits']} "
          f"produites, {sc['priorite']['attendus']} attendues) ===")
    def montre(t):
        if t is None: return '∅'
        if t in ref.entites:
            ty, d, f = ref.entites[t]; return f'{t}:{texte[d:f][:34]!r}'
        return str(t)
    for s in manques:
        cause = []
        if s[0] in ('prefer', 'meta_prefer'):
            for role, val in (('W', s[1]), ('L', s[2])):
                if val is None: cause.append(f'{role} irrésolu')
            for c in s[3]:
                if c not in inverse: cause.append(f'When {montre(c)} non produit')
        print(f"  MANQUE {s[0]:<12} W={s[1]} L={s[2]} When={[montre(c) for c in s[3]]}"
              f"  {'; '.join(cause)}")
    for s in faux:
        print(f"  FAUX   {s[0]:<12} W={s[1]} L={s[2]} When={s[3]}")
