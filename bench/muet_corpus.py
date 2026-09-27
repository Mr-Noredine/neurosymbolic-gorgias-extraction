"""Le pipeline avec un modèle MUET, sur un corpus entier : plancher symbolique."""
import sys, glob, os
BASE = '/home/noureddine/neurosymbolic-gorgias-extraction'
sys.path.insert(0, os.environ.get('GORGIAS_SRC', BASE + '/src'))
sys.path.insert(0, BASE + '/bench')
os.chdir(BASE)
from gorgias import app
import scorer
app._ask_stage = lambda *a, **k: (_ for _ in ()).throw(app.StageError("muet"))
corpus = sys.argv[1] if len(sys.argv) > 1 else 'data/cas'
scores, vides, pollues = [], 0, 0
for txt in sorted(glob.glob(corpus + '/*.txt')):
    texte = open(txt).read()
    try:
        sortie = app.annotate(texte, model=object(), output_format='brat')
    except Exception:
        sortie = ''
    ann = txt[:-4] + '.ann'
    if not sortie.strip():
        vides += 1
    if 'narratif' in corpus:
        if sortie.strip():
            pollues += 1
            print("  POLLUÉ", os.path.basename(txt))
        continue
    scores.append(scorer.scorer_document(
        open(ann).read() if os.path.exists(ann) else '', sortie))
print(f"\n=== {corpus} : plancher symbolique ({len(scores) or pollues} documents, "
      f"{vides} sortie(s) vide(s)) ===")
if 'narratif' in corpus:
    print(f"  récits pollués : {pollues}")
else:
    t = scorer.agreger(scores)
    for cle in ('entites','Context','Option','Marker','rule','prefer','meta_prefer'):
        m = t[cle]
        print(f"  {cle:<13} P {m['precision']*100:5.1f}%  R {m['rappel']*100:5.1f}%"
              f"   {m['justes']:>3}/{m['produits']:>3} produits, {m['attendus']:>3} attendus")
