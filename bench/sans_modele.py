"""Exécute le pipeline avec un modèle MUET : seul le chemin symbolique agit."""
import sys, glob, os
sys.path.insert(0, '/home/noureddine/neurosymbolic-gorgias-extraction/src')
sys.path.insert(0, '/home/noureddine/neurosymbolic-gorgias-extraction/bench')
os.chdir('/home/noureddine/neurosymbolic-gorgias-extraction')
from gorgias import app
import scorer


class Muet:
    def invoke(self, messages):
        raise RuntimeError("aucun modèle")


app._ask_stage.__wrapped__ = None
_vrai = app._ask_stage
def _muet(*a, **k):
    raise app.StageError("modèle muet")
app._ask_stage = _muet

cibles = sys.argv[1:] or ['09-raffinement-explicite', '33-deneigement',
                          '34-hebergement', '35-semis', '01-courses-3-niveaux',
                          '03-pret-objet', '10-scenarios-freres',
                          '36-eclairage', '37-arbitrage-sportif']
scores = []
for nom in cibles:
    txt = f'data/cas/{nom}.txt'
    texte = open(txt).read()
    try:
        sortie = app.annotate(texte, model=Muet(), output_format='brat')
    except Exception as e:
        sortie = ''
        print(f"  ({type(e).__name__}: {e})")
    reference = open(f'data/cas/{nom}.ann').read()
    sc = scorer.scorer_document(reference, sortie)
    scores.append(sc)
    print(f"=== {nom} ===")
    for cle in ('rule', 'prefer', 'meta_prefer'):
        d = sc[cle]
        print(f"    {cle:<12} {d['justes']}/{d['produits']} produits, "
              f"{d['attendus']} attendus")
    print(sortie or '    (vide)')
print("\n=== TOTAL ===")
t = scorer.agreger(scores)
for cle in ('entites', 'Context', 'Option', 'Marker', 'rule', 'prefer',
            'meta_prefer'):
    m = t[cle]
    print(f"  {cle:<14} P {m['precision']*100:5.1f}%  R {m['rappel']*100:5.1f}%"
          f"   {m['justes']}/{m['produits']} produits, {m['attendus']} attendus")
