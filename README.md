# Extraction neuro-symbolique GORGIAS

[![licence MIT](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![tests](https://img.shields.io/badge/tests-138%20passent-brightgreen.svg)](tests/)

**Transforme un texte argumentatif français en programme logique.** Vous lui
donnez un paragraphe ; il vous rend les règles qu'il contient, les préférences
entre ces règles, et les contextes qui les déclenchent — au formalisme
**LPP/GORGIAS**, prêt à être raisonné par un solveur.

Tout tourne **en local, sur votre machine**. Aucune clé d'API, aucun appel
distant.

---

## D'où vient ce projet

Il prolonge un **Travail d'Étude et de Recherche de Master 1**, mené en binôme
à l'Université Paris Cité dans le cadre du projet ANR **GRAIL** :

> **[thmsgo18/lpp-argument-mining](https://github.com/thmsgo18/lpp-argument-mining)**
> — annoter des textes argumentatifs en structures LPP/GORGIAS, en pilotant un
> grand modèle de langue par *prompt engineering*.

Ce TER a montré que la chose était faisable, et où elle butait : le prompt
donnait de bons résultats sur du texte juridique formel, mais demandait
beaucoup de reprise humaine sur du dialogue délibératif. Autrement dit, la
qualité dépendait entièrement de ce que le modèle voulait bien faire ce
jour-là, et rien dans le système ne distinguait ce qu'il *savait* de ce qu'il
*devinait*.

**Ce dépôt part de la même idée et prend une autre direction.** Plutôt que de
mieux demander au modèle, on lui retire du travail : tout ce que la grammaire
française permet de trancher exactement est décidé en Python, et le modèle ne
répond plus qu'à des questions fermées sur ce qui reste. Il n'écrit jamais un
identifiant, un niveau ni un rôle.

Ce n'est pas la suite scientifique du TER — c'est une reprise personnelle,
avec une architecture différente, un banc de mesure doté de contre-exemples,
et la règle de ne rien implémenter avant de l'avoir mesuré.

---

## Un exemple complet

**L'entrée** — trois phrases (`data/cas/08-priorite-simple.txt`) :

> Quand une commande dépasse mille euros, le service applique une remise.
> Quand le client est nouveau, le service demande un paiement d'avance.
> Pour un client stratégique, la remise l'emporte sur le paiement d'avance.

**La sortie** — `./run 08-priorite-simple.txt` :

```prolog
% Composants LPP/Gorgias normalisés
% Les libellés conservent le texte source exact.
scenario(c1, depasser(commande, euro),  "Quand une commande dépasse mille euros").
option(o1,   appliquer(service, remise), "le service applique une remise").
scenario(c2, client_nouveau,             "Quand le client est nouveau").
option(o2,   demander(service, paiement),"le service demande un paiement d'avance").
scenario(c3, client_strategique,         "Pour un client stratégique").

complement(o1, o2).          % les deux options sont incompatibles
rule(r1, [c1], o1).          % commande > 1000 €      -> remise
rule(r2, [c2], o2).          % client nouveau         -> paiement d'avance
prefer(p1, [c3], r1, r2).    % si client stratégique  -> r1 gagne contre r2
```

Le système a trouvé seul que les deux conclusions s'excluent, que la troisième
phrase n'est pas une règle mais un **arbitrage entre deux règles**, et dans
quel contexte cet arbitrage s'applique.

Avec `--brat`, la même analyse sort en annotation [brat](https://brat.nlplab.org/),
ouvrable dans un outil d'annotation :

```
T1	Context 6 38	une commande dépasse mille euros
T2	Option 40 70	le service applique une remise
T6	Marker 180 193	l'emporte sur
E1	rule:T2 Condition:T1 Effect:T2
E3	prefer:T6 Winner:E1 Loser:E2 When:T5
```

---

## Sommaire

- [Installation](#installation)
- [Utilisation](#utilisation)
- [La contrainte qui commande tout](#la-contrainte-qui-commande-tout)
- [Comment ça marche](#comment-ça-marche)
- [Ce qui est extrait](#ce-qui-est-extrait)
- [Le banc de mesure](#le-banc-de-mesure)
- [Résultats](#résultats)
- [Généralisation : ce que le banc ne dit pas](#généralisation--ce-que-le-banc-ne-dit-pas)
- [Limites connues](#limites-connues)
- [Organisation du dépôt](#organisation-du-dépôt)

---

## Installation

### Ce qu'il faut prévoir

| | |
|---|---|
| **Python** | 3.10 ou plus — développé et mesuré sous 3.12 |
| **Disque** | ~7 Go : 5,2 Go pour le modèle de langue, 1,1 Go pour l'environnement Python dont 613 Mo de modèle spaCy |
| **RAM** | 8 Go minimum, 12 Go confortable — le modèle 8B en occupe ~6 |
| **Système** | Linux ou macOS. Mesuré sur un Ryzen 5 PRO 4650U, 6 cœurs, sans GPU |

### 1. Ollama — le moteur qui exécute le modèle en local

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &            # à laisser tourner
ollama pull qwen3:8b      # 5,2 Go
```

> **Debian / Ubuntu.** L'installeur a besoin de `zstd` pour décompresser son
> archive, et **échoue en rendant malgré tout un code de sortie nul** — donc
> sans que rien ne le signale. Installez-le d'abord :
> `sudo apt-get install -y zstd`

`qwen3:8b` est le modèle de toutes les mesures de référence. `qwen3:1.7b`
fonctionne mais est **disqualifié** : il invente une structure argumentative
sur 5 récits sur 32 qui n'en contiennent aucune.

### 2. Le paquet Python

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Le projet suit la disposition `src/` : `import gorgias` échoue tant que le
paquet n'est pas installé. C'est voulu — cela évite qu'un import réussisse par
accident depuis la racine du dépôt et masque une erreur d'empaquetage.

Toutes les dépendances sont déclarées dans `pyproject.toml`, seule source. Il
n'y a volontairement pas de `requirements.txt` : deux listes finissent toujours
par diverger.

### 3. Le modèle spaCy

Il ne s'installe pas par `pip`, c'est un téléchargement séparé :

```bash
python -m spacy download fr_core_news_lg      # 613 Mo
```

> **Ne sautez pas cette étape.** Sans elle, le court-circuit syntaxique et les
> onze mécanismes déterministes se désactivent — proprement, mais vous
> n'utiliseriez plus le même système. Sur une machine contrainte,
> `fr_core_news_sm` (26 Mo) est un repli : 59 relations prouvées sur 116 au
> lieu de 68. Il faut alors changer `MODELE_DEFAUT` dans
> `src/gorgias/syntaxe.py`.

### 4. Vérifier

```bash
python -m pytest tests/       # 138 tests, sans Ollama, ~12 s
```

Ces tests n'appellent aucun modèle de langue : s'ils passent, le code et le
corpus de référence sont sains. Ils ne disent rien de votre installation
d'Ollama — pour ça, annotez un texte.

### En cas de panne

| symptôme | cause |
|---|---|
| `environnement absent` | le `.venv` n'existe pas — reprendre à l'étape 2 |
| `le serveur Ollama ne répond pas` | lancer `ollama serve` |
| `modèle « qwen3:8b » absent` | lancer `ollama pull qwen3:8b` |
| `This version requires zstd` | `sudo apt-get install -y zstd`, puis réinstaller Ollama |
| l'annotation est très lente | vérifier la RAM libre : sous 8 Go le modèle part en swap |
| `ModuleNotFoundError: gorgias` | `pip install -e .` n'a pas été fait, ou le `.venv` n'est pas activé |

Si vous **déplacez ou renommez le dossier**, le `.venv` casse : ses scripts
contiennent le chemin absolu. Le plus simple est de le recréer.

---

## Utilisation

### En ligne de commande

`./run` vérifie l'environnement — venv, Ollama, modèle présent — **avant** de
lancer quoi que ce soit, et vous dit quelle commande passer si quelque chose
manque. Une annotation qui échoue au bout de dix minutes coûte plus cher que
le contrôle.

```bash
./run mon_texte.txt                # -> mon_texte.lpp  (programme logique)
./run mon_texte.txt --brat         # -> mon_texte.ann  (annotation brat)
./run mon_texte.txt --modele X     # avec un autre modèle Ollama
```

L'appel direct fonctionne aussi, avec toutes les options :

```bash
python3 -m gorgias.app FICHIER.txt --format lpp --modele qwen3:8b --etage hybride
```

> **La sortie est écrite à côté de l'entrée.** Lancé sur un corpus déjà annoté,
> le programme en écraserait les annotations de référence sans rien signaler —
> après quoi les documents scoreraient parfaitement contre eux-mêmes. Travaillez
> toujours sur une copie.

### Sur un corpus entier

Le mode batch sépare les sorties du corpus source et écrit un `manifest.json`
auditable : statut par document, durée, nombre d'appels au modèle, empreintes
SHA-256 des entrées et des sorties.

```bash
gorgias-batch ./documents ./sorties --format lpp
```

### En service HTTP

```bash
export GORGIAS_OLLAMA_URL=http://serveur-gpu:11434
export GORGIAS_API_KEY='à fournir par votre gestionnaire de secrets'
pip install -e ".[server]"
gorgias-api --host 0.0.0.0 --port 8000
```

Trois points : `/healthz` pour les sondes de vivacité, `/readyz` qui vérifie
qu'Ollama répond, et `/v1/extractions` pour l'extraction. Le `Dockerfile`
fourni exécute le service **sans privilèges** (UID 10001).

Le service n'envoie le texte qu'au serveur Ollama configuré, n'intègre ni
télémétrie ni stockage applicatif, et ne journalise pas les corps de requête.
Un Ollama distant reste néanmoins une sortie de données : c'est à l'opérateur
d'en contrôler l'hébergement et le chiffrement du transport.

### Variables d'environnement

| | |
|---|---|
| `GORGIAS_OLLAMA_URL` | adresse du serveur Ollama (défaut : local) |
| `GORGIAS_MODEL` | modèle par défaut |
| `GORGIAS_TIMEOUT` | délai maximal d'un appel, en secondes |
| `GORGIAS_NUM_PREDICT` | longueur maximale d'une réponse |
| `GORGIAS_API_KEY` | clé attendue par l'API HTTP ; sans elle, aucune authentification |
| `GORGIAS_MAX_TEXT_CHARS` | taille maximale acceptée par l'API |
| `GORGIAS_MAX_CONCURRENCY` | extractions simultanées |

Le passage CPU → GPU peut modifier les sorties : rejouez votre validation sur
la configuration exacte de production avant de promouvoir une image.

---

## La contrainte qui commande tout

**Sortie entièrement automatique, sans relecture humaine.** Précision proche de
100 %, rappel partiel accepté : *une annotation vide vaut mieux qu'une
annotation fausse.*

Ce choix renverse l'optimisation habituelle. Avec un humain dans la boucle, on
maximise le rappel — l'expert écartera les fausses règles. Sans lui, chaque
règle erronée entre telle quelle dans le programme logique et fausse tout ce
qui en découle : les conclusions dérivées, les priorités qui s'y appuient, et
le raisonnement final.

Toutes les décisions de ce projet se lisent à travers cet arbitrage, y compris
celles qui paraissent contre-intuitives — comme refuser un mécanisme qui
améliore le F1.

---

## Comment ça marche

Le modèle de langue ne décide que ce qu'il est le seul à pouvoir décider. Tout
le reste est dérivé en Python, et une part croissante du travail lui est
retirée parce qu'elle se tranche **grammaticalement**.

```
texte
  ├─ 0   découpage en propositions                     Python, sans modèle
  ├─ 1   proposition ? marqueur ?                      classification
  ├─ 2a  candidats de relations                        classification
  ├─ 2b  court-circuit syntaxique des subordonnées     spaCy, sans modèle
  ├─ 2c  filtres déterministes                         Python, sans modèle
  ├─ 2d  le modèle sur les seuls doutes restants       classification
  ├─ 3   priorités scénarisées                         classification
  └─ 4   rôles, scénarios cumulés, compilation         Python, sans modèle
.ann ou .lpp
```

### Le court-circuit syntaxique

C'est l'exemple central. Dans « Quand le sol est gelé, le sel est appliqué », la
subordonnée porte sa direction dans sa propre grammaire : la relation de
dépendance `advcl` accompagnée d'un `mark` désigne la condition, sans aucune
probabilité, sans aucun appel au modèle.

Mesuré sur les 116 relations de référence : **59 % de couverture, zéro erreur
de direction.** Ces paires ne passent pas par le modèle du tout.

### Onze mécanismes déterministes

Ils corrigent ensuite ce que le modèle assemble mal — et c'est de là qu'est
venu l'essentiel des gains. Les six principaux :

| | |
|---|---|
| **frontière de paragraphe** | une prémisse et sa conclusion ne se répartissent pas de part et d'autre d'une ligne vide |
| **frontière de locuteur** | la conclusion d'un interlocuteur n'est pas la prémisse d'un autre |
| **pureté des conclusions prouvées** | si la grammaire donne déjà la condition, une prémisse supplémentaire est parasite |
| **conjonction de conditions** | « Comme A **et que** B » énonce deux conditions d'une seule règle, pas deux règles |
| **conclusion impossible** | « Puisque X » ou « même en hiver » ne peut pas être un effet |
| **condition impossible** | « sauf lorsque X » ne conditionne pas la règle, il la défait |

### Le filtre de singularité

Il protège les récits. Une paire de propositions sans connecteur, dont les deux
verbes sont au passé, relate un enchaînement d'événements révolus — pas une
règle générale. C'est ce qui permet au système de rester muet sur un texte
narratif au lieu d'y inventer une argumentation.

---

## Ce qui est extrait

Le formalisme vient de Kakas, Moraïtis & Spanoudakis. Un texte qui raisonne
décrit un **problème de décision** :

| entité | |
|---|---|
| `Option` | ce que le raisonnement conclut — une décision, un verdict, une position |
| `Context` | ce dont cette conclusion dépend — faits, conditions, circonstances |
| `Marker` | l'amorce d'une priorité (« prime sur », « l'emporte sur ») |

| relation | |
|---|---|
| `rule` | des conditions mènent à une conclusion |
| `prefer` | dans un scénario donné, une règle bat une autre |
| `meta_prefer` | dans un scénario plus spécifique, une priorité en renverse une autre |

Le niveau d'une priorité est la **profondeur de raffinement du scénario**, et
il n'est pas borné : chaque fois qu'un texte ajoute de l'information à une
situation et change ainsi l'option retenue, il exprime une priorité d'un niveau
au-dessus.

Le module `lpp_asp.py` traduit le résultat en **Answer Set Programming** et le
résout avec [clingo](https://potassco.org/clingo/), ce qui permet de comparer
deux extractions sur ce qu'elles *concluent* et pas seulement sur leur forme.

---

## Le banc de mesure

Le dossier `data/` contient les références qui ont servi à mesurer chaque
décision du projet. Elles sont annotées à la main, et leur structure attendue
ne dépend du jugement d'aucun modèle.

| | | |
|---|---:|---|
| `data/cas/` | 48 | textes argumentatifs annotés. Les trois premiers reprennent la spécification formelle des articles de Kakas, Moraïtis & Spanoudakis — la référence la plus solide disponible, puisque la structure est celle des concepteurs du formalisme |
| `data/narratif/` | 32 | **récits négatifs** : des textes sans aucune structure argumentative, où toute annotation produite est un faux positif |
| `data/horschantillon*/` | 40 | quatre lots de dix paragraphes du **RGPD**, pris verbatim sur EUR-Lex, sélectionnés par une règle mécanique et gelés avant annotation |
| `data/synthetique/` | 140 | documents générés, à structure connue par construction |
| `data/noyau/` | 2 | paires positives et négatives dérivées du banc |

Les **32 récits négatifs** sont le jeu le plus important. Sans eux, on mesure
uniquement ce qu'un système trouve, jamais ce qu'il invente — et un extracteur
qui hallucine une règle sur un fait divers est inutilisable en production.

> Le harnais de mesure qui produit les chiffres ci-dessous n'est pas inclus
> dans ce dépôt, qui se limite au système et à ses références.

---

## Résultats

`qwen3:8b`, sur CPU, mesures contrôlées : même machine, même modèle, seul le
code varie entre deux points de comparaison.

| | départ | **aujourd'hui** |
|---|---:|---:|
| **précision des règles** | 31 % | **96,2 %** |
| rappel des règles | 11 % | **84,3 %** |
| F1 structurel des règles | — | **89,9 %** |
| **F1 sémantique** (métrique principale) | — | **88,8 %** |
| précision / rappel des entités | — | **100 % / 85,5 %** |
| priorités justes | 0 / 46 | **24 / 46** |
| précision des priorités | — | **88,9 %** |
| **récits pollués** | 2 / 32 | **0 / 32** |
| appels au modèle (48 textes) | 338 | **93** |
| durée CPU (48 textes) | 4 640 s | **1 746 s** |

Deux chiffres à ne pas confondre. L'étanchéité aux récits (**0/32**) dit que
rien n'est inventé sur un texte qui n'attend rien. La précision (**96,2 %**)
dit que sur un texte argumentatif, quatre règles produites sur 106 sont encore
fausses.

Ces chiffres sont des **estimations exploratoires**, pas une certification : le
banc ne contient que 48 documents argumentatifs, 121 relations, et une seule
annotation de référence. Intervalles bootstrap à 95 % par document, sur 10 000
réplications : précision des règles **93,0–99,0 %**, rappel **76,4–92,5 %** ;
précision des priorités **70,8–100 %**, rappel **31,0–80,5 %**.

---

## Généralisation : ce que le banc ne dit pas

Les 48 textes ci-dessus ont **tous servi au réglage** : chaque mécanisme y a
été mesuré avant d'être écrit. Ils ne disent donc rien de ce que vaut le
système sur du texte qu'il n'a jamais vu.

Les quatre lots RGPD répondent à cette question. Chacun a été gelé puis annoté
**avant** toute exécution du système. Et un lot ne mesure la généralisation
**qu'une seule fois** : dès qu'un mécanisme en est tiré, ce qu'on y mesure
devient un réglage, pas une généralisation.

| précision des règles au premier contact | lot 1 | lot 2 | lot 3 | lot 4 | banc réglé |
|---|---:|---:|---:|---:|---:|
| | 54 % | 42 % | **21 %** | *scellé* | **96 %** |

**C'est la limite la plus importante du projet, et elle est affichée
volontairement.** Sur du texte jamais vu, la précision des règles n'a jamais
dépassé 54 %.

L'écart n'est pas du surajustement au sens statistique. À chaque lot, le
diagnostic a désigné des **propriétés de surface absentes du banc** — la
numérotation des articles, les incises, les listes internes — que rien dans les
48 cas n'aurait pu révéler, et dont la correction s'est révélée *inerte* sur le
banc : 509 segments avant, 509 après. Il reste que quiconque applique ce
système à un nouveau corpus doit s'attendre à ce régime-là, et prévoir sa
propre passe de diagnostic.

`data/horschantillon4/` (rangs 31 à 40) n'a **jamais été exécuté**. C'est le
seul jeu qui puisse encore mesurer une généralisation honnête, et il perdra
cette propriété à la seconde où un mécanisme en sera tiré.

---

## Limites connues

- **La généralisation est le point faible principal** — voir la section
  ci-dessus.
- **19 règles sur 121 restent manquées.** Elles se concentrent dans les listes
  imbriquées, les dialogues, et les liens réellement implicites.
- **Les priorités restent faibles** : 24 sur 46. Leur rappel est borné par le
  **carré** de celui des règles — une priorité n'existe que si ses deux règles
  ont été trouvées.
- **Le formalisme ne distingue pas règle stricte et règle révisable.** La
  révisabilité est portée uniquement par les préférences.
- **Ni coréférence, ni quantification** : « il », « cette décision » restent
  tels quels dans les littéraux produits.
- **Le banc est petit** — 48 textes, 32 récits, 121 relations. Un motif qui ne
  touche que 5 relations reste sous le seuil de bruit.
- **Références mono-annotateur**, complètes par construction mais sans
  vérification indépendante.
- **Textes longs et langue anglaise non validés.**

Le système est déployable, mais **pas certifié pour les données d'une
organisation inconnue**. Un passage en production sérieux demande un corpus
représentatif annoté à l'aveugle par au moins deux personnes, un lot de test
gelé, des seuils d'acceptation métier, et une mesure sur le matériel réellement
déployé.

---

## La méthode

C'est la partie du projet qui a le plus de valeur, et elle tient en trois
portes qu'un mécanisme candidat doit franchir — dans cet ordre, les deux
premières étant gratuites puisqu'elles ne demandent aucune inférence :

1. **Portée** — combien de relations de référence le motif touche. En dessous
   de cinq, le résultat reste sous le bruit.
2. **Pollution** — combien d'arêtes il produit sur les 32 récits négatifs. La
   réponse attendue est zéro.
3. **Précision marginale** — `ΔC/ΔP` sur le banc complet. Elle doit dépasser la
   précision courante du système, **96,2 %**, faute de quoi le mécanisme
   dégrade la précision quoi qu'il fasse au F1.

Un corollaire contre-intuitif justifie cette dernière porte : **un gain de
rappel peut faire monter le F1 tout en violant le cahier des charges.** Sept
mécanismes plausibles ont été mesurés puis **refusés** par ces portes, dont un
qui améliorait le F1.

Deux règles de mesure, acquises à la dure :

- **compter les règles produites, pas seulement les justes.** Un mécanisme qui
  ajoute 4 règles justes peut en produire 8 ;
- **l'inférence n'est pas reproductible.** Ne jamais conclure d'une exécution
  unique, et répéter des deux côtés de l'interrupteur pour trancher une
  attribution.

---

## Organisation du dépôt

```
src/gorgias/
  app.py          le pipeline complet, de la segmentation à la compilation
  syntaxe.py      court-circuit des subordonnées, filtres grammaticaux
  priorites.py    isolation des énoncés de classement
  singularite.py  rejet des paires relatant des événements singuliers
  predicats.py    forme logique normalisée, négation
  lpp_asp.py      traduction ASP et comparaison par modèles stables
  batch.py        traitement d'un corpus avec manifeste auditable
  service.py      API HTTP optionnelle (extra [server])

data/             le banc : cas argumentatifs, récits négatifs, lots RGPD,
                  documents synthétiques
tests/            138 tests, qui n'appellent aucun modèle de langue
run               le point d'entrée
Dockerfile        l'image du service HTTP, exécutée sans privilèges
```

## Licence

[MIT](LICENSE) — Noureddine Mohammedi.

Les textes du RGPD contenus dans `data/horschantillon*/` proviennent d'EUR-Lex
(CELEX 32016R0679) et restent soumis aux conditions de réutilisation de
l'Union européenne ; leur provenance exacte est tracée dans les fichiers
`SOURCE.json` et `SELECTION.json` de chaque lot.
