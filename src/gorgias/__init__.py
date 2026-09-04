"""Extraction neuro-symbolique de structures d'argumentation LPP/GORGIAS.

Aucun module n'appelle de modèle de langue à l'import : `app.annotate` est le
seul point qui en a besoin.

    app          pipeline complet, de la segmentation à la compilation
    syntaxe      court-circuit des subordonnées, filtres grammaticaux
    priorites    isolation des énoncés de classement
    singularite  rejet des paires relatant des événements singuliers
    predicats    forme logique normalisée, négation
    lpp_asp      traduction ASP et comparaison par modèles stables
    batch        traitement d'un corpus avec manifeste auditable

`service` s'y ajoute avec l'extra `[server]` : il importe FastAPI, absent de
l'installation par défaut, et n'est donc pas exposé ici.
"""
from __future__ import annotations

__version__ = "0.2.0"

__all__ = [
    "app", "batch", "lpp_asp", "predicats", "priorites", "singularite",
    "syntaxe",
]
