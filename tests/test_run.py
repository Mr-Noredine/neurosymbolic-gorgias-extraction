"""Le point d'entrée `./run` doit refuser d'écraser le banc.

`app.py` écrit son `.ann` À CÔTÉ de son entrée. Lancé sur un fichier de
`data/`, il remplace l'annotation de référence par sa propre sortie —
sans rien signaler. Le document se met alors à scorer parfaitement contre
lui-même et toutes les mesures suivantes deviennent fausses.

L'erreur a été commise une fois, le 12 août, sur `43-forets` ; elle n'a été
récupérable que parce que le banc était committé. Ce test transforme la leçon
en garde-fou.
"""
from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
RUN = RACINE / "run"


def lancer(*args):
    return subprocess.run([str(RUN), *args], capture_output=True, text=True,
                          cwd=str(RACINE), timeout=60)


class PointDEntree(unittest.TestCase):
    def test_le_script_est_executable(self):
        self.assertTrue(RUN.exists(), "`run` doit exister à la racine")
        self.assertTrue(RUN.stat().st_mode & 0o111, "`run` doit être exécutable")

    def test_la_syntaxe_est_valide(self):
        resultat = subprocess.run(["bash", "-n", str(RUN)],
                                  capture_output=True, text=True)
        self.assertEqual(resultat.returncode, 0, resultat.stderr)

    def test_l_aide_s_affiche_sans_argument(self):
        resultat = lancer()
        self.assertEqual(resultat.returncode, 0)
        self.assertIn("./run", resultat.stdout)


if __name__ == "__main__":
    unittest.main()
