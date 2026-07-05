---
title: "LexFR-Embed — Présentation projet (soutenance)"
---

# LexFR-Embed — Un embedder juridique français

> **Support de soutenance — présentation du projet personnel.** Ce document présente le
> projet technique sur lequel j'ai travaillé : le *pourquoi* (le besoin), le *comment* (la
> méthodologie et les choix techniques), et le *bilan* (mes résultats à ce jour et ce que
> j'en retire). Le portfolio lui-même (delabie.tech) est présenté séparément.

---

## 1. Le projet en une page

**LexFR-Embed** est un **modèle de plongement (embedder) spécialisé pour le droit français** :
un modèle d'IA qui transforme un texte en vecteur numérique de façon à ce que **deux textes
qui parlent de la même chose se retrouvent proches** dans l'espace vectoriel. C'est le cœur
d'un moteur de recherche sémantique et de tout système RAG (un agent qui *cherche* les bons
textes avant de répondre).

Le projet sert **deux objectifs à la fois** :

1. **Mon projet personnel d'AI Engineer** — un projet de bout en bout (besoin → données →
   entraînement → évaluation → déploiement), mené avec une exigence de rigueur et d'honnêteté.
2. **Un besoin réel en entreprise** — améliorer la qualité de recherche de **LDS
   (LegalDataSpace)**, un espace de données juridiques souverain, dont la « brique de tuning
   vectoriel » est justement ce type de modèle.

**Contrainte assumée dès le départ :** **données publiques uniquement** (le corpus client de
LDS est contractuellement fermé, avec garantie de non-entraînement). Le modèle est donc
entraîné sur des données ouvertes et *déployable* tel quel — la souveraineté relève de
l'hébergement, pas de l'entraînement.

[[FIG:gestion]]

---

## 2. Le besoin — pourquoi ce projet

**La réflexion sur les besoins a précédé toute ligne de code.** J'ai commencé par une
recherche structurée (comparaison de l'existant, faisabilité, budget) pour vérifier que le
projet était **justifié** et non redondant. Trois constats :

- **Aucun embedder ouvert n'existe pour le droit *national* français.** Les seuls modèles
  juridiques ouverts en français sont entraînés sur le droit **belge** (BSARD/LLeQA), sur une
  base ancienne et de qualité modeste. **Construire est donc justifié.**
- **Les utilisateurs cibles sont des professionnels du droit et des agents logiciels**, pas
  le grand public. Ils emploient un registre professionnel (jargon, citations d'articles) et
  attendent une compréhension de la *structure* du droit (renvois entre articles). Cela oriente
  la conception et l'évaluation.
- **Un précédent probant existe** (un embedder juridique néerlandais, entraîné pour ~1 € de
  GPU, battant des solutions propriétaires) : la recette est ouverte et transposable.

**Contre-preuve gardée en tête (honnêteté) :** sur une base déjà excellente, le gain d'une
spécialisation est plus faible en relatif que sur une base modeste. Je devais donc *mesurer*
le gain, pas le supposer.

---

## 3. La solution — méthodologie et choix techniques

### La recette d'entraînement (apprentissage contrastif)

On part d'un modèle de base solide et ouvert — **BGE-M3** (licence MIT, multilingue, 568 M
paramètres) — et on le **spécialise** sur des paires **(question juridique → bon article)**.
La méthode, dite *contrastive*, apprend au modèle à **rapprocher** la question du bon article
et à **éloigner** les mauvais.

[[FIG:pipeline]]

Les briques techniques, et *pourquoi* chacune :

- **MNRL** (*Multiple Negatives Ranking Loss*) — la fonction d'entraînement standard : pour
  chaque question d'un lot, les bons articles des autres questions servent gratuitement de
  contre-exemples (« négatifs en lot »). Efficace et économe.
- **Négatif difficile filtré** — on ajoute 1 contre-exemple *piégeux* (proche mais faux) pour
  forcer le modèle à mieux discriminer. **Un seul, et filtré**, pour éviter de prendre par
  erreur un vrai bon article pour un négatif.
- **LoRA** — on n'entraîne que ~1 % des paramètres (des « adaptateurs »), pas les 568 M.
  Rapide, tient sur une carte 24 Go, et **protège le savoir généraliste** du modèle.
- **Matryoshka** — le vecteur devient *tronquable* (256 ou 512 dimensions au lieu de 1024)
  sans tout ré-encoder → moins de stockage et plus de vitesse en production.

### Les données

- **Entraînement : LegalKit** (~53 000 paires question→article de droit français, licence
  ouverte). Auditée, pas crue aveuglément (générée par un LLM, déséquilibrée par matière).
- **Répétition anti-oubli** — on mélange ~7 % de paires générales (FR + EN) pour que le modèle
  ne « désapprenne » pas le langage courant en se spécialisant.
- **Évaluation : BSARD** (droit belge). Utilisé **uniquement pour mesurer**, jamais pour
  entraîner — et présenté honnêtement comme un **proxy de transfert** (belge + profane), en
  attendant le jeu français professionnel, qui est le prochain livrable.

### L'outillage (démarche d'ingénieur)

- **RunPod** (GPU loués à la demande, auto-extinction : pas de facture oubliée) + **Weights &
  Biases** (suivi continu des entraînements) ;
- **TDD** (test avant le code — 48 tests automatiques) + **CI** (vérification à chaque
  modification) + **empreintes anti-fuite** (le jeu de test est gelé et signé *avant* tout
  entraînement, pour prouver l'absence de fuite de données).

---

## 4. L'évaluation — comment je mesure (et pourquoi c'est le cœur du projet)

Un modèle d'IA ne vaut que par la **confiance** qu'on peut accorder à ses chiffres. J'ai donc
construit une **évaluation à 4 axes** et une **discipline d'honnêteté** explicite.

[[FIG:axes]]

- **NDCG@10** — la note de qualité de recherche (0 à 1) : les bons articles sont-ils bien
  placés dans le top 10 ?
- **Avant / après à configuration identique** — tout gain vient de *mon* travail, pas d'un
  changement de réglage. (J'ai d'ailleurs détecté et corrigé un piège où un chiffre mélangeait
  deux configurations — la rigueur, c'est aussi refuser ses propres bons chiffres trop beaux.)
- **Intervalle de confiance (IC) à 95 %** — si l'IC d'un gain **exclut zéro**, le gain est
  statistiquement **réel**, pas un coup de chance.
- **MDE** (plus petit effet détectable) — un gain plus petit que ~0,03 est déclaré « dans le
  bruit », honnêtement.
- **Garde anti-régression** — un test sur des tâches *non-juridiques* FR/EN pour vérifier que
  la spécialisation ne dégrade pas le français courant.

---

## 5. Les résultats à ce jour — le bilan

### Le pipeline est prouvé de bout en bout

[[FIG:results]]

| Modèle | NDCG@10 zéro-shot → fine-tuné | Gain |
|---|---|---|
| MiniLM (petit modèle) | 0,055 → 0,148 | +0,093 (×2,7) |
| **BGE-M3 (recette complète, GPU)** | **0,240 → 0,292** | **+0,052** |

### Le résultat principal, mesuré proprement

> **BSARD NDCG@10 : 0,240 → 0,292** (gain **+0,052**),
> intervalle de confiance 95 % **[+0,031 ; +0,076]** — **exclut zéro** ⇒ gain **statistiquement réel**,
> et au-dessus du seuil de détection (±0,031). Résultat **reproduit sur 3 entraînements** (+0,050 / +0,044 / +0,052).

### La garde anti-régression a fait son travail

Sur 7 tâches générales, **6 sont préservées ou améliorées** (le français est intégralement
protégé, la similarité sémantique progresse même). **Une seule régresse** : la recherche
**financière en anglais** (−0,026) — la tâche la plus éloignée du droit français. J'ai
diagnostiqué la cause (le plancher de répétition anti-oubli n'était pas encore branché), je
l'ai implémenté, et j'ai compris précisément pourquoi ce résidu subsiste (la répétition
généraliste ne protège pas *spécifiquement* le sous-domaine financier).

**Ce n'est pas un échec, c'est une démonstration de rigueur :** j'ai construit un garde-fou,
il a détecté une vraie régression, je l'ai expliquée et cadrée. C'est exactement le genre de
mesure honnête qui fait la valeur d'un projet d'IA.

### Les limites, assumées

- BSARD est **belge et profane** : un proxy de transfert, pas une preuve « français + pro ».
  Le jeu d'évaluation français professionnel est le premier livrable de la suite.
- LegalKit est **généré par un LLM** → à auditer.
- Un petit **résidu de régression** subsiste sur un domaine hors-cible (finance anglaise).

---

## 6. La gestion de projet

Le projet a été conduit en **trois phases**, avec une logique de risque maîtrisé :

[[FIG:timeline]]

- **Phase 0 — squelette gratuit** (Kaggle) : prouver que *toute la chaîne* fonctionne avant
  de dépenser le moindre euro de GPU.
- **Phase 1 — recette complète** (GPU) : hard negatives, répétition, garde anti-régression,
  résultat principal avec IC.
- **Phase 2 — publication** (après) : jeu d'évaluation français, préprint, diffusion ouverte.

**Décisions structurantes, tracées** (chaque choix a un *pourquoi*) : construire plutôt que
réutiliser ; base BGE-M3 ; LoRA plutôt que tout ré-entraîner ; données publiques uniquement ;
BSARD en proxy honnête. Un **journal de projet** trace décisions, étapes et résultats.

**Gestion du risque et du budget** — exemple concret : un entraînement à plus grande échelle
s'est révélé trop lent puis s'est **bloqué** (GPU à 0 % d'utilisation). Je l'ai détecté via le
monitoring, **arrêté immédiatement** pour ne pas gaspiller, et j'en ai tiré un correctif de
configuration. Le budget GPU total du projet à ce jour reste de l'ordre de **quelques euros**,
grâce à l'auto-extinction systématique des machines.

---

## 7. Ce que ce projet m'a apporté

*(Section réflexive — à personnaliser oralement.)*

- **Compétences techniques acquises :** fine-tuning contrastif d'embedders (MNRL, LoRA,
  Matryoshka), **évaluation rigoureuse** (NDCG, intervalles de confiance, MDE, contrôle de
  fuite), **MLOps** (GPU cloud, suivi d'expériences, auto-extinction), et une **démarche
  d'ingénierie logicielle** (TDD, CI, journal de décisions).
- **Évolution de ma représentation du métier :** j'ai compris que le cœur du métier d'AI
  Engineer n'est pas « faire tourner un entraînement », mais **cadrer un besoin réel, mesurer
  honnêtement, et gérer le risque** (compute, données, sur-spécialisation). Un chiffre n'a de
  valeur que par la confiance qu'on peut y accorder.
- **Réflexivité :** refuser un bon chiffre trop beau (la correction du « mélange de
  configurations »), présenter un échec de garde comme une preuve de rigueur, tracer chaque
  décision — c'est là que se joue le sérieux d'un projet.
- **Objectif professionnel :** consolider un profil d'**AI Engineer** orienté NLP/recherche
  d'information, avec un ancrage sur des cas d'usage à fort enjeu (le juridique, la souveraineté
  des données).

---

## 8. La suite

- **Court terme :** construire le **jeu d'évaluation français professionnel** (le vrai
  juge-de-paix, via les citations réelles comme signal de pertinence) ; relancer
  l'entraînement à plus grande échelle avec une configuration allégée.
- **Moyen terme :** requêtes praticien synthétiques, signal de *graphe/renvois* entre articles,
  droit européen (EUR-Lex).
- **Phase 2 :** préprint et diffusion ouverte du modèle et du jeu de données.
