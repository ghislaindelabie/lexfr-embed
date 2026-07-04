#!/usr/bin/env python3
"""Render lexfr-soutenance-mentor.md (the mentor-facing soutenance presentation of the embedder
project) into ONE self-contained, mobile HTML published into the vault
(served at /doc/lexfr-soutenance-mentor). Source of truth = the .md; figures injected via
[[FIG:*]]. Re-run to refresh.  Run:  uv run --with markdown python3 build_soutenance_html.py
"""
import html
import re
from pathlib import Path

import markdown

SRC = Path(__file__).resolve().parent / "lexfr-soutenance.md"
OUT = SRC.with_suffix(".html")

SVG_GESTION = """
<svg viewBox="0 0 540 170" role="img" aria-label="Besoin, solution, bilan">
<defs><marker id="ga" markerWidth="9" markerHeight="9" refX="5" refY="3" orient="auto">
<path d="M0,0 L6,3 L0,6 Z" fill="var(--muted)"/></marker></defs>
<g font-family="inherit" text-anchor="middle">
<rect x="8" y="22" width="156" height="126" rx="10" fill="var(--card)" stroke="var(--line)"/>
<text x="86" y="48" font-size="14" font-weight="700" fill="var(--fg)">1 · Le besoin</text>
<text x="86" y="76" font-size="11" fill="var(--muted)">Aucun embedder</text>
<text x="86" y="93" font-size="11" fill="var(--muted)">FR national ouvert</text>
<text x="86" y="116" font-size="11" fill="var(--muted)">users pros + agents</text>
<text x="86" y="133" font-size="11" fill="var(--muted)">données publiques</text>
<line x1="166" y1="85" x2="190" y2="85" stroke="var(--muted)" stroke-width="1.6" marker-end="url(#ga)"/>
<rect x="192" y="22" width="156" height="126" rx="10" fill="var(--accent-soft)" stroke="var(--accent)"/>
<text x="270" y="48" font-size="14" font-weight="700" fill="var(--fg)">2 · La solution</text>
<text x="270" y="76" font-size="11" fill="var(--muted)">BGE-M3 + LoRA</text>
<text x="270" y="93" font-size="11" fill="var(--muted)">contrastif · 4 axes</text>
<text x="270" y="116" font-size="11" fill="var(--muted)">RunPod · W&amp;B</text>
<text x="270" y="133" font-size="11" fill="var(--muted)">TDD · CI</text>
<line x1="350" y1="85" x2="374" y2="85" stroke="var(--muted)" stroke-width="1.6" marker-end="url(#ga)"/>
<rect x="376" y="22" width="156" height="126" rx="10" fill="var(--card)" stroke="var(--line)"/>
<text x="454" y="48" font-size="14" font-weight="700" fill="var(--fg)">3 · Le bilan</text>
<text x="454" y="76" font-size="11" fill="var(--ok)" font-weight="700">+0,052 NDCG@10</text>
<text x="454" y="93" font-size="11" fill="var(--muted)">IC exclut zéro</text>
<text x="454" y="116" font-size="11" fill="var(--muted)">français préservé</text>
<text x="454" y="133" font-size="11" fill="var(--muted)">limites assumées</text>
</g></svg>
"""

SVG_PIPELINE = """
<svg viewBox="0 0 480 412" role="img" aria-label="Pipeline d'entrainement">
<defs><marker id="arr" markerWidth="9" markerHeight="9" refX="5" refY="3" orient="auto">
<path d="M0,0 L6,3 L0,6 Z" fill="var(--muted)"/></marker></defs>
<g font-family="inherit" text-anchor="middle">
<rect x="20" y="8" width="440" height="58" rx="10" fill="var(--card)" stroke="var(--line)"/>
<text x="240" y="32" font-size="14" font-weight="700" fill="var(--fg)">Données publiques</text>
<text x="240" y="50" font-size="11" fill="var(--muted)">LegalKit (~53k paires) + répétition générale (~7%)</text>
<line x1="240" y1="69" x2="240" y2="89" stroke="var(--muted)" stroke-width="1.5" marker-end="url(#arr)"/>
<rect x="20" y="92" width="440" height="58" rx="10" fill="var(--card)" stroke="var(--line)"/>
<text x="240" y="116" font-size="14" font-weight="700" fill="var(--fg)">Construction des paires</text>
<text x="240" y="134" font-size="11" fill="var(--muted)">(requête → article) + 1 négatif difficile filtré</text>
<line x1="240" y1="153" x2="240" y2="173" stroke="var(--muted)" stroke-width="1.5" marker-end="url(#arr)"/>
<rect x="20" y="176" width="440" height="58" rx="10" fill="var(--accent-soft)" stroke="var(--accent)"/>
<text x="240" y="200" font-size="14" font-weight="700" fill="var(--fg)">Entraînement contrastif</text>
<text x="240" y="218" font-size="11" fill="var(--muted)">MNRL ⊂ MatryoshkaLoss · LoRA sur BGE-M3</text>
<line x1="240" y1="237" x2="240" y2="257" stroke="var(--muted)" stroke-width="1.5" marker-end="url(#arr)"/>
<rect x="20" y="260" width="440" height="58" rx="10" fill="var(--card)" stroke="var(--line)"/>
<text x="240" y="284" font-size="14" font-weight="700" fill="var(--fg)">Évaluation (séparée)</text>
<text x="240" y="302" font-size="11" fill="var(--muted)">suite à 4 axes + garde anti-régression</text>
<line x1="240" y1="321" x2="240" y2="341" stroke="var(--muted)" stroke-width="1.5" marker-end="url(#arr)"/>
<rect x="20" y="344" width="440" height="58" rx="10" fill="var(--card)" stroke="var(--line)"/>
<text x="240" y="368" font-size="14" font-weight="700" fill="var(--fg)">Déploiement</text>
<text x="240" y="386" font-size="11" fill="var(--muted)">quantisation + dimensions Matryoshka (LDS)</text>
</g></svg>
"""

SVG_RESULTS = """
<svg viewBox="0 0 540 312" role="img" aria-label="Résultats NDCG@10 zéro-shot vs fine-tuné">
<g font-family="inherit">
<line x1="60" y1="30" x2="60" y2="255" stroke="var(--line)"/>
<line x1="60" y1="255" x2="515" y2="255" stroke="var(--line)"/>
<g font-size="11" fill="var(--muted)" text-anchor="end">
<line x1="60" y1="191" x2="515" y2="191" stroke="var(--line)" stroke-dasharray="3 4"/>
<line x1="60" y1="126" x2="515" y2="126" stroke="var(--line)" stroke-dasharray="3 4"/>
<line x1="60" y1="62" x2="515" y2="62" stroke="var(--line)" stroke-dasharray="3 4"/>
<text x="52" y="259">0,0</text><text x="52" y="195">0,1</text><text x="52" y="130">0,2</text><text x="52" y="66">0,3</text>
</g>
<text x="60" y="20" font-size="11" fill="var(--muted)">NDCG@10 (BSARD)</text>
<rect x="350" y="14" width="12" height="12" fill="var(--muted)"/><text x="368" y="24" font-size="11" fill="var(--fg)">zéro-shot</text>
<rect x="445" y="14" width="12" height="12" fill="var(--accent)"/><text x="463" y="24" font-size="11" fill="var(--fg)">fine-tuné</text>
<rect x="110" y="220" width="60" height="35" fill="var(--muted)"/>
<rect x="180" y="160" width="60" height="95" fill="var(--accent)"/>
<rect x="330" y="101" width="60" height="154" fill="var(--muted)"/>
<rect x="400" y="68" width="60" height="187" fill="var(--accent)"/>
<g font-size="12" fill="var(--fg)" text-anchor="middle" font-weight="700">
<text x="140" y="214">0,055</text><text x="210" y="154">0,148</text>
<text x="360" y="95">0,240</text><text x="430" y="62">0,292</text>
</g>
<g font-size="12" fill="var(--fg)" text-anchor="middle">
<text x="175" y="274" font-weight="700">MiniLM</text><text x="175" y="290" font-size="10.5" fill="var(--muted)">petit · +0,093</text>
<text x="395" y="274" font-weight="700">BGE-M3</text><text x="395" y="290" font-size="10.5" fill="var(--muted)">recette complète · +0,052</text>
</g>
</g></svg>
"""

SVG_AXES = """
<svg viewBox="0 0 480 268" role="img" aria-label="Suite d'évaluation à 4 axes">
<g font-family="inherit" text-anchor="middle">
<rect x="15" y="15" width="215" height="110" rx="10" fill="var(--accent-soft)" stroke="var(--accent)" stroke-width="2"/>
<text x="122" y="48" font-size="13" font-weight="700" fill="var(--fg)">1 · Requête pro → article</text>
<text x="122" y="74" font-size="11" fill="var(--accent)" font-weight="700">PRINCIPAL — le métier</text>
<text x="122" y="96" font-size="11" fill="var(--muted)">NDCG@10 · Recall@100</text>
<rect x="250" y="15" width="215" height="110" rx="10" fill="var(--card)" stroke="var(--line)"/>
<text x="357" y="48" font-size="13" font-weight="700" fill="var(--fg)">2 · Graphe / renvois</text>
<text x="357" y="74" font-size="11" fill="var(--muted)">relatedness (agents)</text>
<text x="357" y="96" font-size="11" fill="var(--muted)">extraction — sans confondant</text>
<rect x="15" y="145" width="215" height="110" rx="10" fill="var(--card)" stroke="var(--line)"/>
<text x="122" y="178" font-size="13" font-weight="700" fill="var(--fg)">3 · Garde anti-régression</text>
<text x="122" y="204" font-size="11" fill="var(--muted)">langue générale FR/EN</text>
<text x="122" y="226" font-size="11" fill="var(--muted)">contrat ±0,02 (codé)</text>
<rect x="250" y="145" width="215" height="110" rx="10" fill="var(--card)" stroke="var(--line)"/>
<text x="357" y="178" font-size="13" font-weight="700" fill="var(--fg)">4 · Robustesse profane</text>
<text x="357" y="204" font-size="11" fill="var(--muted)">secondaire</text>
<text x="357" y="226" font-size="11" fill="var(--muted)">BSARD (transfert)</text>
</g></svg>
"""

SVG_TIMELINE = """
<svg viewBox="0 0 540 200" role="img" aria-label="Plan en trois phases">
<defs><marker id="arr2" markerWidth="10" markerHeight="10" refX="6" refY="3" orient="auto">
<path d="M0,0 L6,3 L0,6 Z" fill="var(--muted)"/></marker></defs>
<line x1="40" y1="84" x2="505" y2="84" stroke="var(--muted)" stroke-width="2" marker-end="url(#arr2)"/>
<g text-anchor="middle" font-family="inherit">
<circle cx="110" cy="84" r="9" fill="var(--ok)"/>
<text x="110" y="52" font-size="13" font-weight="700" fill="var(--fg)">Phase 0 ✅</text>
<text x="110" y="112" font-size="10.5" fill="var(--muted)">Squelette gratuit</text>
<text x="110" y="128" font-size="10.5" fill="var(--muted)">(Kaggle) →</text>
<text x="110" y="144" font-size="10.5" fill="var(--muted)">chaîne prouvée</text>
<circle cx="285" cy="84" r="9" fill="var(--accent)"/>
<text x="285" y="52" font-size="13" font-weight="700" fill="var(--fg)">Phase 1 ✅</text>
<text x="285" y="112" font-size="10.5" fill="var(--muted)">Recette complète :</text>
<text x="285" y="128" font-size="10.5" fill="var(--muted)">hard-neg, répétition,</text>
<text x="285" y="144" font-size="10.5" fill="var(--muted)">résultat + IC</text>
<circle cx="455" cy="84" r="9" fill="none" stroke="var(--muted)" stroke-width="2"/>
<text x="455" y="52" font-size="13" font-weight="700" fill="var(--fg)">Phase 2</text>
<text x="455" y="112" font-size="10.5" fill="var(--muted)">Éval FR pro,</text>
<text x="455" y="128" font-size="10.5" fill="var(--muted)">publication</text>
<text x="455" y="144" font-size="10.5" fill="var(--muted)">(NLLP/JURIX)</text>
</g></svg>
"""

FIGS = {
    "gestion": (SVG_GESTION, "Schéma 1 — La démarche : besoin → solution → bilan."),
    "pipeline": (SVG_PIPELINE, "Schéma 2 — Le pipeline : des données publiques au déploiement."),
    "results": (SVG_RESULTS, "Schéma 3 — NDCG@10 sur BSARD, zéro-shot → fine-tuné (à configuration identique)."),
    "axes": (SVG_AXES, "Schéma 4 — La suite d'évaluation à 4 axes."),
    "timeline": (SVG_TIMELINE, "Schéma 5 — Le plan en trois phases."),
}

raw = SRC.read_text(encoding="utf-8")
if raw.startswith("---"):
    raw = raw.split("---", 2)[-1].lstrip("\n")
title = next((ln[2:].strip() for ln in raw.splitlines() if ln.startswith("# ")), "LexFR-Embed — Soutenance")

md = markdown.Markdown(extensions=["extra", "toc", "sane_lists"], extension_configs={"toc": {"toc_depth": "2"}})
body = md.convert(raw)

for key, (svg, cap) in FIGS.items():
    fig = f'<figure class="fig">{svg}<figcaption>{html.escape(cap)}</figcaption></figure>'
    body = body.replace(f"<p>[[FIG:{key}]]</p>", fig)

body = re.sub(r"(<h2\b[^>]*>)(.*?)(</h2>)",
              lambda m: f'{m.group(1)}{m.group(2)} <a class="toclink" href="#sommaire">↩</a>{m.group(3)}', body)

toc = "\n".join(
    f'<li><a href="#{html.escape(t["id"])}">{html.escape(t["name"])}</a></li>'
    for t in md.toc_tokens[0]["children"]
) if md.toc_tokens else ""

CSS = """
:root{--bg:#fff;--fg:#1c1e21;--muted:#5b6470;--line:#e5e7eb;--card:#f8fafc;--accent:#1d4ed8;--accent-soft:#eff6ff;--code-bg:#f3f4f6;--warn:#b45309;--warn-bg:#fffbeb;--ok:#16a34a;}
@media (prefers-color-scheme:dark){:root{--bg:#0f1115;--fg:#e6e8eb;--muted:#9aa3af;--line:#262a31;--card:#161a20;--accent:#7aa2ff;--accent-soft:#15203a;--code-bg:#11151b;--warn:#fbbf24;--warn-bg:#231d10;--ok:#34d399;}}
*{box-sizing:border-box}html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.62 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;word-wrap:break-word;overflow-wrap:break-word}
.wrap{max-width:820px;margin:0 auto;padding:0 18px 110px}
header.top{position:sticky;top:0;z-index:20;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 18px;display:flex;align-items:center;gap:12px}
header.top .h{font-weight:700;font-size:15px;flex:1;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
header.top a.cbtn{font-size:14px;text-decoration:none;color:#fff;background:var(--accent);padding:6px 12px;border-radius:8px;white-space:nowrap}
h1,h2,h3{line-height:1.28;margin:1.4em 0 .5em;scroll-margin-top:60px}
h1{font-size:1.6rem;border-bottom:2px solid var(--line);padding-bottom:.25em}
h2{font-size:1.3rem;border-bottom:1px solid var(--line);padding-bottom:.2em}
a{color:var(--accent)}p,li{font-size:1rem}
code{background:var(--code-bg);padding:.12em .35em;border-radius:5px;font-size:.86em;font-family:ui-monospace,Menlo,Consolas,monospace}
blockquote{margin:1em 0;padding:.6em 1em;background:var(--accent-soft);border-left:4px solid var(--accent);border-radius:0 8px 8px 0}
blockquote p{margin:.3em 0}
table{border-collapse:collapse;width:100%;display:block;overflow-x:auto;margin:1em 0;font-size:.95rem}
th,td{border:1px solid var(--line);padding:7px 10px;text-align:left;vertical-align:top}th{background:var(--card)}
#sommaire{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 18px;margin:16px 0;scroll-margin-top:60px}
#sommaire h2{border:none;margin:.2em 0 .4em;font-size:1.05rem}
#sommaire ol{margin:.2em 0 .2em 1.2em;padding:0}#sommaire li{margin:.15em 0}
.fig{margin:1.3em 0;text-align:center}
.fig svg{width:100%;height:auto;max-width:560px}
figcaption{color:var(--muted);font-size:.85rem;margin-top:.4em;font-style:italic}
a.toclink{font-size:.7em;text-decoration:none;color:var(--muted);opacity:.5}a.toclink:hover{opacity:1}
"""

JS = "var f=document.getElementById('fab');addEventListener('scroll',function(){f.style.opacity=scrollY>400?'1':'0';},{passive:true});f.style.opacity='0';f.style.transition='opacity .2s';"

page = f"""<!doctype html>
<html lang="fr"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="color-scheme" content="light dark">
<title>{html.escape(title)}</title>
<style>{CSS}</style></head>
<body><a id="haut"></a>
<header class="top"><div class="h">LexFR-Embed · soutenance</div><a class="cbtn" href="#sommaire">Sommaire</a></header>
<main class="wrap">
<nav id="sommaire"><h2>Sommaire</h2><ol>{toc}</ol></nav>
{body}
</main>
<a class="cbtn" id="fab" href="#sommaire" style="position:fixed;right:16px;bottom:16px;z-index:30;border-radius:23px;height:46px;display:flex;align-items:center;box-shadow:0 3px 10px rgba(0,0,0,.25)">↩ Sommaire</a>
<script>{JS}</script>
</body></html>
"""

OUT.write_text(page, encoding="utf-8")
print(f"Published {OUT}  ({len(page):,} bytes, {page.count('<figure')} figures, {page.count('<svg')} svg)")
