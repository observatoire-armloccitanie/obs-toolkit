"""Graphiques standardisés (matplotlib) pour les publications de l'Observatoire.

Toutes les fonctions renvoient ``(fig, ax)`` pour permettre des retouches, et
acceptent un ``ax`` existant pour composer des figures. La charte (palette,
polices, grille discrète) est appliquée par :func:`appliquer_style` et
modifiable via :data:`PALETTE`.

Principes : une couleur par entité, toujours dans le même ordre ; un seul axe
des ordonnées ; marques fines ; légende dès deux séries ; textes en gris
foncé (jamais dans la couleur de la série).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

__all__ = [
    "PALETTE",
    "SEQUENTIELLE",
    "appliquer_style",
    "diagramme_barres",
    "histogramme",
    "boites",
    "courbes_survie",
    "forest_plot",
    "heatmap_correlation",
    "plan_factoriel",
    "courbe_roc",
    "evolution_temporelle",
    "enregistrer",
]

#: Palette catégorielle (ordre fixe, validée pour les daltonismes sur séries adjacentes).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
#: Rampe séquentielle (bleu, du clair au foncé).
SEQUENTIELLE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
#: Pôles divergents (bleu / rouge) et point neutre gris.
DIVERGENTE = ("#2a78d6", "#f0efec", "#e34948")
ENCRE = {"principale": "#0b0b0b", "secondaire": "#52514e", "discrete": "#8a8984", "grille": "#e4e3df"}
FOND = "#fcfcfb"


def _plt():
    import matplotlib.pyplot as plt

    return plt


def appliquer_style() -> None:
    """Applique la charte graphique à matplotlib (à appeler une fois en début de script)."""
    import matplotlib as mpl
    from cycler import cycler

    mpl.rcParams.update({
        "axes.prop_cycle": cycler(color=PALETTE),
        "figure.facecolor": FOND, "axes.facecolor": FOND, "savefig.facecolor": FOND,
        "axes.edgecolor": ENCRE["discrete"], "axes.labelcolor": ENCRE["secondaire"],
        "axes.titlecolor": ENCRE["principale"], "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": ENCRE["grille"], "grid.linewidth": 0.6,
        "axes.axisbelow": True, "xtick.color": ENCRE["secondaire"], "ytick.color": ENCRE["secondaire"],
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.frameon": False, "legend.fontsize": 9,
        "lines.linewidth": 2, "lines.markersize": 6, "font.family": "sans-serif",
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def _axe(ax, taille=(8, 5)):
    plt = _plt()
    if ax is None:
        fig, ax = plt.subplots(figsize=taille)
    else:
        fig = ax.figure
    return fig, ax


def _source(ax, source: str | None):
    if source:
        ax.annotate(source, xy=(0, -0.12), xycoords="axes fraction", fontsize=8,
                    color=ENCRE["discrete"], ha="left", va="top")


def diagramme_barres(
    donnees: pd.DataFrame | pd.Series,
    valeur: str | None = None,
    categorie: str | None = None,
    groupe: str | None = None,
    horizontal: bool = True,
    trier: bool = True,
    etiquettes: bool = True,
    format_etiquette: str = "{:.0f}",
    titre: str | None = None,
    source: str | None = None,
    ax=None,
):
    """Diagramme en barres (simples ou groupées).

    Parameters
    ----------
    donnees : DataFrame ou Series
        Series (index = catégories) ou DataFrame long avec ``categorie``, ``valeur``
        et éventuellement ``groupe``. Un DataFrame large (une colonne par série)
        est aussi accepté si ``valeur`` est ``None``.
    horizontal : bool
        Barres horizontales (recommandé pour des libellés longs).
    etiquettes : bool
        Affiche les valeurs au bout des barres.

    Examples
    --------
    >>> tf = descriptif.tableau_frequences(df, "niveau_formation", total=False)   # doctest: +SKIP
    >>> diagramme_barres(tf["pct"], format_etiquette="{:.1f} %")                  # doctest: +SKIP
    """
    if isinstance(donnees, pd.Series):
        large = donnees.to_frame(donnees.name or "valeur")
    elif valeur is not None:
        if groupe:
            large = donnees.pivot_table(index=categorie, columns=groupe, values=valeur, aggfunc="sum", observed=True)
        else:
            large = donnees.set_index(categorie)[[valeur]]
    else:
        large = donnees
    if trier and large.shape[1] == 1:
        large = large.sort_values(large.columns[0], ascending=horizontal)
    elif trier and horizontal:
        large = large.iloc[::-1]
    n_cat, n_ser = large.shape
    if horizontal:
        taille = (8, max(3, 0.4 * n_cat * max(1, n_ser * 0.6) + 1))
    else:
        taille = (max(6, 0.6 * n_cat + 2), 5)
    fig, ax = _axe(ax, taille)
    hauteur = 0.8 / n_ser
    pos = np.arange(n_cat)
    for j, col in enumerate(large.columns):
        # en horizontal, la première série est en haut du groupe (même ordre que la légende)
        decal = pos + 0.4 - hauteur * (j + 0.5) if horizontal else pos - 0.4 + hauteur * (j + 0.5)
        vals = large[col].to_numpy(dtype=float)
        couleur = PALETTE[j % len(PALETTE)]
        if horizontal:
            barres = ax.barh(decal, vals, height=hauteur * 0.92, color=couleur, label=str(col))
        else:
            barres = ax.bar(decal, vals, width=hauteur * 0.92, color=couleur, label=str(col))
        if etiquettes:
            for b, v in zip(barres, vals):
                if np.isfinite(v):
                    if horizontal:
                        ax.annotate(format_etiquette.format(v), (b.get_width(), b.get_y() + b.get_height() / 2),
                                    xytext=(3, 0), textcoords="offset points", va="center", fontsize=8,
                                    color=ENCRE["secondaire"])
                    else:
                        ax.annotate(format_etiquette.format(v), (b.get_x() + b.get_width() / 2, b.get_height()),
                                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8,
                                    color=ENCRE["secondaire"])
    lib = [str(i) for i in large.index]
    if horizontal:
        ax.set_yticks(pos, lib)
        ax.grid(axis="y", visible=False)
    else:
        longs = max(map(len, lib)) > 8
        ax.set_xticks(pos, lib, rotation=30 if longs else 0, ha="right" if longs else "center")
        ax.grid(axis="x", visible=False)
    if n_ser > 1:
        ax.legend(title=groupe, loc="best")
    if titre:
        ax.set_title(titre)
    _source(ax, source)
    return fig, ax


def histogramme(df: pd.DataFrame, variable: str, groupe: str | None = None, bins: int | str = "auto",
                densite: bool = False, titre: str | None = None, ax=None):
    """Histogramme d'une variable numérique, éventuellement superposé par groupe."""
    fig, ax = _axe(ax)
    if groupe:
        for j, (g, s) in enumerate(df.groupby(groupe, observed=True)[variable]):
            ax.hist(s.dropna(), bins=bins, density=True, histtype="step", linewidth=2,
                    color=PALETTE[j % len(PALETTE)], label=str(g))
        ax.legend(title=groupe)
        ax.set_ylabel("Densité")
    else:
        ax.hist(df[variable].dropna(), bins=bins, density=densite, color=PALETTE[0], edgecolor=FOND, linewidth=1)
        ax.set_ylabel("Densité" if densite else "Effectif")
    ax.set_xlabel(variable)
    if titre:
        ax.set_title(titre)
    return fig, ax


def boites(df: pd.DataFrame, variable: str, groupe: str, titre: str | None = None, ax=None):
    """Boîtes à moustaches d'une variable numérique par groupe."""
    fig, ax = _axe(ax)
    groupes = [(g, s.dropna().to_numpy()) for g, s in df.groupby(groupe, observed=True)[variable]]
    bp = ax.boxplot([v for _, v in groupes], patch_artist=True, widths=0.5,
                    medianprops={"color": ENCRE["principale"], "linewidth": 2},
                    flierprops={"marker": "o", "markersize": 3, "markerfacecolor": ENCRE["discrete"],
                                "markeredgecolor": "none", "alpha": 0.5})
    ax.set_xticks(range(1, len(groupes) + 1), [str(g) for g, _ in groupes])
    for patch in bp["boxes"]:
        patch.set_facecolor(SEQUENTIELLE[1])
        patch.set_edgecolor(PALETTE[0])
    ax.set_ylabel(variable)
    ax.set_xlabel(groupe)
    ax.grid(axis="x", visible=False)
    if titre:
        ax.set_title(titre)
    return fig, ax


def courbes_survie(
    km, ax=None, ic: bool = True, incidence: bool = False, titre: str | None = None,
    xlabel: str = "Durée", afficher_logrank: bool = True, effectifs_a_risque: Sequence[float] | None = None,
):
    """Courbes de Kaplan-Meier (en escalier) avec intervalles de confiance.

    Parameters
    ----------
    km : ResultatKaplanMeier
        Sortie de :func:`obskit.survie.kaplan_meier`.
    incidence : bool
        Trace ``1 - S(t)`` (part cumulée ayant connu l'événement), souvent plus
        parlant pour un accès à l'emploi.
    effectifs_a_risque : list of float, optional
        Durées auxquelles afficher les effectifs encore à risque sous le graphique.
    """
    fig, ax = _axe(ax)
    for j, (g, t) in enumerate(km.tables.items()):
        c = PALETTE[j % len(PALETTE)]
        y = 1 - t["survie"] if incidence else t["survie"]
        ax.step(t["temps"], y, where="post", color=c, label=str(g), linewidth=2)
        if ic:
            lo = 1 - t["ic_haut"] if incidence else t["ic_bas"]
            hi = 1 - t["ic_bas"] if incidence else t["ic_haut"]
            ax.fill_between(t["temps"], lo, hi, step="post", color=c, alpha=0.15, linewidth=0)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(left=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Part ayant connu l'événement" if incidence else "Probabilité de survie")
    if len(km.tables) > 1:
        ax.legend(title=km.groupe)
    if afficher_logrank and km.logrank:
        p = km.logrank["p_valeur"]
        ax.annotate(f"Log-rank : p {'< 0,001' if p < 0.001 else '= ' + format(p, '.3f')}",
                    xy=(0.98, 0.02 if not incidence else 0.95), xycoords="axes fraction", ha="right",
                    va="bottom" if not incidence else "top", fontsize=9, color=ENCRE["secondaire"])
    if effectifs_a_risque is not None:
        lignes = []
        for g, t in km.tables.items():
            vals = []
            for x in effectifs_a_risque:
                i = np.searchsorted(t["temps"].to_numpy(), x, side="left")
                vals.append(str(int(round(t["n_risque"].iloc[i]))) if i < len(t) else "0")
            lignes.append([str(g)] + vals)
        tab = ax.table(cellText=[l[1:] for l in lignes], rowLabels=[l[0] for l in lignes],
                       colLabels=[f"{x:g}" for x in effectifs_a_risque], loc="bottom",
                       bbox=[0, -0.35 - 0.06 * len(lignes), 1, 0.12 + 0.06 * len(lignes)], edges="open")
        tab.set_fontsize(8)
        fig.subplots_adjust(bottom=0.3)
    if titre:
        ax.set_title(titre)
    return fig, ax


def forest_plot(resultat, exposant: bool = True, termes: Sequence[str] | None = None,
                titre: str | None = None, ax=None):
    """Graphique en forêt des effets (OR, HR, IRR ou coefficients) avec IC.

    Parameters
    ----------
    resultat : ResultatRegression, ResultatCox ou DataFrame
        DataFrame avec colonnes ``coef``, ``ic_bas``, ``ic_haut``.
    exposant : bool
        Affiche ``exp(coef)`` sur échelle logarithmique (OR/HR/IRR) ; mettre
        ``False`` pour une régression linéaire.
    """
    tab = resultat.tableau if hasattr(resultat, "tableau") else resultat
    tab = tab.drop(index="const", errors="ignore")
    if termes is not None:
        tab = tab.loc[list(termes)]
    tab = tab.iloc[::-1]
    est, lo, hi = tab["coef"], tab["ic_bas"], tab["ic_haut"]
    if exposant:
        est, lo, hi = np.exp(est), np.exp(lo), np.exp(hi)
    fig, ax = _axe(ax, (8, max(3, 0.4 * len(tab) + 1)))
    y = np.arange(len(tab))
    signif = (lo > (1 if exposant else 0)) | (hi < (1 if exposant else 0))
    couleurs = [PALETTE[0] if s else ENCRE["discrete"] for s in signif]
    ax.hlines(y, lo, hi, color=couleurs, linewidth=2)
    ax.scatter(est, y, color=couleurs, s=40, zorder=3)
    ax.axvline(1 if exposant else 0, color=ENCRE["secondaire"], linewidth=1, linestyle="--")
    ax.set_yticks(y, list(tab.index))
    ax.grid(axis="y", visible=False)
    if exposant:
        ax.set_xscale("log")
        from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

        bas, haut = float(np.nanmin(lo)), float(np.nanmax(hi))
        reperes = [v for v in (0.1, 0.2, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3, 4, 5, 10, 20)
                   if bas / 1.15 <= v <= haut * 1.15] or [1]
        ax.xaxis.set_major_locator(FixedLocator(reperes))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
        ax.set_xlabel("Rapport (échelle log) - IC à 95 %")
    else:
        ax.set_xlabel("Coefficient - IC à 95 %")
    if titre:
        ax.set_title(titre)
    return fig, ax


def heatmap_correlation(matrice: pd.DataFrame, annoter: bool = True, titre: str | None = None, ax=None,
                        vmin: float | None = None, vmax: float = 1):
    """Carte de chaleur d'une matrice de corrélation ou d'association.

    Palette divergente bleu–gris–rouge pour des corrélations signées (-1 à 1),
    séquentielle bleue pour des associations positives (V de Cramér, eta…,
    détectées automatiquement quand toutes les valeurs sont ≥ 0).
    """
    from matplotlib.colors import LinearSegmentedColormap

    positive = bool(np.nanmin(matrice.to_numpy(dtype=float)) >= 0)
    if vmin is None:
        vmin = 0.0 if positive else -1.0
    if positive:
        cmap = LinearSegmentedColormap.from_list("obs_seq", ["#f0efec", *SEQUENTIELLE[1:]])
    else:
        cmap = LinearSegmentedColormap.from_list("obs_div", [DIVERGENTE[0], DIVERGENTE[1], DIVERGENTE[2]])
    n = len(matrice)
    fig, ax = _axe(ax, (max(5, 0.6 * n + 2), max(4, 0.6 * n + 1)))
    im = ax.imshow(matrice.to_numpy(dtype=float), cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(n), matrice.columns, rotation=45, ha="right")
    ax.set_yticks(range(n), matrice.index)
    ax.grid(False)
    if annoter:
        for i in range(n):
            for j in range(matrice.shape[1]):
                v = matrice.iat[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                            color=ENCRE["principale"] if abs(v) < 0.6 else "white")
    fig.colorbar(im, ax=ax, shrink=0.8)
    if titre:
        ax.set_title(titre)
    return fig, ax


def plan_factoriel(res, axes=(1, 2), quoi: str = "variables", n_max: int | None = 40,
                   supplementaires: bool = True, ax=None, titre: str | None = None):
    """Plan factoriel d'une ACP/ACM/AFC.

    Parameters
    ----------
    res : ResultatFactoriel
    axes : (int, int)
        Axes à représenter (numérotés à partir de 1).
    quoi : {"variables", "individus"}
        ``"variables"`` : cercle des corrélations (ACP), modalités (ACM) ou
        colonnes + lignes (AFC).
    n_max : int, optional
        Nombre maximal de points étiquetés (les mieux représentés, cos²).
    """
    a, b = (f"Dim.{axes[0]}", f"Dim.{axes[1]}")
    vp = res.valeurs_propres["pct_inertie"]
    fig, ax = _axe(ax, (7, 7) if res.methode == "ACP" and quoi == "variables" else (8, 6))
    ax.axhline(0, color=ENCRE["discrete"], linewidth=0.8)
    ax.axvline(0, color=ENCRE["discrete"], linewidth=0.8)
    if quoi == "individus":
        c = res.individus["coord"]
        ax.scatter(c[a], c[b], s=8, color=PALETTE[0], alpha=0.4, linewidth=0)
    else:
        nom_actif = {"ACP": "actif", "ACM": "modalités actives", "AFC": "colonnes"}[res.methode]
        blocs = [(nom_actif, res.variables, PALETTE[0])]
        if res.methode == "AFC":
            blocs.append(("lignes", res.individus, PALETTE[1]))
        if supplementaires and res.supplementaires:
            for nom, bloc in res.supplementaires.items():
                lib = f"{nom} (suppl.)"
                blocs.append((lib, bloc if isinstance(bloc, dict) else {"coord": bloc}, PALETTE[2]))
        for nom, bloc, couleur in blocs:
            c = bloc["coord"]
            if n_max is not None and "cos2" in bloc and len(c) > n_max:
                qual = bloc["cos2"][[a, b]].sum(axis=1)
                c = c.loc[qual.sort_values(ascending=False).index[:n_max]]
            if res.methode == "ACP" and nom == nom_actif:
                for lab, row in c.iterrows():
                    ax.annotate("", xy=(row[a], row[b]), xytext=(0, 0),
                                arrowprops={"arrowstyle": "->", "color": couleur, "linewidth": 1.5})
                    ax.text(row[a] * 1.07, row[b] * 1.07, str(lab), fontsize=8, color=ENCRE["principale"],
                            ha="center", va="center")
                t = np.linspace(0, 2 * np.pi, 200)
                ax.plot(np.cos(t), np.sin(t), color=ENCRE["discrete"], linewidth=0.8)
                ax.set_xlim(-1.1, 1.1)
                ax.set_ylim(-1.1, 1.1)
                ax.set_aspect("equal")
            else:
                ax.scatter(c[a], c[b], s=25, color=couleur, label=nom if len(blocs) > 1 else None)
                for lab, row in c.iterrows():
                    ax.annotate(str(lab), (row[a], row[b]), xytext=(3, 3), textcoords="offset points",
                                fontsize=8, color=ENCRE["principale"])
        if len(blocs) > 1:
            ax.legend()
    ax.set_xlabel(f"{a} ({vp.iloc[axes[0] - 1]:.1f} %)")
    ax.set_ylabel(f"{b} ({vp.iloc[axes[1] - 1]:.1f} %)")
    ax.set_title(titre or f"{res.methode} - plan {axes[0]}-{axes[1]}")
    return fig, ax


def courbe_roc(y, score, ax=None, titre: str | None = None, label: str | None = None):
    """Courbe ROC et AUC d'un score (ex. probabilités prédites par une logistique)."""
    from .regression import auc as _auc

    y = np.asarray(y, dtype=float)
    s = np.asarray(score, dtype=float)
    seuils = np.unique(s)[::-1]
    tpr = [0.0] + [np.mean(s[y == 1] >= t) for t in seuils]
    fpr = [0.0] + [np.mean(s[y == 0] >= t) for t in seuils]
    a = _auc(y, s)
    fig, ax = _axe(ax, (5.5, 5.5))
    ax.plot(fpr, tpr, color=PALETTE[0], label=f"{label + ' - ' if label else ''}AUC = {a:.3f}")
    ax.plot([0, 1], [0, 1], color=ENCRE["discrete"], linestyle="--", linewidth=1)
    ax.set_xlabel("1 - spécificité")
    ax.set_ylabel("Sensibilité")
    ax.set_aspect("equal")
    ax.legend(loc="lower right")
    if titre:
        ax.set_title(titre)
    return fig, ax


def evolution_temporelle(serie: pd.DataFrame | pd.Series, titre: str | None = None, ylabel: str | None = None,
                         marqueurs: bool = False, ax=None, source: str | None = None):
    """Courbe(s) d'évolution (sortie de :func:`obskit.descriptif.evolution` par exemple).

    Les colonnes ``variation_pct`` et ``glissement_annuel_pct`` sont ignorées
    (un seul axe des ordonnées : les tracer dans un graphique séparé).
    """
    d = serie.to_frame() if isinstance(serie, pd.Series) else serie
    d = d.drop(columns=[c for c in ("variation_pct", "glissement_annuel_pct") if c in d.columns])
    x = d.index.to_timestamp() if isinstance(d.index, pd.PeriodIndex) else d.index
    fig, ax = _axe(ax, (9, 4.5))
    for j, col in enumerate(d.columns):
        ax.plot(x, d[col], color=PALETTE[j % len(PALETTE)], label=str(col), marker="o" if marqueurs else None,
                markersize=4)
    if d.shape[1] > 1:
        ax.legend()
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0) if (d.min().min() >= 0) else None
    if titre:
        ax.set_title(titre)
    _source(ax, source)
    fig.autofmt_xdate()
    return fig, ax


def enregistrer(fig, chemin: str, formats: Sequence[str] = ("png",)) -> list[str]:
    """Enregistre une figure dans un ou plusieurs formats (``png``, ``svg``, ``pdf``)."""
    from pathlib import Path

    p = Path(chemin)
    p.parent.mkdir(parents=True, exist_ok=True)
    sorties = []
    for f in formats:
        cible = p.with_suffix(f".{f}")
        fig.savefig(cible)
        sorties.append(str(cible))
    return sorties
