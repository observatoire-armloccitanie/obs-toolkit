"""Lecture des extractions (CSV, Excel, Parquet) et export des résultats vers Excel."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping

import pandas as pd

__all__ = ["lire_fichier", "exporter_excel"]

_ENCODAGES = ("utf-8-sig", "utf-8", "cp1252", "latin-1")


def _detecter_csv(chemin: Path) -> tuple[str, str]:
    brut = chemin.read_bytes()[:200_000]
    for enc in _ENCODAGES:
        try:
            texte = brut.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:  # pragma: no cover - latin-1 décode tout
        enc, texte = "latin-1", brut.decode("latin-1")
    try:
        sep = csv.Sniffer().sniff(texte.splitlines()[0] if texte else ",", delimiters=";,\t|").delimiter
    except csv.Error:
        sep = ";" if texte.count(";") > texte.count(",") else ","
    return enc, sep


def lire_fichier(chemin: str | Path, nettoyer_noms: bool = False, **kwargs) -> pd.DataFrame:
    """Lit un fichier de données en détectant format, encodage et séparateur.

    Gère les exports usuels des outils décisionnels (OBIEE / i-milo,
    Power BI, Excel) : CSV en ``;`` encodés en Windows-1252, décimales à
    virgule, fichiers Excel ``.xlsx/.xls`` et Parquet.

    Parameters
    ----------
    chemin : str or Path
        Fichier ``.csv``, ``.txt``, ``.xlsx``, ``.xls`` ou ``.parquet``.
    nettoyer_noms : bool
        Applique :func:`obskit.preparation.nettoyer_noms_colonnes`.
    **kwargs
        Transmis à ``pandas.read_csv`` / ``read_excel`` / ``read_parquet``
        (ex. ``sheet_name="Feuille2"``, ``skiprows=2``). Pour conserver les
        zéros initiaux des codes (département ``"09"``, code commune), passer
        ``dtype={"departement": str}``.
    """
    chemin = Path(chemin)
    if not chemin.exists():
        raise FileNotFoundError(chemin)
    ext = chemin.suffix.lower()
    if ext in (".csv", ".txt", ".tsv"):
        enc, sep = _detecter_csv(chemin)
        options = {"sep": sep, "encoding": enc, "low_memory": False}
        if sep == ";":
            options["decimal"] = ","
        options.update(kwargs)
        df = pd.read_csv(chemin, **options)
    elif ext in (".xlsx", ".xlsm", ".xls"):
        df = pd.read_excel(chemin, **kwargs)
    elif ext == ".parquet":
        df = pd.read_parquet(chemin, **kwargs)
    else:
        raise ValueError(f"Extension non gérée : {ext}")
    if nettoyer_noms:
        from .preparation import nettoyer_noms_colonnes

        df = nettoyer_noms_colonnes(df)
    return df


def exporter_excel(
    tableaux: Mapping[str, pd.DataFrame],
    chemin: str | Path,
    index: bool = True,
    largeur_auto: bool = True,
    figer_entete: bool = True,
    notes: Mapping[str, str] | None = None,
) -> Path:
    """Exporte plusieurs tableaux dans un classeur Excel (un onglet par tableau).

    Parameters
    ----------
    tableaux : dict
        ``{"nom_onglet": DataFrame}`` (noms tronqués à 31 caractères).
    chemin : str or Path
        Fichier ``.xlsx`` de sortie.
    index : bool
        Écrit l'index des DataFrames.
    largeur_auto : bool
        Ajuste la largeur des colonnes.
    figer_entete : bool
        Fige la ligne d'en-tête.
    notes : dict, optional
        ``{"nom_onglet": "texte"}`` : note (source, champ, date) écrite sous le tableau.

    Returns
    -------
    Path du fichier créé.
    """
    from openpyxl.utils import get_column_letter

    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    noms_vus = set()
    with pd.ExcelWriter(chemin, engine="openpyxl") as writer:
        for nom, tab in tableaux.items():
            onglet = str(nom)[:31]
            for c in '[]:*?/\\':
                onglet = onglet.replace(c, "_")
            base, i = onglet, 2
            while onglet in noms_vus:
                onglet = f"{base[:28]}_{i}"
                i += 1
            noms_vus.add(onglet)
            t = tab.copy()
            if isinstance(t.columns, pd.MultiIndex):
                t.columns = [" | ".join(map(str, c)) for c in t.columns]
            t.to_excel(writer, sheet_name=onglet, index=index)
            ws = writer.sheets[onglet]
            if figer_entete:
                ws.freeze_panes = "B2" if index else "A2"
            if largeur_auto:
                for j, col in enumerate(ws.iter_cols(min_row=1, max_row=min(ws.max_row, 500)), start=1):
                    largeur = max((len(str(c.value)) for c in col if c.value is not None), default=8)
                    ws.column_dimensions[get_column_letter(j)].width = min(max(10, largeur + 2), 60)
            if notes and nom in notes:
                ws.cell(row=ws.max_row + 2, column=1, value=notes[nom])
    return chemin
