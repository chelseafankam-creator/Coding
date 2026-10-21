"""Carte des prix prédits par un réseau de neurones, pour le TP2A (Bordeaux).

On fige les caractéristiques du bien (50 m², 3 pièces) et on fait varier
uniquement la position : le modèle répond alors « combien vaut CE bien, ICI ».
Le résultat est une carte de la prime de localisation apprise par le réseau.

Ce module ne contient QUE le code d'affichage — il ne définit ni n'entraîne
aucun modèle. Il s'utilise depuis le notebook, avec le modèle qui vient d'y
être entraîné :

    from carte_prix import trace_carte_prix
    trace_carte_prix(model, examples, feature_mean, feature_std, price_mean, price_std)
"""

import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize

# --- Le bien dont on trace le prix, par défaut ---------------------------
SURFACE = 50.0
NB_PIECES = 3.0

# --- Emprise géographique (commune de Bordeaux, légèrement élargie) ------
LON_MIN, LON_MAX = -0.642, -0.532
LAT_MIN, LAT_MAX = 44.807, 44.920

# --- Résolution de la grille (cases d'environ 75 m) ----------------------
N_LON, N_LAT = 115, 168

# Une case n'est coloriée que si le modèle a vu assez de ventes autour d'elle.
# Sans ce masque, on afficherait des prix inventés au-dessus de la Garonne, du
# parc bordelais ou des zones d'activité, là où le réseau n'a fait
# qu'extrapoler. Seuils calibrés pour ~3 100 appartements sur 49 km².
RAYON_CASES = 7      # demi-fenêtre du voisinage, en cases (~525 m)
VENTES_MIN = 5       # nb de ventes réelles exigé dans ce voisinage

ROOT = pathlib.Path(__file__).parent

# Quelques repères pour ancrer la lecture (nom, longitude, latitude).
REPERES = [
    ("Place de la Bourse", -0.5697, 44.8412),
    ("Chartrons",          -0.5720, 44.8545),
    ("Bacalan",            -0.5585, 44.8675),
    ("Le Lac",             -0.5810, 44.8865),
    ("La Bastide",         -0.5530, 44.8425),
    ("Caudéran",           -0.6120, 44.8455),
    ("Saint-Augustin",     -0.5985, 44.8320),
    ("Nansouty",           -0.5765, 44.8215),
    ("Gare Saint-Jean",    -0.5562, 44.8258),
]


def masque_densite(examples, lon_edges, lat_edges):
    """True là où le modèle extrapole (trop peu de ventes réelles autour).

    On compte les ventes par case avec un histogramme 2D, puis on somme sur un
    voisinage carré grâce à une image intégrale (summed-area table) : deux
    cumsum et quatre lectures par case, au lieu d'un calcul de distance entre
    les 19 320 cases et les 3 100 ventes.
    """
    counts, _, _ = np.histogram2d(
        examples.latitude.values, examples.longitude.values, bins=[lat_edges, lon_edges]
    )
    sat = np.pad(counts, ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)

    n_lat, n_lon = counts.shape
    r = RAYON_CASES
    i = np.arange(n_lat)
    j = np.arange(n_lon)
    i0 = np.clip(i - r, 0, n_lat)[:, None]
    i1 = np.clip(i + r + 1, 0, n_lat)[:, None]
    j0 = np.clip(j - r, 0, n_lon)[None, :]
    j1 = np.clip(j + r + 1, 0, n_lon)[None, :]

    voisinage = sat[i1, j1] - sat[i0, j1] - sat[i1, j0] + sat[i0, j0]
    return voisinage < VENTES_MIN


def trace_commune(ax):
    """Contour de la commune de Bordeaux, si le GeoJSON est présent."""
    chemin = ROOT / "data/bordeaux_commune.geojson"
    if not chemin.exists():
        print(f"  ({chemin.name} absent — pas de fond de carte)")
        return

    geo = json.load(open(chemin))
    for feature in geo["features"]:
        geometry = feature["geometry"]
        polygones = (
            [geometry["coordinates"]]
            if geometry["type"] == "Polygon"
            else geometry["coordinates"]
        )
        for polygone in polygones:
            anneau = np.array(polygone[0])  # contour extérieur
            ax.plot(
                anneau[:, 0], anneau[:, 1],
                color="#2b2b2b", linewidth=0.8, alpha=0.55, zorder=3,
            )


def trace_carte_prix(
    model,
    examples,
    feature_mean=None,
    feature_std=None,
    price_mean=None,
    price_std=None,
    surface=SURFACE,
    nb_pieces=NB_PIECES,
    rmse_val=None,
    titre="Ce que le réseau a appris de la localisation",
    figsize=(9, 10),
):
    """Trace la carte des prix prédits par `model` pour un bien figé.

    Args:
        model:      un réseau entraîné prenant, standardisées et DANS CET ORDRE,
                    (surface, nb_pieces, latitude, longitude), et rendant un
                    prix standardisé. Si les constantes de standardisation ne
                    sont pas fournies, le modèle doit prendre les valeurs brutes
                    et rendre un prix en euros (il standardise lui-même).
        examples:   le DataFrame des ventes, pour le masque de densité.
        feature_mean, feature_std: les constantes de standardisation des entrées,
                    de taille 4. Ce doivent être EXACTEMENT celles employées à
                    l'entraînement — donc, une fois les jeux séparés, celles
                    calculées sur le seul jeu d'entraînement. Sinon la grille est
                    standardisée dans un autre référentiel que le modèle, et la
                    carte ne veut rien dire.
        price_mean, price_std: les constantes de standardisation des prix.
        surface, nb_pieces: les caractéristiques du bien qu'on fige.
        rmse_val:   RMSE de validation, affichée dans le titre si fournie.
        titre:      première ligne du titre, par exemple le nom du modèle tracé.

    Returns:
        (fig, ax) matplotlib.
    """
    # Sans constantes, le modèle travaille directement en unités réelles.
    if feature_mean is None:
        feature_mean, feature_std = torch.zeros(4), torch.ones(4)
    if price_mean is None:
        price_mean, price_std = 0.0, 1.0
    price_mean, price_std = float(price_mean), float(price_std)

    lon_edges = np.linspace(LON_MIN, LON_MAX, N_LON + 1)
    lat_edges = np.linspace(LAT_MIN, LAT_MAX, N_LAT + 1)
    lons = (lon_edges[:-1] + lon_edges[1:]) / 2
    lats = (lat_edges[:-1] + lat_edges[1:]) / 2
    grid_lon, grid_lat = np.meshgrid(lons, lats)

    # Une ligne par case : seules latitude et longitude varient.
    # L'ordre des colonnes doit être celui de `features` dans le notebook.
    grille = np.column_stack([
        np.full(grid_lon.size, float(surface)),
        np.full(grid_lon.size, float(nb_pieces)),
        grid_lat.ravel(),
        grid_lon.ravel(),
    ])
    if grille.shape[1] != len(feature_mean):
        raise ValueError(
            f"Les constantes portent sur {len(feature_mean)} caractéristiques, la grille "
            f"en construit {grille.shape[1]}. Attendu : surface, nb_pieces, latitude, longitude."
        )
    G = (torch.tensor(grille, dtype=torch.float32) - feature_mean) / feature_std

    # Le notebook laisse le modèle en mode train après sa boucle : on bascule en
    # eval le temps de prédire, puis on rétablit l'état trouvé.
    etait_en_train = model.training
    model.eval()
    with torch.no_grad():
        prix = (model(G) * price_std + price_mean).numpy().reshape(grid_lat.shape)
    if etait_en_train:
        model.train()

    prix = np.ma.masked_where(masque_densite(examples, lon_edges, lat_edges), prix)

    # Magnitude → rampe séquentielle à teinte unique, clair = bon marché.
    # Bornes aux percentiles 2/95 : quelques poches très chères suffiraient
    # sinon à écraser tout le reste de la ville en blanc. La flèche du haut de
    # la barre annonce que des valeurs sortent de l'échelle — celle-ci reste
    # linéaire, on ne déforme pas la rampe.
    vmin, vmax = np.percentile(prix.compressed(), [2, 95])

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f4f4f2")  # le "hors données" reste neutre, pas blanc

    maille = ax.pcolormesh(
        lon_edges, lat_edges, prix,
        cmap="Reds", norm=Normalize(vmin, vmax),
        shading="flat", zorder=2,
    )
    trace_commune(ax)

    for nom, lon, lat in REPERES:
        ax.plot(lon, lat, "o", ms=4, color="#1a1a1a", zorder=4)
        ax.annotate(
            nom, (lon, lat), xytext=(5, 4), textcoords="offset points",
            fontsize=8.5, color="#1a1a1a", zorder=5,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.72),
        )

    barre = fig.colorbar(maille, ax=ax, shrink=0.82, pad=0.02, extend="max")
    barre.set_label(f"prix prédit d'un {surface:.0f} m² / {nb_pieces:.0f} pièces [€]")
    barre.ax.yaxis.set_major_formatter(lambda v, _: f"{v / 1000:,.0f} k€")
    barre.outline.set_visible(False)

    sous_titre = (
        f"Prix d'un appartement de {surface:.0f} m², {nb_pieces:.0f} pièces "
        "— Bordeaux, ventes 2025"
    )
    if rmse_val is not None:
        sous_titre += f"\nRMSE validation {rmse_val:,.0f} €"
    ax.set_title(
        f"{titre}\n{sous_titre}",
        fontsize=13, pad=12,
    )
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    # 1° de longitude est plus court qu'1° de latitude : sans ça, la carte est étirée.
    ax.set_aspect(1 / np.cos(np.deg2rad(44.86)))
    for bord in ax.spines.values():
        bord.set_visible(False)
    ax.tick_params(length=0, labelsize=8, colors="#666")

    fig.text(
        0.5, 0.025,
        f"Zones grises : moins de {VENTES_MIN} ventes dans un rayon de ~525 m "
        "— le modèle n'y aurait qu'extrapolé.",
        ha="center", fontsize=8, color="#666",
    )
    return fig, ax
