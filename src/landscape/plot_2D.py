import os

from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
import seaborn as sns

import numpy as np

from src.search.args import args
if not args.show:
  matplotlib.use('Agg')


def plot_2D(X, Y, Z, landscape_dir, landscape_name, levels=None, show=False, azim=-60, elev=30):
  if levels is None:
    vmin, vmax, vlevel = 0.1, 10, 0.5
    levels = np.arange(vmin, vmax, vlevel)
  # --------------------------------------------------------------------
  # Plot 2D contours
  # --------------------------------------------------------------------
  fig = plt.figure()
  CS = plt.contour(X, Y, Z, cmap='summer', levels=levels)
  plt.clabel(CS, inline=1, fontsize=8)
  save_name = os.path.join(landscape_dir, '%s_2dcontour.pdf'%landscape_name)
  fig.savefig(save_name, dpi=300,
              bbox_inches='tight', format='pdf')

  fig = plt.figure()
  save_name = os.path.join(landscape_dir, '%s_2dcontourf.pdf'%landscape_name)
  print(save_name)
  CS = plt.contourf(X, Y, Z, cmap='summer', levels=levels)
  fig.savefig(save_name, dpi=300,
              bbox_inches='tight', format='pdf')

  # --------------------------------------------------------------------
  # Plot 2D heatmaps
  # --------------------------------------------------------------------
  fig = plt.figure()
  sns_plot = sns.heatmap(Z, cmap='viridis', cbar=True, vmin=levels.min(), vmax=levels.max(),
                         xticklabels=False, yticklabels=False)
  sns_plot.invert_yaxis()
  save_name = os.path.join(landscape_dir, '%s_2dheat.pdf'%landscape_name)
  sns_plot.get_figure().savefig(save_name,
                                dpi=300, bbox_inches='tight', format='pdf')

  # --------------------------------------------------------------------
  # Plot 3D surface
  # --------------------------------------------------------------------
  fig = plt.figure()
  ax = Axes3D(fig, azim=azim, elev=elev)
  projections = []
  def on_click(event):
    azim, elev = ax.azim, ax.elev
    projections.append((azim, elev))
    print(azim, elev)

  surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
  fig.colorbar(surf, shrink=0.5, aspect=5)
  
  save_name = os.path.join(landscape_dir, '%s_3dsurface.pdf'%landscape_name)
  print(save_name)
  
  fig.savefig(save_name, dpi=300,
              bbox_inches='tight', format='pdf')
  if show: 
    cid = fig.canvas.mpl_connect('button_release_event', on_click)
    plt.show()


