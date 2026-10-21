# Environnement de développement du TP, pour NixOS.
#
# Les paquets du .venv (numpy, torch, pandas) sont des wheels pip précompilées : elles
# sont liées dynamiquement contre libstdc++ et quelques autres bibliothèques système,
# que NixOS ne place pas dans les chemins habituels. D'où le fameux
#   ImportError: libstdc++.so.6: cannot open shared object file
# On les expose via LD_LIBRARY_PATH.
#
# Usage :  nix-shell        puis   python, jupyter, etc.

{ pkgs ? import <nixpkgs> { } }:

pkgs.mkShell {
  packages = [
    pkgs.python313 # doit correspondre à la version du .venv
    pkgs.uv
  ];

  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
    pkgs.stdenv.cc.cc.lib # libstdc++.so.6 — numpy, torch
    pkgs.zlib # libz.so.1
    pkgs.zstd
    pkgs.libGL # matplotlib
    pkgs.glib
  ];

  shellHook = ''
    if [ -d .venv ]; then
      source .venv/bin/activate
    else
      echo "Pas de .venv : créez-le avec  uv venv && uv pip install -r requirements.txt"
    fi
  '';
}
