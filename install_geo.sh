#!/bin/bash
echo "Instalando Parche GEO - Leonel Hernán Torreblanca"
patch -p1 < GEO_HUBBLE_FINAL.patch
make -j
python3 -m pip install ./python --user --break-system-packages
python3 scripts/verify_hubble_geo.py
