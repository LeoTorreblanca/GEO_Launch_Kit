#!/bin/bash
echo "Instalando Parche GEO - Leonel Hernán Torreblanca"

# Aplicamos el parche ignorando los archivos que no existen (como SOURCES.txt)
patch -p1 --forward --batch --force < GEO_HUBBLE_FINAL.patch

echo "Compilando motor de C modificado..."
make clean
make -j

echo "Instalando interfaz classy..."
python3 -m pip install ./python --user --break-system-packages

echo "Ejecutando validación de 12 dígitos..."
python3 scripts/verify_hubble_geo.py
