"""Contratos HTTP — o que entra e sai pela rede.

Separados do `cardiofusion.domain` de propósito: o formato do JSON pode mudar
por razões de API (versionamento, ergonomia do cliente) sem que o vocabulário
do problema mude junto. A tradução entre os dois acontece aqui.
"""
