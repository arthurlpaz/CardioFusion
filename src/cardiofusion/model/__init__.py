"""O modelo prognóstico: especificação, treino e artefato em disco.

Três responsabilidades separadas de propósito. `pipeline` define *o que* é o
modelo, `training` o ajusta e o avalia, `registry` sabe como ele vive em disco.
Quem serve o modelo depende só do `registry` — não do código de treino.
"""
