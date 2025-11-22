import numpy as np
import re
"""
Función que transforma los datos mesh para lectura de python.

De preferencia tener el archivo con los datos en el mismo entorno de ejecución.
El nombre de la función identificarlo con el nombre de archivo

"""
def convertir_malla_matlab_a_funcion(ruta_archivo, nombre_funcion):
    """
    Convierte un archivo .m de Gmsh (formato MATLAB) en una función Python
    con el formato tipo read_<nombre>().
    Elimina ';' y espacios innecesarios del bloque de datos.
    """
    with open(ruta_archivo, 'r') as f:
        texto = f.read()

    # Buscar todas las secciones tipo msh.<NOMBRE> = [ ... ];
    secciones = re.findall(r"msh\.(\w+)\s*=\s*\[(.*?)\];", texto, re.DOTALL)
    nbNod = re.search(r"msh\.nbNod\s*=\s*(\d+)", texto)
    nbNod = int(nbNod.group(1)) if nbNod else None

    def parse_array(bloque):
        lineas = [l.strip() for l in bloque.splitlines() if l.strip()]
        filas = []
        for linea in lineas:
            # 🔹 Eliminar ';' al final de la línea y espacios extra
            linea = linea.rstrip(';').strip()
            if not linea:
                continue
            valores = [v.strip() for v in linea.split()]
            fila = ", ".join(valores)
            filas.append(f"[{fila}]")
        return ",\n        ".join(filas)

    codigo = []
    codigo.append(f"def read_{nombre_funcion}():")
    codigo.append(f'    """Reads mesh data equivalent to {nombre_funcion}.m (MATLAB version)."""')
    codigo.append("    msh = Mesh()")
    if nbNod:
        codigo.append(f"    msh.nbNod = {nbNod}\n")

    for nombre, bloque in secciones:
        bloque_formateado = parse_array(bloque)
        tipo = "int" if nombre.upper() in ["LINES", "TRIANGLES", "TETS", "QUADS"] else "float"
        codigo.append(f"    msh.{nombre} = np.array([\n        {bloque_formateado},\n    ], dtype={tipo})\n")

    codigo.append("    msh.MIN = msh.POS.min(axis=0)")
    codigo.append("    msh.MAX = msh.POS.max(axis=0)")
    codigo.append("\n    return msh")

    return "\n".join(codigo)

"""
Como ejecutar y guardar.
"""

python_code = convertir_malla_matlab_a_funcion("Maze01.m", "Maze01") 

with open("Maze01.py", "w") as f: # Aqui va el nombre del archivo
    f.write(python_code)