# photofix

Deja las fotos de producto listas para publicar. Le das una foto tal cual sale
del móvil —torcida, con la mesa o una taza asomando por una esquina— y devuelve
el producto recto, centrado y con el encuadre limpio. El fondo se conserva: la
madera del suelo sigue siendo la madera del suelo.

Pensado para vender móviles, sobre todo iPhone, pero funciona con cualquier
producto apoyado sobre una superficie lisa y fotografiado desde arriba.

## Qué hace, en orden

1. **Aplica la orientación EXIF.** El iPhone casi nunca gira los píxeles: guarda
   la foto de lado y una etiqueta que dice cómo verla. Este es el arreglo de la
   foto «tumbada».
2. **Encuentra el producto.** Modela el color de la superficie a partir del
   borde del encuadre y marca todo lo que se sale de ese modelo; después afina
   el contorno con GrabCut. No descarga ningún modelo ni necesita internet.
3. **Endereza.** Mide la inclinación con dos métodos independientes —el
   rectángulo que mejor se ajusta al producto y la orientación dominante de sus
   bordes— y los combina. Si discrepan mucho, se queda con el más fiable; si el
   ángulo medido es absurdo, no gira nada.
4. **Quita lo que no interesa.** Descarta lo que el encuadre corta por un borde
   (una taza, el canto de la mesa, una mano) y las motas sueltas. Lo que está
   entero y junto al producto —un cable, la caja, la funda— se queda.
5. **Reencuadra.** Elige la proporción que mejor le va al conjunto, centra el
   producto y le deja aire alrededor.
6. **Retoca lo justo.** Un ajuste suave de niveles y nitidez para que un lote
   entero quede parejo. Sin tocar el color cálido de la madera.

Nunca corta el producto: si algo tiene que ceder, cede el margen, y siempre te
dice qué ha cedido.

## Instalación

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`pillow-heif` es lo que permite abrir los `.HEIC` directamente del iPhone. Si no
lo instalas, todo lo demás sigue funcionando con JPG y PNG.

## Uso desde el navegador

```bash
.venv/bin/python -m webapp.app
```

Abre <http://127.0.0.1:5000>, arrastra las fotos y las verás antes y después.
Puedes cambiar los ajustes y pulsar **Aplicar cambios** sin volver a subirlas, y
descargarlo todo en un `.zip`. Las fotos no salen de tu equipo.

## Uso desde la terminal

```bash
# una carpeta entera a la carpeta "listas"
.venv/bin/python -m photofix.cli fotos/ -o listas

# todo en 4:5, con el producto algo más pequeño
.venv/bin/python -m photofix.cli fotos/ -o listas --ratio 4:5 --fill 0.80

# ver qué ha detectado (escribe una copia con el análisis dibujado encima)
.venv/bin/python -m photofix.cli fotos/ -o listas --debug
```

Opciones que más se usan:

| Opción | Para qué |
| --- | --- |
| `--ratio` | `auto` (por defecto), `original`, `free` o fijo: `4:5`, `1:1`, `2:3`… |
| `--fill` | Cuánto ocupa el producto, de 0.3 a 1.0. Por defecto `0.86` |
| `--pad-to-ratio` | Fuerza un formato fijo ampliando el fondo si no cabe |
| `--keep-extras` | No descartar los objetos secundarios |
| `--no-straighten` | Recortar sin girar |
| `--max-side` | Lado mayor del resultado, `2000` por defecto |
| `--debug` | Escribe también el análisis dibujado sobre la foto |

Un formato fijo no siempre cabe: una foto vertical de un móvil no tiene suelo
suficiente para un encuadre apaisado. En ese caso te avisa y usa la proporción
más parecida, en lugar de devolverte otra forma sin decírtelo. Con
`--pad-to-ratio` lo consigue ampliando el encuadre con la superficie reflejada,
y solo hasta donde puede hacerlo sin duplicar el propio producto.

## Desde Python

```python
from photofix import Config, process_file

result = process_file("IMG_1234.HEIC", Config(ratio="4:5", fill=0.84))
result.image                    # la imagen final (BGR, como en OpenCV)
result.report.rotated_deg       # cuánto la ha girado
result.report.summary()         # una línea con lo que ha hecho
result.report.warnings          # avisos con código estable y texto
```

## Estructura

```
photofix/
  config.py     todos los ajustes en un sitio
  io_utils.py   lectura y escritura, incluido HEIC
  segment.py    localiza el producto y separa lo que sobra
  geometry.py   mide la inclinación y gira sin perder nada
  framing.py    decide el recorte: forma, margen y posición
  enhance.py    el retoque final
  pipeline.py   une los pasos y explica lo que ha hecho
  notes.py      avisos con código estable, para poder traducirlos
  cli.py        la línea de comandos
webapp/         interfaz de arrastrar y soltar (Flask)
tools/          generador de escenas de prueba
tests/          pruebas
```

## Pruebas

```bash
.venv/bin/python -m pytest tests/ -q
```

Las pruebas se apoyan en escenas generadas por `tools/make_samples.py`: un suelo
de madera con objetos encima, inclinados un ángulo conocido. Al conocerse el
ángulo real, el enderezado se puede puntuar en lugar de mirarlo a ojo — en estas
escenas el error queda por debajo de 0,1°.

Para verlas:

```bash
.venv/bin/python tools/make_samples.py --out samples
.venv/bin/python -m photofix.cli samples/ -o out --debug
```

## Límites conocidos

- Cuenta con que el fondo sea razonablemente uniforme y el producto no lo tape
  casi entero. Sobre una alfombra estampada o un mantel de cuadros la detección
  falla; cuando eso pasa, deja la foto como estaba y lo dice, en vez de girarla
  a ciegas.
- Endereza en el plano. No corrige la perspectiva de una foto tomada en
  diagonal: para eso hay que disparar desde arriba, como en las fotos de
  referencia.
- Los objetos que toquen el borde del encuadre se consideran ajenos. Si el
  producto llega hasta el borde, usa `--keep-extras`.
