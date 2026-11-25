# photofix

Deja las fotos de producto listas para publicar. Le das una foto tal cual sale
del móvil —torcida, con la mesa o una taza asomando por una esquina— y devuelve
el producto recto, centrado y con el encuadre limpio.

Por defecto respeta la superficie sobre la que hiciste la foto: la madera del
suelo sigue siendo la madera del suelo. Y cuando no hay superficie que respetar
—el móvil en la mano, una habitación de fondo— recorta el producto y lo coloca
sobre una.

Pensado para vender móviles, sobre todo iPhone, pero funciona con cualquier
producto fotografiado desde arriba.

## Ponerlo de pie

Si el móvil está tumbado en la foto o boca abajo, se gira para que quede
vertical con la parte de arriba arriba. Para saber cuál es la parte de arriba
busca dos cosas:

- **por detrás, el módulo de cámaras**, que va en la esquina superior
  izquierda. Es lo único con detalle en una trasera lisa;
- **por delante, la Dynamic Island**, la píldora negra centrada junto al borde
  de arriba.

Encuentra el móvil aunque esté dentro de la bandeja de la caja. Si no ve con
claridad ninguna de las dos cosas (el móvil de canto en la mano, una funda que
tapa las cámaras, un modelo con notch), **no gira nada**: girar mal es peor que
no girar. Para esos casos cada foto tiene un botón **Girar ↻** que da un cuarto
de vuelta a la derecha y recuerda tu elección aunque luego cambies los ajustes.
Desde la terminal: `--turn 90` (o `180`, `270`), o `--no-upright` para
desactivarlo.

## Cambiar el fondo y quitar la mano

Por defecto se conserva la superficie de la foto. Con `--background auto` (o
«Automático» en la web) se decide por foto:

- Si el producto ya está apoyado en una superficie lisa y uniforme, **se
  conserva**. Sustituir una madera real por una generada solo resta.
- Si no hay superficie aprovechable —el móvil en la mano, una habitación de
  fondo— **se recorta el producto y se coloca sobre madera**, centrado, recto y
  con su sombra.

La decisión no es un capricho: mide la dispersión de color del borde del
encuadre. En una foto cenital sobre una mesa de madera real sale entre 1,2 y 1,8
(la junta entre tablas o la esquina de la caja suman); en una foto con el móvil
en alto, 2,4. El corte está en 2,0. En la web, la opción por defecto es
«Conservar el original»: la madera generada se ve peor que la tuya.

Con `--background wood` se sustituye siempre, y pasando la ruta de una foto tuya
se usa esa superficie en lugar de la generada.

**La mano.** La que queda *detrás* del producto desaparece limpiamente. Los
dedos que se cierran *por delante* son otra cosa: se recupera la silueta del
producto para que no quede mordida, y lo que el dedo tapaba se reconstruye a
partir de lo que hay alrededor. Eso son píxeles inventados, así que te dice qué
porcentaje del producto ha tenido que rellenar. Si es más de un 5% te avisa de
que esa foto se hace mejor con el móvil apoyado — un arañazo debajo de un dedo
no aparecería en la foto, y eso es un problema para quien compra.

El recorte usa un modelo de 4,5 MB que se descarga una vez y se guarda en
caché. Sin él, se conserva el fondo original y se dice por qué.

## Qué hace, en orden

1. **Aplica la orientación EXIF.** El iPhone casi nunca gira los píxeles: guarda
   la foto de lado y una etiqueta que dice cómo verla.
2. **Pone el móvil de pie.** Cuartos de vuelta según dónde estén las cámaras o
   la Dynamic Island (ver arriba).
3. **Encuentra el producto.** Modela el color de la superficie a partir del
   borde del encuadre y marca todo lo que se sale de ese modelo; después afina
   el contorno con GrabCut. No descarga ningún modelo ni necesita internet.
4. **Endereza.** Mide la inclinación con dos métodos independientes —el
   rectángulo que mejor se ajusta al producto y la orientación dominante de sus
   bordes— y los combina. Si discrepan mucho, se queda con el más fiable; si el
   ángulo medido es absurdo, no gira nada.
5. **Quita lo que no interesa.** Descarta lo que el encuadre corta por un borde
   (una taza, el canto de la mesa, una mano) y las motas sueltas. Lo que está
   entero y junto al producto —un cable, la caja, la funda— se queda.
6. **Reencuadra.** Elige la proporción que mejor le va al conjunto, centra el
   producto y le deja aire alrededor.
7. **Retoca lo justo.** Un ajuste suave de niveles y nitidez para que un lote
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

Abre <http://127.0.0.1:5000>, arrastra las fotos y las verás antes y después con
un comparador que se arrastra con el dedo o el ratón. Puedes cambiar los ajustes
y pulsar **Aplicar cambios** sin volver a subirlas.

En macOS el puerto 5000 lo ocupa AirPlay; usa `--port 8000`.

## Uso desde el iPhone

Arranca el servidor abriéndolo también a la red local:

```bash
.venv/bin/python -m webapp.app --phone
```

Imprime dos direcciones: la de este ordenador y la del móvil, del tipo
`http://192.168.1.42:5000`. Escribe esa segunda en Safari, con el iPhone en la
misma wifi. Las fotos van del móvil a tu ordenador y vuelven; no salen de tu
casa.

En Safari, **Compartir → Añadir a pantalla de inicio** lo deja como una app más,
con su icono y a pantalla completa. El botón **Guardar** abre la hoja de
compartir de iOS, donde eliges «Guardar imagen» para que vaya al carrete.

El ordenador tiene que estar encendido y con el servidor abierto. Si prefieres
una dirección fija que funcione desde cualquier sitio, mira la sección
siguiente.

## Publicarla en internet

Para tener una dirección fija que funcione desde cualquier sitio, sin depender
de que el ordenador esté encendido.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/alcaparro2938/Reselling)

1. Pulsa el botón y entra en [Render](https://render.com) con tu cuenta de
   GitHub. Es gratis y no pide tarjeta.
2. Render lee el `render.yaml` del repositorio y rellena la configuración solo.
   Solo tienes que darle a **Apply**.
3. La primera construcción tarda unos cinco minutos, porque instala OpenCV.
4. Te queda una dirección del tipo `https://photofix-xxxx.onrender.com`. Ábrela
   en el iPhone y añádela a la pantalla de inicio.

Sobre el plan gratuito: se duerme tras un rato sin usarse, así que la primera
visita del día tarda unos 30 segundos en despertar; las siguientes van rápidas.
Tiene 512 MB de memoria, que da para una foto a la vez — un lote de veinte se
procesa igual, solo que en fila.

También sirve cualquier otro hosting que acepte un contenedor:

```bash
docker build -t photofix .
docker run -p 8000:8000 photofix
```

Variables que entiende: `PORT`, `PHOTOFIX_WORKDIR` (dónde guarda los lotes),
`PHOTOFIX_CONCURRENCY` (fotos a la vez), `PHOTOFIX_MAX_INPUT` (resolución máxima
de entrada) y `PHOTOFIX_MODEL_SIZE` (entrada de la red de recorte).

Sobre la memoria, que es lo que limita el plan gratuito de 512 MB: cambiar el
fondo cuesta bastante más que recortar, porque interviene la red neuronal.
Con los valores por defecto —entrada limitada a 2600 px y red a 256— una foto
de 12 MP llega a unos 470 MB y sale a 1876 px. Si le das más memoria, sube
`PHOTOFIX_MAX_INPUT` a 3000 y `PHOTOFIX_MODEL_SIZE` a 320 para la calidad
completa.

Ten en cuenta que así las fotos sí viajan a un servidor ajeno mientras se
procesan. Se borran solas al cabo de una hora, pero si te importa la
privacidad, la opción de la red local es la buena.

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
| `--fill` | Cuánto ocupa el producto, de 0.3 a 1.0. Por defecto `0.78` |
| `--pad-to-ratio` | Fuerza un formato fijo ampliando el fondo si no cabe |
| `--keep-extras` | No descartar los objetos secundarios |
| `--no-straighten` | Recortar sin enderezar |
| `--turn` | `auto` (por defecto) o un giro fijo a la derecha: `90`, `180`, `270` |
| `--no-upright` | No poner el móvil de pie |
| `--max-side` | Lado mayor del resultado, `2000` por defecto |
| `--background` | `auto`, `wood`, o la ruta de una foto de tu superficie |
| `--keep-hands` | No intentar quitar la mano |
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
  segment.py    localiza el producto contra su propia superficie
  cutout.py     lo recorta de cualquier fondo, y quita la mano
  background.py genera la superficie nueva y monta el producto encima
  orient.py     decide qué lado del móvil es arriba
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

- Ponerlo de pie está ajustado con iPhone de los últimos años (cámaras en
  esquina, Dynamic Island) y probado con pocas fotos reales. Con un Samsung, un
  iPhone con notch o una funda, lo normal es que no gire y toque el botón.

- Cuenta con que el fondo sea razonablemente uniforme y el producto no lo tape
  casi entero. Sobre una alfombra estampada o un mantel de cuadros la detección
  falla; cuando eso pasa, deja la foto como estaba y lo dice, en vez de girarla
  a ciegas.
- Endereza en el plano. No corrige la perspectiva de una foto tomada en
  diagonal: para eso hay que disparar desde arriba, como en las fotos de
  referencia.
- Los objetos que toquen el borde del encuadre se consideran ajenos. Si el
  producto llega hasta el borde, usa `--keep-extras`.
- La mano se detecta por color de piel. Un producto de color carne o madera muy
  clara puede confundirla; si el recorte se comería casi todo, no toca nada.
- Lo que un dedo tapa se reconstruye inventando píxeles. Sirve para que la foto
  no quede con agujeros, no para ocultar el estado del producto.
