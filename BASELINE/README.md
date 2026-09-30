# Explicaciones Entrega 1

* **Autor:** Leandro Alejo Rodriguez Alvarez
* **Fecha de entrega:** 30/09/2026
* **Asignatura:** HPCTools

---

## Archivos entregados

* `setup_env.sh`: Script para crear el entorno virtual e instalar las librerías necesarias.
* `train_baseline.py`: Código en Python con el modelo BERT para SQuAD y las diferentes opciones para activar las mejoras (DataLoader, precisión mixta, compilado, etc.).
* `run_baseline.sh`: Script para Slurm que lanza todas las pruebas una detrás de otra y saca las tablas de resultados.
* `README.md`: Explicación de los pasos seguidos y las decisiones tomadas.

---

## Guía rápida de pasos para ejecutar

1. **Crear el entorno virtual:**  
   Ejecutar en la terminal:
   ```sh setup_env.sh ```
2. **Configurar los pasos:**  
   Abrir el archivo `run_baseline.sh` y revisar la variable `STEPS`. En este caso se configuró en **500 pasos** (más 10 de calentamiento) tras hacer varias pruebas, buscando cumplir la indicación de la tarea de que las ejecuciones durasen alrededor de 1 minuto.
3. **Lanzar las pruebas en el cluster:**  
   Enviar el trabajo con Slurm:
   ```sbatch run_baseline.sh```

---

## Explicacion de cada archivo entregado


## 0. (`bert_profiler_results_10149881`) Ultimo .log generado, usado para sacar las conclusiones finales en el apartado 5. 




## 1. (`setup_env.sh`) Configuracion del entorno

Se crea un entorno virtual (`venv`) para no tocar las librerías del sistema del Finisterrae III y asegurarse de que se instalan exactamente las versiones que necesitamos.

Además, para facilitar la creacion del entorno virtual, el script detecta automáticamente en qué carpeta está guardado y crea la carpeta del entorno (`entornoTarea1`) ahí dentro.

### Versión de PyTorch y CUDA elegida
Al utilizar `nvidia-smi` en un nodo que utilice una GPU Nvidia A100 se aprecia que el sistema cuenta con **CUDA 12.8**. 

Al ir a la página web oficial de PyTorch, la versión recomendada para CUDA 12 era la **12.6**. Como CUDA es retrocompatible (una versión superior del driver esta preparada para versiones anteriores), instalamos esa versión con soporte para CUDA 12.6.

### Librerías adicionales instaladas
Además de `torch`, se instalan los siguientes paquetes:
* `transformers`: Es la librería de Hugging Face de donde descargamos la arquitectura y los pesos ya entrenados del modelo BERT (`bert-base-uncased`), además del tokenizador para procesar el texto.
* `datasets`: Permite descargar, cargar y formatear directamente el conjunto de datos de preguntas y respuestas SQuAD sin tener que bajarlo ni procesarlo a mano.

### Modelo y Dataset
Tal y como se pide en el enunciado de la práctica, se han utilizado:
* **Modelo:** `bert-base-uncased` adaptado para respuesta a preguntas (Question Answering).
* **Dataset:** `rajpurkar/squad` (SQuAD), limitando las secuencias a una longitud fija de 384 tokens para que el entrenamiento sea regular.






## 2. (`requirements.txt`) Dependencias del proyecto

Este archivo contiene la lista de librerías necesarias para ejecutar los scripts. Es leído automáticamente durante la ejecución de `setup_env.sh` para instalar todas las herramientas dentro del entorno virtual.

### Contenido del archivo:
```text
--extra-index-url [https://download.pytorch.org/whl/cu126](https://download.pytorch.org/whl/cu126)
torch
transformers
datasets
```






## 3. (`run_baseline.sh`) Ejecución de las pruebas con Slurm

Este script se encarga de solicitar los recursos necesarios en el cluster (1 nodo, 1 GPU NVIDIA A100 y 32 CPUs) y lanzar de forma secuencial el archivo `train_baseline.py` 5 veces, siguiendo la misma metodología vista en el Lab 2 de las asignatura:

1. **[1/5] Baseline (Punto de partida):**  
   Ejecuta el entrenamiento en precisión simple estándar (FP32), con un tamaño de lote (`batch_size`) de 16 y sin procesos adicionales de CPU para la carga de datos (`num_workers=0`). Sirve como referencia base para medir el impacto de las mejoras posteriores.

2. **[2/5] Optimized DataLoader:**  
   Mantiene la configuración anterior pero activa 16 trabajadores en la CPU (`num_workers=16`) y fija la memoria en RAM (`pin_memory`). El objetivo es comprobar si la preparación de datos en la CPU acelera el flujo de batches hacia la GPU.

3. **[3/5] Mixed Precision (AMP - BF16):**  
   Activa la precisión mixta automática mediante `torch.autocast` usando el formato `bfloat16`. Esto permite que las multiplicaciones de matrices densas de BERT se ejecuten directamente en los **Tensor Cores** de la GPU A100, reduciendo el consumo de memoria VRAM y multiplicando la velocidad de cálculo.

4. **[4/5] Double Batch Size:**  
   Aprovechando la memoria VRAM liberada por el uso de BF16, se duplica el tamaño del batch de 16 a 32 (`batch_size=32`). Al enviar bloques más grandes de datos a la vez, se saturan mejor las unidades de cálculo de la GPU y aumenta el rendimiento por segundo (`samples/s`).

5. **[5/5] Compiled Model (`torch.compile`):**  
   Aplica `torch.compile` sobre el modelo. El compilador analiza el grafo de operaciones y fusiona capas consecutivas en kernels de cómputo más grandes y eficientes, eliminando transferencias innecesarias de memoria interna en la GPU.

Al finalizar las 5 ejecuciones, el script agrupa los parametros relevantes y genera 3 tablas comparativas, donde se pueden ver las mejoras en tiempo de cada uno de los cambios aplicados.




## 4. (`train_baseline.py`) Código de entrenamiento y profiling

Tal y como se permite y menciona en el enunciado del entregable, la implementación de este script se generó con la ayuda de herramientas de IA, centrándose en cumplir de forma estricta los objetivos técnicos que se pedían.

El script está dividido en los siguientes bloques principales:

### Parámetros por línea de comandos (`argparse`)
Se definieron argumentos independientes (`--batch_size`, `--num_workers`, `--pin_memory`, `--use_amp`, `--compile`, `--profile`) para poder activar o desactivar cada técnica de forma limpia desde el script de Slurm sin tener que tocar el código de Python entre prueba y prueba.

### Cálculo automático de FLOPs con `FlopCounterMode`
En lugar de hacer una estimación teórica manual a ojo de las operaciones matemáticas, se utiliza la herramienta nativa `FlopCounterMode` de PyTorch. 
Antes de empezar el entrenamiento, se pasa una secuencia de prueba por el modelo (haciendo el forward y el backward) para contar exactamente cuántas operaciones de coma flotante (FLOPs) cuesta procesar una muestra. Ese valor exacto se usa después para calcular de forma fiable los **achieved TFLOP/s** y la **MFU** alcanzada.

### Preparación del dataset y padding constante
El texto de las preguntas y contextos de SQuAD se procesa fijando siempre la longitud máxima a 384 tokens con relleno (`padding="max_length"`). Además, en el DataLoader se activa `drop_last=True` para descartar el último lote si queda incompleto. Esto es muy importante sobre todo al usar `torch.compile`, ya que evita que el tamaño de las matrices cambie y obligue a la GPU a recompilar los kernels en mitad del entrenamiento.

### Paso de entrenamiento eficiente
La función `train_one_step` incluye buenas prácticas para no frenar la GPU:
* Se usa `optimizer.zero_grad(set_to_none=True)` para liberar memoria de gradientes de forma más rápida.
* Con `--pin_memory`, los datos se pasan a la GPU de forma no bloqueante (`non_blocking=True`).
* Si se activa `--use_amp`, el forward se envuelve en `torch.autocast('cuda', dtype=torch.bfloat16)` para usar los Tensor Cores.
* No se llama a `.item()` en cada iteración del bucle para no obligar a la CPU y a la GPU a sincronizarse continuamente, lo que arruinaría la medición del tiempo.

### Medición de tiempos y calentamiento (Warmup)
Para que los datos de rendimiento sean representativos del estado estacionario:
* Se descartan los primeros 10 pasos (`--warmup 10`), evitando medir el coste inicial de arranque o el tiempo extra que tarda `torch.compile` en compilar el grafo en su primer paso.
* Se usa `torch.cuda.synchronize()` antes de iniciar y al detener el cronómetro (`time.perf_counter()`), asegurando que medimos el tiempo real que tarda la GPU en procesar los 500 pasos.

### Análisis con `torch.profiler`
Si se activa el flag `--profile`, el script ejecuta una ventana corta de rastreo con `schedule(wait=1, warmup=1, active=3)`. Esto permite desglosar qué operaciones internas de CUDA y de CPU consumen más tiempo sin llenar la memoria RAM del nodo con millones de trazas.





## 5. Resultados y Conclusiones

A continuación se muestran las tablas obtenidas tras la ejecución completa de los 500 pasos en un nodo con NVIDIA A100 del Finisterrae III:

```text
=============================================================================================================================================
TABLA 1: RESUMEN COMPARATIVO DE RENDIMIENTO (NVIDIA A100-PCIE-40GB)
=============================================================================================================================================
Etapa / Configuración              Precisión   BS   Workers  Throughput   Achieved     MFU Real      MFU Global    Peak VRAM   Speedup
                                                             (samples/s)  (TFLOP/s)   (Hardware)    (vs 312 TF)      (MB)
---------------------------------------------------------------------------------------------------------------------------------------------
[1/5] Baseline                      FP32       16      0        53.98      11.5915       59.44%*       3.72%     5095.14 MB    1.00x
[2/5] Optimized DataLoader          FP32       16     16        54.25      11.6477       59.73%*       3.73%     5095.14 MB    1.01x
[3/5] Mixed Precision (AMP)         BF16       16     16       114.56      24.5986        7.88%        7.88%     3903.25 MB    2.12x
[4/5] Double Batch Size             BF16       32     16       191.48      41.1146       13.18%       13.18%     6326.34 MB    3.55x
[5/5] Compiled Model (Inductor)     BF16       32     16       290.61      62.4000       20.00%       20.00%     5713.08 MB    5.38x
---------------------------------------------------------------------------------------------------------------------------------------------
- En FP32 (Etapas 1 y 2), el hardware opera sobre CUDA Cores convencionales (techo de 19.5 TFLOP/s).
  En BF16 (Etapas 3, 4 y 5), se activan los Tensor Cores de la arquitectura Ampere (techo de 312.0 TFLOP/s).
=============================================================================================================================================

=============================================================================================================================================
TABLA 2: CUMPLIMIENTO DE MÉTRICAS EXIGIDAS EN LA MEMORIA DEL LABORATORIO
=============================================================================================================================================
Etapa                               samples / s     peak GPU memory     achieved TFLOP/s        MFU %            final loss (500 steps)
---------------------------------------------------------------------------------------------------------------------------------------------
[1/5] Baseline                         53.98          5095.14 MB             11.5915                3.72%                1.937586
[2/5] Optimized DataLoader             54.25          5095.14 MB             11.6477                3.73%                2.057086
[3/5] Mixed Precision (BF16)          114.56          3903.25 MB             24.5986                7.88%                1.336832
[4/5] Double Batch Size               191.48          6326.34 MB             41.1146               13.18%                0.916244
[5/5] Compiled Model                  290.61          5713.08 MB             62.4000               20.00%                1.023299
=============================================================================================================================================

=============================================================================================================================================
TABLA 3: TIEMPOS DE ENTRENAMIENTO MEDIDOS (1 GPU NVIDIA A100)
=============================================================================================================================================
Etapa                               Pasos Medidos   Batch Size   Throughput        Tiempo Medido       Tiempo Medido
                                                                (samples/s)         (Segundos)          (Minutos)
---------------------------------------------------------------------------------------------------------------------------------------------
[1/5] Baseline                        500            16          53.98           148.19 s            2.47 min
[2/5] Optimized DataLoader            500            16          54.25           147.48 s            2.46 min
[3/5] Mixed Precision (BF16)          500            16         114.56            69.83 s            1.16 min
[4/5] Double Batch Size               500            32         191.48            83.56 s            1.39 min
[5/5] Compiled Model                  500            32         290.61            55.06 s            0.92 min
=============================================================================================================================================
```

---

### Conclusiones del análisis

1. **Etapa 1 vs Etapa 2 (¿Por qué no hay mejora con 16 workers?):**  
   Al pasar de 0 a 16 workers apenas se nota diferencia (de 53.98 a 54.25 muestras/segundo). Esto ocurre porque el dataset ya se dejó procesado en memoria RAM al inicio del script. Como entregar los números desde la RAM a la CPU tarda poquísimo tiempo (apenas un par de milisegundos) y la GPU en FP32 tarda mucho más en hacer los cálculos matemáticos pesados, la GPU nunca se queda esperando por datos. El cuello de botella está 100% en el cálculo de la GPU y no en la carga de datos.

2. **Precisión mixta con BF16 (El mayor salto de rendimiento):**  
   Activar `use_amp` dobla la velocidad (Speedup de **2.12x** y 114.56 muestras/s) y además reduce el consumo de memoria VRAM (de ~5 GB a ~3.9 GB). Esto se debe a que dejamos de usar los núcleos estándar de FP32 y entran en juego los **Tensor Cores** de la A100, diseñados específicamente para acelerar este tipo de operaciones.

3. **Duplicar el Batch Size (Aprovechar la memoria libre):**  
   Al consumir menos memoria con BF16, podemos subir el batch size a 32 sin quedarnos sin VRAM. Esto ayuda a alimentar mejor la GPU con más trabajo simultáneo, alcanzando **191.48 muestras/s** (**3.55x** respecto al baseline).

4. **Compilación del modelo con `torch.compile`:**  
   Es la optimización final más potente. TorchInductor analiza el flujo de operaciones y las fusiona para ejecutarlas en bloque, evitando lecturas y escrituras intermedias en la memoria de la tarjeta. Con esto llegamos a **290.61 muestras/s**, consiguiendo una aceleración total de **5.38x** respecto a cómo empezamos.

5. **Sobre el MFU y los tiempos:**  
   * En FP32 alcanzamos casi un 60% de MFU real frente al límite de los núcleos FP32 (19.5 TFLOP/s), pero representa solo un 3.7% del total que puede dar la A100 si usamos Tensor Cores (312 TFLOP/s). Con el modelo compilado y en BF16 conseguimos exprimir un **20% de MFU real de la máquina completa**.
   * Respecto a los tiempos de la Tabla 3, debido a la gran aceleración lograda en la etapa 5 (5.38x más rápido), los 500 pasos se completaron en solo **55.06 segundos** (0.92 min), mientras que las etapas más lentas superaron con holgura los 2 minutos.