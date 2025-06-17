# Sistema de Aprendizaje Federado y Centralizado

Este repositorio contiene la implementación de un sistema de aprendizaje automático que permite comparar enfoques centralizados y federados bajo distintas condiciones de configuración y disponibilidad de datos.

El trabajo ha sido desarrollado en el contexto de un proyecto académico, con fines experimentales y de análisis.

Se dispone de una guía completa del sistema desarrollado generada con DeepWiki. Se puede acceder a esta guía online a través del siguiente enlace: https://deepwiki.com/rafcasceb/Federated-learning

---

# Características
* Realización de procesos de aprendizaje federado y aprendizaje centralizado.
* Comparación de resultados entre modelos centralizados y federados.
* Configuración flexible y personalizable.
* Métricas de evaluación detalladas (accuracy, precision, recall, F1 score, MCC).
* Gráficas de evolución de pérdida y rendimiento.
* Código modular, fácilmente extensible y reutilizable.

---

## Estructura del proyecto

El repositorio está organizado en los siguientes archivos y carpetas:

### Carpetas
* `/aggregated_models/`: (autogenerada) almacena los pesos de los modelos agregados por el servidor en el aprendizaje federado.
* `/data/`: contiene los datos locales utilizados por cada cliente y el nodo centralizado.
* `/logs/`: (autogenerada) guarda los registros de ejecución, resultados de los experimentos y otros ficheros temporales.
* `/plots/`: (autogenerada) contiene los gráficos generados durante la ejecución, tanto para visualizar propiedades de los datos como los resultados del aprendizaje.

### Archivos principales
* `centralized.py`: script para ejecutar el sistema en modo de aprendizaje centralizado.
* `client_app.py`: lógica común a todos los clientes del aprendizaje federado.
* `client_1.py`, `client_2.py`, etc.: scripts para lanzar cada cliente de forma individual, ejecutan `client_app.py` con su respectivo identificador.
* `client_manager.py`: herramienta para gestionar automáticamente la ejecución de múltiples clientes (inicio, seguimiento y parada).
* `model.py`: definición del modelo de red neuronal compartido entre clientes, servidor y nodo centralizado.
* `strategy.py`: implementación de la estrategia de agregación del servidor federado.
* `server.py`: script que ejecuta el servidor central del sistema federado.
* `task.py`: contiene utilidades auxiliares como gestión de logs, generación de gráficos y manejo del contexto de ejecución.

### Archivos de configuración
* `config.yaml`: archivo principal de configuración. Permite definir los parámetros de ejecución para los experimentos, como el número de rondas, la estrategia de agregación, los modelos a utilizar, el número de clientes, etc. Puede modificarse para ajustar fácilmente el comportamiento del sistema sin necesidad de cambiar el código.
* `requirements.txt`: lista de dependencias necesarias para ejecutar el proyecto.

---

## Requisitos

Las dependencias necesarias para ejecutar este proyecto se encuentran en el archivo `requirements.txt`.

Se recomienda trabajar dentro de un entorno virtual para evitar conflictos con otras instalaciones de Python. A continuación se muestra cómo crear y activar un entorno virtual:

**En Windows:**

```bash
python -m venv venv          # Crear entorno
venv\Scripts\activate        # Activar entorno
deactivate                   # Desactivar entorno
```

**En Linux (Ubuntu) o macOS:**

```bash
python -m venv venv          # Crear entorno
source venv/bin/activate     # Activar entorno
deactivate                   # Desactivar entorno
```

Una vez activado el entorno, instalar las dependencias con:

```bash
pip install -r requirements.txt
```

---

## Configuración del entorno

El comportamiento del sistema puede ajustarse mediante el archivo de configuración `config.yaml`, donde se definen aspectos como el número de rondas de entrenamiento, el número de clientes o el modo de ejecución.

El socket del servidor se ha de configurar en el mismo archivo, determinando una IP privada y un puerto libre. Para conocer la IP privada se puede utilizar:

Windows:
```bash
ipconfig
```

Ubuntu (Linux):
```bash
hostname -I
```


---

## Ejecución del aprendizaje federado

El sistema federado puede ejecutarse de distintas formas, según el nivel de control deseado.

> Nota: Aunque el número de clientes es configurable, la base de datos incluida en la carpeta `data/` está actualmente particionada para cuatro clientes. Para usar más, será necesario adaptar las particiones manualmente. Para usar menos, no.


### Opción 1: Manual
**_(Cada uno en una terminal independiente)_**

Iniciar primero el servidor y después cada cliente en una terminal independiente:

```bash
python server.py
python client_1.py
python client_2.py
python client_3.py
python client_4.py
```

### Opción 2: Lanzamiento automático de clientes
**_(Cada uno en una terminal independiente)_**

Ejecutar el servidor y utilizar `client_manager.py` para iniciar múltiples clientes automáticamente:

```bash
python server.py
python client_manager.py start <NUM_CLIENTS>
```

Aparte, también se cuenta con dos comandos adicionales para listar y detener clientes:
```bash
python client_manager.py list 
python client_manager.py stop 
```

### Opción 3: Ejecución de experimento completo

Ejecutar un experimento completo con configuración predefinida. La configuración del experimento se puede modificar al inicio del propio archivo `experiment.py`:

```bash
python experiment.py
```

Para lanzar experimentos de manera reproducible, usar:

```bash
python experiment.py --test
```

Los comandos para listar y detener `client_manager.py` también son efectuables aquí.


---

## Ejecución del aprendizaje centralizado

Para lanzar el sistema en modo centralizado:

```bash
python centralized.py
```

---

## Licencia

Este proyecto ha sido desarrollado con fines académicos. Para cualquier otro uso, por favor contacte con el autor.

---

## Referencia

El sistema ha sido desarrollado utilizando el framework Flower como base para el aprendizaje federado:

```bibtex
@article{beutel2020flower,
  title={Flower: A Friendly Federated Learning Research Framework},
  author={Beutel, Daniel J and Topal, Taner and Mathur, Akhil and Qiu, Xinchi and Fernandez-Marques, Javier and Gao, Yan and Sani, Lorenzo and Kwing, Hei Li and Parcollet, Titouan and Gusmão, Pedro PB de and Lane, Nicholas D},
  journal={arXiv preprint arXiv:2007.14390},
  year={2020}
}
```