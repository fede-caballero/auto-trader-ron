# 🚀 Manual de Vuelo: Auto-Trader (Distributed Karpathy Loop)

Bienvenido(a) a tu infraestructura de Inteligencia Artificial para el escaneo de mercados. Esta guía contiene todos los pasos operativos que necesitas desde arrancar en cero, hasta simular tu propio *Paper Trading*.

---

## 🛠️ Fase 1: Configuración Pre-Vuelo

Antes de lanzar el orquestador hacia los servidores de Vast.ai, asegúrate de tener dos cosas listas localmente:

1. **Variables de Entorno (`.env`)**:
   Asegúrate de que tu archivo `.env` contenga la referencia al "alias" de tu servidor SSH de Vast:
   ```env
   VAST_INSTANCE_ID="vast-a10"
   ```

2. **Dataset Vivo (`prepare.py`)**:
   La Inteligencia Artificial no puede entrenar sin datos actuales. Una vez a la semana, o justo antes de tu entrenamiento masivo, fuerza la descarga de las últimas 4,000 horas de mercado accionando:
   ```bash
   python3 predict.py
   ```
   *Nota:* Este script, además de predecir el futuro, invoca secretamente a `DataPreparer` para descargar de Binance las velas actualizadas y forjar las columnas matemáticas necesarias (OHLCV + ATR), guardándolas en tu carpeta local `data/`.

---

## 🧬 Fase 2: Lanzamiento Genético (Orquestador)

El "Distributed Karpathy Loop" es el motor que crea redes neuronales mutantes y las enfrenta al mercado para quedarse solo con la que sobrevive.

Para iniciar la evolución, ejecuta:
```bash
python3 karpathy_loop.py
```

Al iniciarse, el script:
1. Subirá (Sincronizará) automáticamente el dataset vivo local a Vast.ai de una sola vez.
2. Comenzará a inyectar variaciones de hiperparámetros (Batch Sizes, Learning Rates).
3. Entrenará **50 Generaciones** remotamente (durando cada una 15 minutos exactos).

---

## ⏸️ Fase 3: Pausar Entrenamiento (Ir a Casa / Sin Internet)

Dado que entrenar 50 generaciones completas tardará unas **12.5 horas**, no necesitas dejar tu computadora congelada allí.

1. **Matar el Proceso**: Puedes presionar `Ctrl + C` en terminal en ***cualquier momento***.
2. **Apagar Computadora**: Cierra tu laptop, corta el Wi-Fi, vete a casa. Tu progreso es **inmune** a esto.

### ¿Cómo funciona la inmunidad?
El Orquestador salva su estado maestro secretamente. Cada vez que descubre una capa genética superior en Vast.ai, sobrescribe los parámetros ganadores en el archivo local `models/train.py` y guarda la pérdida matemática récord (`BCE Loss`) en el archivo `.best_metric_cache`.

---

## ▶️ Fase 4: Reanudar Entrenamiento (Restaurando el Estado)

Al llegar a tu escritorio (o al día siguiente), asegúrate de que tu instancia de Vast.ai sigue encendida y ejecuta **literalmente** el mismo comando:
```bash
python3 karpathy_loop.py
```

Verás este hermoso mensaje en el Log confirmando la magia:
> `[INFO] Resuming genetic loop from previous best BCE Loss...`

El sistema retomará la optimización exactamente donde la mataste el día anterior, apuntando a seguir mejorando ese récord en las generaciones que resten.

---

## 🔮 Fase 5: Inferencia Manual (Predicción Única)

Para obtener una predicción instantánea del cerebro de la IA sin registrar nada:
```bash
python3 predict.py
```

---

## 📊 Fase 6: Paper Trading Automatizado

El script `paper_trader.py` es el corazón de tus pruebas con dinero ficticio. Cada vez que se ejecuta:

1. Consulta a la IA por la señal actual (LONG/SHORT) y su nivel de confianza.
2. Verifica si la predicción **anterior** fue correcta comparando con el precio real.
3. Gestiona posiciones abiertas (Stop Loss 5%, Take Profit 7%).
4. Abre nuevas posiciones solo si la confianza supera el 55%.
5. Registra **todo** en `logs/paper_trades.csv`.

### Comandos Disponibles:
```bash
# Ejecutar una ronda de paper trading
python3 paper_trader.py

# Ver el estado actual del portafolio
python3 paper_trader.py --status

# Resetear el portafolio a $10,000 iniciales
python3 paper_trader.py --reset
```

### Columnas del CSV (`logs/paper_trades.csv`):
| Columna | Descripción |
|---|---|
| `timestamp` | Fecha y hora UTC de la ejecución |
| `btc_price` | Precio actual de BTC/USDT |
| `signal` | Señal de la IA (LONG/SHORT) |
| `confidence` | Nivel de confianza (%) |
| `action` | Acción tomada (OPEN_LONG, CLOSE_SHORT, SKIP, etc.) |
| `pnl_trade` | Ganancia/pérdida de la operación cerrada |
| `balance` | Saldo total del portafolio |
| `prev_correct` | ¿La predicción anterior fue correcta? (True/False) |
| `win_rate` | Tasa de acierto acumulada (%) |

### Reglas de Riesgo (Sistema 3-5-7):
- **3%**: Se arriesga como máximo el 3% del capital por operación.
- **5%**: Stop Loss automático al -5% de la posición.
- **7%**: Take Profit automático al +7% de la posición.

---

## 🖥️ Fase 7: Despliegue en VPS (Automático 24/7)

Para que el Paper Trader corra automáticamente cada 4 horas sin necesidad de tu computadora:

### 1. Subir el proyecto al VPS
```bash
rsync -avz --exclude='venv/' --exclude='__pycache__/' \
  auto-trader/ usuario@tu-vps:/home/usuario/auto-trader/
```

### 2. Instalar dependencias en el VPS
```bash
ssh usuario@tu-vps
cd /home/usuario/auto-trader
python3 -m venv venv
source venv/bin/activate
pip install torch pandas numpy requests python-dotenv
```

### 3. Configurar Cron Job (cada 4 horas)
```bash
crontab -e
```
Añadir la siguiente línea:
```cron
0 */4 * * * cd /home/usuario/auto-trader && /home/usuario/auto-trader/venv/bin/python3 paper_trader.py >> /home/usuario/auto-trader/logs/cron.log 2>&1
```

### 4. Verificar que funciona
```bash
# Ejecutar manualmente la primera vez
python3 paper_trader.py

# Revisar el estado después de unas horas
python3 paper_trader.py --status

# Ver el CSV con todas las operaciones
cat logs/paper_trades.csv
```

> **Nota:** El VPS necesita acceso a internet para consultar la API de Binance. No necesita GPU ni acceso a Vast.ai.
