"""
===============================================================================
 INFORME DE TENDENCIA SEMANAL — % OCUPACIÓN POR ENTRENADOR Y ACTIVIDAD
===============================================================================
Script standalone (NO depende de Streamlit), pensado para ejecutarse una vez
por semana — los LUNES a las 16:00, analizando la semana que acaba de
terminar (viernes-domingo incluidos) — con GitHub Actions (workflow en
.github/workflows/informe-tendencia-semanal.yml). Es un email SEPARADO del
informe de los viernes (informe_semanal.py); no lo sustituye.

Qué hace:
    1. Descarga el CSV más reciente de la carpeta de Google Drive (misma
       fuente que el resto de informes).
    2. Para cada entrenador con actividades en la semana que acaba de
       terminar, dibuja un gráfico de líneas: una línea por cada actividad
       que imparte (% ocupación semana a semana, últimas ~12 semanas) más
       una línea de "Promedio general" de ese entrenador.
    3. Envía un email con un gráfico por entrenador.

Variables de entorno necesarias: las mismas 5 que informe_semanal.py
(GOOGLE_SERVICE_ACCOUNT_JSON, DRIVE_FOLDER_ID, DRIVE_FILE_NAME opcional,
GMAIL_USER, GMAIL_APP_PASSWORD, EMAIL_TO).

Probarlo en local:
    pip install pandas numpy matplotlib google-api-python-client google-auth
    export GOOGLE_SERVICE_ACCOUNT_JSON="$(cat credenciales_drive.json)"
    export DRIVE_FOLDER_ID="1qIPq_nzde79ShZ-_syfIrD61FL7eV4W7"
    export GMAIL_USER="pedrozunigav@gmail.com"
    export GMAIL_APP_PASSWORD="xxxx xxxx xxxx xxxx"
    export EMAIL_TO="pedrozunigav@gmail.com"
    python informe_tendencia_semanal.py
===============================================================================
"""

import io
import json
import os
import smtplib
from datetime import datetime, timedelta
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import matplotlib
matplotlib.use("Agg")  # backend sin pantalla: necesario para generar PNGs en un servidor/CI
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

SCOPES_DRIVE = ["https://www.googleapis.com/auth/drive.readonly"]

COLUMNAS_ESPERADAS = {
    "Fecha_Hora",
    "Nombre_Clase",
    "Nombre_Monitor",
    "Capacidad_Máxima_Clase",
    "Asistentes_Reales",
    "Cancelaciones_Última_Hora",
}

# Tu CSV real viene de un sistema de reservas con sus propios nombres de
# columna (en vez del esquema genérico de ejemplo). Este mapeo traduce esos
# nombres "de origen" a los canónicos que usa el resto del pipeline, así no
# hace falta tocar el CSV en Drive cada vez que se exporta.
MAPEO_COLUMNAS_ALTERNATIVAS = {
    "Fecha de inicio del curso": "Fecha_Hora",
    "Número de plazas": "Capacidad_Máxima_Clase",
    "Actividad": "Nombre_Clase",
    "Presente": "Asistentes_Reales",
    "Ausente": "Cancelaciones_Última_Hora",
    "ID del entrenador": "ID_Monitor",
    "Inscrito": "Inscritos",
}

# Actividades que NO son clases dirigidas normales (tours, entrenamiento
# personal, etc.) y que se excluyen de todos los análisis y gráficos.
# Edita esta lista si cambian los nombres exactos en el CSV.
ACTIVIDADES_EXCLUIDAS = {
    "motivaction be ready",
    "motivaction despegue",
    "motivaction impulso",
    "presoterapia",
    "tour",
    "entrenamiento personal",
    "spinergie virtual",
}


# ==============================================================================
# 1. CARGA DE DATOS DESDE DRIVE (misma lógica que el dashboard, sin Streamlit)
# ==============================================================================
def obtener_servicio_drive():
    """Autentica con la cuenta de servicio a partir del JSON en una variable de entorno."""
    info = json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"])
    credenciales = service_account.Credentials.from_service_account_info(
        info, scopes=SCOPES_DRIVE
    )
    return build("drive", "v3", credentials=credenciales, cache_discovery=False)


def buscar_csv_en_carpeta(servicio, carpeta_id: str, nombre_archivo: str = ""):
    """Localiza el CSV en la carpeta de Drive (el más reciente si no se da nombre)."""
    query = f"'{carpeta_id}' in parents and trashed = false and mimeType = 'text/csv'"
    if nombre_archivo:
        nombre_escapado = nombre_archivo.replace("'", "\\'")
        query += f" and name = '{nombre_escapado}'"

    resultados = servicio.files().list(
        q=query,
        fields="files(id, name, modifiedTime)",
        orderBy="modifiedTime desc",
        pageSize=10,
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()

    archivos = resultados.get("files", [])
    if not archivos:
        detalle = f" con nombre '{nombre_archivo}'" if nombre_archivo else ""
        raise FileNotFoundError(f"No se encontró ningún CSV{detalle} en la carpeta '{carpeta_id}'.")
    return archivos[0]


def descargar_csv_drive(servicio, file_id: str) -> io.BytesIO:
    """Descarga el CSV de Drive a un buffer en memoria."""
    peticion = servicio.files().get_media(fileId=file_id)
    buffer = io.BytesIO()
    descargador = MediaIoBaseDownload(buffer, peticion)
    completado = False
    while not completado:
        _, completado = descargador.next_chunk()
    buffer.seek(0)
    return buffer


def normalizar_columnas_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Traduce las columnas reales del CSV (si vienen del sistema de reservas)
    a los nombres canónicos, valida que no falte nada, parsea fechas y
    rellena IDs si no vienen en el CSV.
    """
    df = df.copy()
    df.columns = [c.strip().strip('"') for c in df.columns]

    # 1) Traducir columnas "de origen" -> nombres canónicos, solo si el
    #    nombre canónico no viene ya puesto en el CSV.
    columnas_a_renombrar = {
        origen: destino
        for origen, destino in MAPEO_COLUMNAS_ALTERNATIVAS.items()
        if origen in df.columns and destino not in df.columns
    }
    if columnas_a_renombrar:
        df = df.rename(columns=columnas_a_renombrar)

    # 2) El nombre del monitor puede venir partido en nombre + apellidos.
    if "Nombre_Monitor" not in df.columns:
        if "Nombre del entrenador" in df.columns and "Apellidos del entrenador" in df.columns:
            df["Nombre_Monitor"] = (
                df["Nombre del entrenador"].fillna("").astype(str).str.strip()
                + " "
                + df["Apellidos del entrenador"].fillna("").astype(str).str.strip()
            ).str.strip()
        elif "Nombre del entrenador" in df.columns:
            df["Nombre_Monitor"] = df["Nombre del entrenador"]

    faltantes = COLUMNAS_ESPERADAS - set(df.columns)
    if faltantes:
        raise ValueError(
            f"Al CSV le faltan columnas obligatorias: {sorted(faltantes)}. "
            f"Columnas encontradas: {sorted(df.columns)}"
        )

    # dayfirst=True porque las fechas del sistema de reservas vienen en
    # formato día/mes/año, como es habitual en España.
    df["Fecha_Hora"] = pd.to_datetime(df["Fecha_Hora"], errors="raise", dayfirst=True)

    if "ID_Clase" not in df.columns:
        df["ID_Clase"] = df["Nombre_Clase"].astype("category").cat.codes + 1
    if "ID_Monitor" not in df.columns:
        df["ID_Monitor"] = df["Nombre_Monitor"].astype("category").cat.codes + 1

    for col in ["Capacidad_Máxima_Clase", "Asistentes_Reales", "Cancelaciones_Última_Hora"]:
        df[col] = pd.to_numeric(df[col], errors="raise")

    # "Inscritos" es opcional: solo viene en el CSV real (no en el dataset
    # sintético de demo). Si está, se usa para calcular % Ocupación.
    if "Inscritos" in df.columns:
        df["Inscritos"] = pd.to_numeric(df["Inscritos"], errors="coerce")

    # 3) Quitar actividades que no son clases dirigidas normales (tours,
    #    entrenamiento personal, etc. — ver ACTIVIDADES_EXCLUIDAS arriba).
    nombre_normalizado = df["Nombre_Clase"].astype(str).str.strip().str.lower()
    df = df[~nombre_normalizado.isin(ACTIVIDADES_EXCLUIDAS)]

    return df.sort_values("Fecha_Hora").reset_index(drop=True)


def leer_csv_desde_buffer(buffer: io.BytesIO) -> pd.DataFrame:
    """
    Lee un CSV probando varias codificaciones habituales (UTF-8 con y sin
    BOM, Windows-1252, Latin-1) y detectando automáticamente el separador
    (coma o punto y coma, típico de exportaciones de Excel en español).
    """
    codificaciones = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]
    ultimo_error = None
    for codificacion in codificaciones:
        try:
            buffer.seek(0)
            # sep=None + engine="python" activa la detección automática del
            # separador (csv.Sniffer). Si detecta mal y deja todo en una
            # sola columna, reintentamos forzando ";", el más habitual en
            # CSVs de Excel en español.
            df = pd.read_csv(buffer, encoding=codificacion, sep=None, engine="python")
            if df.shape[1] == 1:
                buffer.seek(0)
                df_punto_coma = pd.read_csv(buffer, encoding=codificacion, sep=";")
                if df_punto_coma.shape[1] > 1:
                    df = df_punto_coma
            return df
        except UnicodeDecodeError as error:
            ultimo_error = error
            continue
    raise ultimo_error


def cargar_datos() -> pd.DataFrame:
    """Pipeline completo de carga: autenticar -> localizar -> descargar -> normalizar."""
    carpeta_id = os.environ["DRIVE_FOLDER_ID"]
    nombre_archivo = os.environ.get("DRIVE_FILE_NAME", "")

    servicio = obtener_servicio_drive()
    archivo = buscar_csv_en_carpeta(servicio, carpeta_id, nombre_archivo)
    buffer = descargar_csv_drive(servicio, archivo["id"])
    df = leer_csv_desde_buffer(buffer)
    return normalizar_columnas_csv(df)


# ==============================================================================
# 2. CÁLCULO DE MÉTRICAS
# ==============================================================================
def calcular_metricas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Reservas_Totales"] = df["Asistentes_Reales"] + df["Cancelaciones_Última_Hora"]

    # % Ocupación = Presentes / Plazas disponibles * 100. np.where evita
    # división por cero cuando Capacidad_Máxima_Clase viene a 0 o vacía.
    df["Pct_Ocupacion"] = np.where(
        df["Capacidad_Máxima_Clase"] > 0,
        (df["Asistentes_Reales"] / df["Capacidad_Máxima_Clase"] * 100).round(2),
        np.nan,
    )
    df["Tasa_Cancelacion"] = np.where(
        df["Reservas_Totales"] > 0,
        (df["Cancelaciones_Última_Hora"] / df["Reservas_Totales"] * 100).round(2),
        0.0,
    )

    # Inicio de la semana ISO (lunes) de cada sesión — clave para agrupar
    # la tendencia semana a semana.
    df["Semana_Inicio"] = (df["Fecha_Hora"] - pd.to_timedelta(df["Fecha_Hora"].dt.weekday, unit="D")).dt.normalize()

    return df


# ==============================================================================
# 3. GRÁFICOS DE LÍNEAS: % OCUPACIÓN SEMANAL POR ENTRENADOR Y ACTIVIDAD
# ==============================================================================
_COLORES_LINEAS = [
    "#38bdf8", "#f97316", "#22c55e", "#a855f7", "#ef4444",
    "#eab308", "#06b6d4", "#ec4899", "#84cc16", "#6366f1",
    "#fb7185", "#14b8a6", "#f59e0b", "#8b5cf6",
]


def grafico_lineas_monitor(datos_monitor: pd.DataFrame, monitor: str, semanas_ventana: int = 12) -> bytes:
    """
    Gráfico de líneas: % ocupación semana a semana, una línea por cada
    actividad que imparte el monitor, más una línea de "Promedio general"
    (más gruesa y discontinua). Se muestran las últimas `semanas_ventana`
    semanas con datos, para que el gráfico no se sature con todo el
    histórico.
    """
    pivote = datos_monitor.pivot_table(
        index="Semana_Inicio", columns="Nombre_Clase", values="Pct_Ocupacion", aggfunc="mean"
    ).sort_index()
    promedio = datos_monitor.groupby("Semana_Inicio")["Pct_Ocupacion"].mean().sort_index()

    if len(pivote) > semanas_ventana:
        pivote = pivote.tail(semanas_ventana)
        promedio = promedio.tail(semanas_ventana)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, actividad in enumerate(pivote.columns):
        serie = pivote[actividad]
        ax.plot(
            serie.index, serie.values, marker="o", markersize=3, linewidth=1.6,
            label=actividad, color=_COLORES_LINEAS[i % len(_COLORES_LINEAS)],
        )
    ax.plot(
        promedio.index, promedio.values, linewidth=2.8, linestyle="--",
        color="#0f172a", label="Promedio general",
    )

    ax.set_title(f"% Ocupación semanal — {monitor}", fontsize=13)
    ax.set_ylabel("% Ocupación")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    fig.autofmt_xdate(rotation=30)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110)
    plt.close(fig)
    buffer.seek(0)
    return buffer.read()


# ==============================================================================
# 4. CONSTRUCCIÓN DEL EMAIL
# ==============================================================================
def _imagen_html(cid: str, alt: str) -> str:
    return f'<img src="cid:{cid}" alt="{alt}" style="max-width:100%; height:auto; margin: 6px 0 22px 0;">'


def construir_informe_html(df: pd.DataFrame, fecha_referencia: pd.Timestamp):
    """
    fecha_referencia es el LUNES en que se envía el email. La semana que se
    analiza es la que acaba de terminar: los 7 días anteriores a ese lunes.
    Devuelve (html, imagenes).
    """
    inicio_semana_objetivo = (fecha_referencia - pd.Timedelta(days=7)).normalize()
    fin_semana_objetivo = (fecha_referencia - pd.Timedelta(days=1)).normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)

    df_semana_objetivo = df[
        (df["Fecha_Hora"] >= inicio_semana_objetivo) & (df["Fecha_Hora"] <= fin_semana_objetivo)
    ]
    # Muy importante (igual que en el informe de los viernes): solo se
    # consideran los monitores que dieron clase en la semana que se analiza.
    monitores_activos = sorted(df_semana_objetivo["Nombre_Monitor"].dropna().unique())

    imagenes: dict = {}
    bloques_monitor = []

    if not monitores_activos:
        bloques_monitor.append("<p>No hay entrenadores con actividades registradas en la semana analizada.</p>")
    else:
        # Solo se usa el histórico HASTA el final de la semana analizada,
        # para que el gráfico no incluya datos de días posteriores al envío.
        df_historico = df[df["Fecha_Hora"] <= fin_semana_objetivo]
        for i, monitor in enumerate(monitores_activos):
            datos_monitor = df_historico[df_historico["Nombre_Monitor"] == monitor]
            cid = f"grafico_tendencia_{i}"
            imagenes[cid] = grafico_lineas_monitor(datos_monitor, monitor)
            bloques_monitor.append(f"<h3 style='margin-top:26px;'>{monitor}</h3>" + _imagen_html(cid, f"Tendencia semanal — {monitor}"))

    html = f"""
    <html>
    <body style="font-family: Arial, sans-serif; color:#1f2937;">
        <h2>📈 Tendencia semanal de ocupación por entrenador</h2>
        <p>Semana analizada: {inicio_semana_objetivo.strftime('%d/%m')} al {fin_semana_objetivo.strftime('%d/%m/%Y')}</p>
        <p>
            Para cada entrenador, la evolución del % de ocupación semana a semana en cada una de
            sus actividades dirigidas, junto con su promedio general (línea discontinua).
        </p>

        {''.join(bloques_monitor)}

        <p style="margin-top:28px;">Un cordial saludo,</p>
        <p>
            <b>Pedro Zuñiga Vergara</b><br>
            Fitness Manager<br>
            Energie Fitness Sant Cugat
        </p>
    </body>
    </html>
    """
    return html, imagenes


# ==============================================================================
# 5. ENVÍO DEL EMAIL
# ==============================================================================
def enviar_email(asunto: str, html: str, imagenes: dict) -> None:
    gmail_user = os.environ["GMAIL_USER"]
    gmail_password = os.environ["GMAIL_APP_PASSWORD"]
    destinatarios = [d.strip() for d in os.environ["EMAIL_TO"].split(",") if d.strip()]

    mensaje = MIMEMultipart("related")
    mensaje["Subject"] = asunto
    mensaje["From"] = gmail_user
    mensaje["To"] = ", ".join(destinatarios)

    parte_alternativa = MIMEMultipart("alternative")
    parte_alternativa.attach(MIMEText(html, "html", "utf-8"))
    mensaje.attach(parte_alternativa)

    for cid, contenido_png in imagenes.items():
        imagen = MIMEImage(contenido_png)
        imagen.add_header("Content-ID", f"<{cid}>")
        imagen.add_header("Content-Disposition", "inline", filename=f"{cid}.png")
        mensaje.attach(imagen)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as servidor:
        servidor.login(gmail_user, gmail_password)
        servidor.sendmail(gmail_user, destinatarios, mensaje.as_string())


# ==============================================================================
# 6. PUNTO DE ENTRADA
# ==============================================================================
def main() -> None:
    df_raw = cargar_datos()
    df = calcular_metricas(df_raw)
    df = df[np.isfinite(df["Pct_Ocupacion"])].copy()

    fecha_referencia = pd.Timestamp(datetime.now().date())  # el lunes en que corre el workflow
    inicio_semana_objetivo = (fecha_referencia - pd.Timedelta(days=7)).normalize()
    fin_semana_objetivo = (fecha_referencia - pd.Timedelta(days=1)).normalize()

    html, imagenes = construir_informe_html(df, fecha_referencia)
    asunto = (
        f"📈 Tendencia semanal de ocupación — "
        f"{inicio_semana_objetivo.strftime('%d/%m')} al {fin_semana_objetivo.strftime('%d/%m/%Y')}"
    )

    enviar_email(asunto, html, imagenes)
    print(f"Email enviado correctamente a {os.environ['EMAIL_TO']}")


if __name__ == "__main__":
    main()
