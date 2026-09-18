"""
===============================================================================
 INFORME SEMANAL POR EMAIL — RENDIMIENTO DE CLASES DEL GIMNASIO
===============================================================================
Script standalone (NO depende de Streamlit), pensado para ejecutarse una vez
por semana — los viernes, cuando subes el CSV de esa semana a Drive — por
ejemplo con GitHub Actions (workflow en .github/workflows/informe-semanal.yml).

Qué hace:
    1. Descarga el CSV más reciente de la carpeta de Google Drive.
    2. Calcula las mismas métricas que el dashboard (% ocupación, tasa de
       cancelación).
    3. Construye un email en HTML con:
         - Ranking de ocupación por entrenador (semana en curso).
         - Comparativa por entrenador y clase: semana actual vs. semana
           anterior (mismo entrenador, misma clase).
         - Comportamiento mensual por entrenador y clase: mes en curso vs.
           mes anterior.
         - Comparativa interanual: mes en curso vs. mismo mes del año
           anterior (global y por entrenador).
    4. Lo envía por Gmail (SMTP con contraseña de aplicación).

Variables de entorno necesarias:
    GOOGLE_SERVICE_ACCOUNT_JSON  -> contenido COMPLETO del JSON de la cuenta
                                     de servicio, como texto (no una ruta)
    DRIVE_FOLDER_ID              -> ID de la carpeta de Drive con el CSV
    DRIVE_FILE_NAME              -> (opcional) nombre exacto del CSV; si se
                                     omite, coge el modificado más reciente
    GMAIL_USER                   -> email remitente, ej. pedrozunigav@gmail.com
    GMAIL_APP_PASSWORD           -> contraseña de aplicación de ese Gmail
                                     (Cuenta de Google > Seguridad >
                                     Verificación en 2 pasos > Contraseñas de
                                     aplicaciones — una cuenta de Gmail normal
                                     no deja usar su contraseña de acceso)
    EMAIL_TO                     -> destinatario(s), separados por coma

Probarlo en local:
    pip install pandas numpy google-api-python-client google-auth
    export GOOGLE_SERVICE_ACCOUNT_JSON="$(cat credenciales_drive.json)"
    export DRIVE_FOLDER_ID="1qIPq_nzde79ShZ-_syfIrD61FL7eV4W7"
    export GMAIL_USER="pedrozunigav@gmail.com"
    export GMAIL_APP_PASSWORD="xxxx xxxx xxxx xxxx"
    export EMAIL_TO="pedrozunigav@gmail.com"
    python informe_semanal.py
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
    "motivactions despegue",
    "presoterapia",
    "tour",
    "entrenamiento personal",
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
# 2. CÁLCULO DE MÉTRICAS (idéntico al del dashboard)
# ==============================================================================
def calcular_metricas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Reservas_Totales"] = df["Asistentes_Reales"] + df["Cancelaciones_Última_Hora"]

    # % Ocupación = Presentes / Inscritos * 100 (según los inscritos reales a
    # la clase, no la capacidad máxima de la sala). Si el CSV no trae la
    # columna "Inscritos" (p. ej. el dataset sintético de demo), se usa la
    # capacidad máxima como respaldo para no romper el resto del pipeline.
    if "Inscritos" in df.columns:
        df["Pct_Ocupacion"] = np.where(
            df["Inscritos"] > 0,
            (df["Asistentes_Reales"] / df["Inscritos"] * 100).round(2),
            0.0,
        )
    else:
        df["Pct_Ocupacion"] = (
            df["Asistentes_Reales"] / df["Capacidad_Máxima_Clase"] * 100
        ).round(2)

    df["Tasa_Cancelacion"] = np.where(
        df["Reservas_Totales"] > 0,
        (df["Cancelaciones_Última_Hora"] / df["Reservas_Totales"] * 100).round(2),
        0.0,
    )
    df["Año"] = df["Fecha_Hora"].dt.year
    df["Mes"] = df["Fecha_Hora"].dt.month
    df["Dia_Semana"] = df["Fecha_Hora"].dt.day_name()
    df["Hora"] = df["Fecha_Hora"].dt.hour
    iso = df["Fecha_Hora"].dt.isocalendar()
    df["Año_ISO"] = iso["year"]
    df["Semana_ISO"] = iso["week"]
    return df


# ==============================================================================
# 3. CONSTRUCCIÓN DEL INFORME SEMANAL
# ==============================================================================
def _fmt_pct(x) -> str:
    return f"{x:.1f} %" if pd.notna(x) else "—"


def _limites_semana(fecha_referencia: pd.Timestamp):
    """Devuelve (inicio, fin) de la semana ISO que contiene `fecha_referencia`."""
    inicio = fecha_referencia - pd.Timedelta(days=fecha_referencia.weekday())
    fin = inicio + pd.Timedelta(days=6, hours=23, minutes=59, seconds=59)
    return inicio, fin


def _tabla_html(encabezados, filas) -> str:
    """Genera una tabla HTML simple a partir de una lista de encabezados y filas."""
    ths = "".join(f"<th>{h}</th>" for h in encabezados)
    trs = "".join(
        "<tr>" + "".join(f"<td>{valor}</td>" for valor in fila) + "</tr>"
        for fila in filas
    )
    return f"""
    <table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse; width:100%; font-size:13px;">
        <tr style="background:#1f2937;color:#ffffff;">{ths}</tr>
        {trs}
    </table>
    """


# Paleta de colores para los gráficos, inspirada en el estilo de los
# informes que ya usas (barras de colores alternos, sin degradado).
_COLORES_GRAFICO = [
    "#38bdf8", "#5eead4", "#fde047", "#f97316", "#1f2937",
    "#fca5a5", "#fb7185", "#1e3a8a", "#0f172a", "#0ea5e9",
    "#2dd4bf", "#facc15", "#ef4444", "#57534e",
]


def _grafico_barras_horizontal(categorias, valores, titulo: str, xlabel: str = "") -> bytes:
    """
    Genera un PNG de barras horizontales tipo ranking (el valor más alto
    arriba), con una barra de color distinto por categoría.
    """
    alto = max(2.5, 0.4 * len(categorias) + 1)
    fig, ax = plt.subplots(figsize=(8, alto))
    posiciones = list(range(len(categorias)))
    colores = [_COLORES_GRAFICO[i % len(_COLORES_GRAFICO)] for i in posiciones]

    ax.barh(posiciones, valores, color=colores)
    ax.set_yticks(posiciones)
    ax.set_yticklabels(categorias)
    ax.invert_yaxis()  # el primero de la lista (más alto) queda arriba

    for i, valor in enumerate(valores):
        etiqueta = f"{valor:.0f}" if float(valor).is_integer() else f"{valor:.1f}"
        ax.text(valor, i, f" {etiqueta}", va="center", fontsize=9, color="#374151")

    ax.set_xlabel(xlabel)
    ax.set_title(titulo)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110)
    plt.close(fig)
    buffer.seek(0)
    return buffer.read()


def _grafico_barras_comparativo(
    categorias, valores_actual, valores_anterior, titulo: str,
    etiqueta_actual: str, etiqueta_anterior: str, ylabel: str = "% Ocupación",
) -> bytes:
    """Genera un PNG de barras verticales agrupadas: periodo actual vs. anterior."""
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(categorias) + 2), 4.5))
    x = np.arange(len(categorias))
    ancho = 0.35

    ax.bar(x - ancho / 2, valores_actual, width=ancho, label=etiqueta_actual, color="#2563eb")
    ax.bar(x + ancho / 2, valores_anterior, width=ancho, label=etiqueta_anterior, color="#93c5fd")

    ax.set_xticks(x)
    ax.set_xticklabels(categorias, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(titulo)
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110)
    plt.close(fig)
    buffer.seek(0)
    return buffer.read()


def _imagen_html(cid: str, alt: str) -> str:
    return f'<img src="cid:{cid}" alt="{alt}" style="max-width:100%; height:auto; margin: 8px 0;">'


def bloque_ranking_clases(df_semana_actual: pd.DataFrame, imagenes: dict) -> str:
    """Ranking de clases por asistencia total en la semana en curso (barras horizontales)."""
    if df_semana_actual.empty:
        return "<p>No hay clases registradas esta semana.</p>"

    resumen = (
        df_semana_actual.groupby("Nombre_Clase")["Asistentes_Reales"]
        .sum()
        .sort_values(ascending=False)
    )

    cid = "grafico_ranking_clases"
    imagenes[cid] = _grafico_barras_horizontal(
        list(resumen.index), list(resumen.values),
        "Ranking de clases por asistencia (esta semana)", xlabel="Asistentes",
    )
    return _imagen_html(cid, "Ranking de clases por asistencia")


def bloque_ranking_entrenadores(df_semana_actual: pd.DataFrame, imagenes: dict) -> str:
    """
    Ranking de ocupación media por entrenador, calculado sobre la semana en
    curso (la que se acaba de subir). De más a menos ocupación.
    """
    if df_semana_actual.empty:
        return "<p>No hay clases registradas esta semana.</p>"

    ranking = (
        df_semana_actual.groupby("Nombre_Monitor")
        .agg(
            Ocupacion_Media=("Pct_Ocupacion", "mean"),
            Asistencia_Total=("Asistentes_Reales", "sum"),
            Num_Clases=("Nombre_Monitor", "count"),
        )
        .round(1)
        .sort_values("Ocupacion_Media", ascending=False)
    )

    cid = "grafico_ranking_entrenadores"
    imagenes[cid] = _grafico_barras_horizontal(
        list(ranking.index), list(ranking["Ocupacion_Media"].values),
        "Ranking de ocupación por entrenador (esta semana)", xlabel="% Ocupación",
    )

    filas = [
        (i, fila.Index, _fmt_pct(fila.Ocupacion_Media), int(fila.Asistencia_Total), int(fila.Num_Clases))
        for i, fila in enumerate(ranking.itertuples(index=True), start=1)
    ]
    tabla = _tabla_html(
        ["#", "Entrenador", "% Ocupación media", "Asistencia total", "Clases impartidas"],
        filas,
    )
    return _imagen_html(cid, "Ranking de entrenadores") + tabla


def bloque_comparativa_wow_entrenador_clase(df: pd.DataFrame, fecha_referencia: pd.Timestamp, imagenes: dict) -> str:
    """
    Para cada combinación (entrenador, clase), compara la ocupación de la
    semana en curso frente a la misma combinación en la semana anterior.
    Solo entran combinaciones que tuvieron clase en la semana actual.
    También añade un gráfico agregado por entrenador (actual vs. anterior).
    """
    inicio_actual, fin_actual = _limites_semana(fecha_referencia)
    inicio_anterior, fin_anterior = _limites_semana(fecha_referencia - pd.Timedelta(days=7))

    df_actual = df[(df["Fecha_Hora"] >= inicio_actual) & (df["Fecha_Hora"] <= fin_actual)]
    df_anterior = df[(df["Fecha_Hora"] >= inicio_anterior) & (df["Fecha_Hora"] <= fin_anterior)]

    if df_actual.empty:
        return "<p>No hay clases registradas esta semana.</p>"

    # --- Gráfico agregado por entrenador ---------------------------------------
    resumen_actual_monitor = df_actual.groupby("Nombre_Monitor")["Pct_Ocupacion"].mean().round(1)
    resumen_anterior_monitor = df_anterior.groupby("Nombre_Monitor")["Pct_Ocupacion"].mean().round(1)
    monitores = sorted(resumen_actual_monitor.index, key=lambda m: resumen_actual_monitor[m], reverse=True)

    cid = "grafico_wow_entrenadores"
    imagenes[cid] = _grafico_barras_comparativo(
        monitores,
        [resumen_actual_monitor.get(m, 0) for m in monitores],
        [resumen_anterior_monitor.get(m, 0) for m in monitores],
        "Ocupación media por entrenador: esta semana vs. semana pasada",
        "Esta semana", "Semana pasada",
    )

    # --- Tabla de detalle por entrenador + clase --------------------------------
    resumen_actual = df_actual.groupby(["Nombre_Monitor", "Nombre_Clase"])["Pct_Ocupacion"].mean().round(1)
    resumen_anterior = df_anterior.groupby(["Nombre_Monitor", "Nombre_Clase"])["Pct_Ocupacion"].mean().round(1)

    filas = []
    for (monitor, clase), ocupacion_actual in resumen_actual.sort_values(ascending=False).items():
        ocupacion_anterior = resumen_anterior.get((monitor, clase), np.nan)
        if pd.notna(ocupacion_anterior):
            delta = round(ocupacion_actual - ocupacion_anterior, 1)
            flecha = "🔺" if delta >= 0 else "🔻"
            delta_txt = f"{flecha} {delta:+.1f} p.p."
        else:
            delta_txt = "— (sin datos la semana pasada)"
        filas.append((monitor, clase, _fmt_pct(ocupacion_actual), _fmt_pct(ocupacion_anterior), delta_txt))

    tabla = _tabla_html(
        ["Entrenador", "Clase", "% Ocup. esta semana", "% Ocup. semana pasada", "Variación"],
        filas,
    )
    return _imagen_html(cid, "Comparativa semanal por entrenador") + tabla


def bloque_comportamiento_mensual(df: pd.DataFrame, fecha_referencia: pd.Timestamp, imagenes: dict) -> str:
    """
    Para cada combinación (entrenador, clase), compara el mes en curso con
    el mes anterior (comportamiento a lo largo del mes), con un gráfico
    agregado por entrenador.
    """
    mes_actual, año_actual = fecha_referencia.month, fecha_referencia.year
    fecha_mes_anterior = (fecha_referencia.replace(day=1) - pd.Timedelta(days=1))
    mes_anterior, año_mes_anterior = fecha_mes_anterior.month, fecha_mes_anterior.year

    df_mes_actual = df[(df["Año"] == año_actual) & (df["Mes"] == mes_actual)]
    df_mes_anterior = df[(df["Año"] == año_mes_anterior) & (df["Mes"] == mes_anterior)]

    if df_mes_actual.empty:
        return "<p>No hay datos del mes en curso todavía.</p>"

    # --- Gráfico agregado por entrenador ---------------------------------------
    resumen_actual_monitor = df_mes_actual.groupby("Nombre_Monitor")["Pct_Ocupacion"].mean().round(1)
    resumen_anterior_monitor = df_mes_anterior.groupby("Nombre_Monitor")["Pct_Ocupacion"].mean().round(1)
    monitores = sorted(resumen_actual_monitor.index, key=lambda m: resumen_actual_monitor[m], reverse=True)

    nombre_mes_actual = fecha_referencia.strftime("%B").capitalize()
    cid = "grafico_mensual_entrenadores"
    imagenes[cid] = _grafico_barras_comparativo(
        monitores,
        [resumen_actual_monitor.get(m, 0) for m in monitores],
        [resumen_anterior_monitor.get(m, 0) for m in monitores],
        f"Ocupación media por entrenador: {nombre_mes_actual} vs. mes anterior",
        nombre_mes_actual, "Mes anterior",
    )

    # --- Tabla de detalle por entrenador + clase --------------------------------
    resumen_actual = (
        df_mes_actual.groupby(["Nombre_Monitor", "Nombre_Clase"])
        .agg(Ocupacion_Media=("Pct_Ocupacion", "mean"), Asistencia_Total=("Asistentes_Reales", "sum"))
        .round(1)
    )
    resumen_anterior = (
        df_mes_anterior.groupby(["Nombre_Monitor", "Nombre_Clase"])
        .agg(Ocupacion_Media=("Pct_Ocupacion", "mean"), Asistencia_Total=("Asistentes_Reales", "sum"))
        .round(1)
    )

    filas = []
    for (monitor, clase), fila_actual in resumen_actual.sort_values("Ocupacion_Media", ascending=False).iterrows():
        if (monitor, clase) in resumen_anterior.index:
            fila_anterior = resumen_anterior.loc[(monitor, clase)]
            delta = round(fila_actual.Ocupacion_Media - fila_anterior.Ocupacion_Media, 1)
            flecha = "🔺" if delta >= 0 else "🔻"
            delta_txt = f"{flecha} {delta:+.1f} p.p."
            ocup_anterior_txt = _fmt_pct(fila_anterior.Ocupacion_Media)
        else:
            delta_txt = "— (sin datos el mes pasado)"
            ocup_anterior_txt = "—"
        filas.append((
            monitor, clase,
            _fmt_pct(fila_actual.Ocupacion_Media), int(fila_actual.Asistencia_Total),
            ocup_anterior_txt, delta_txt,
        ))

    titulo = f"<p><b>Mes en curso:</b> {nombre_mes_actual} {año_actual} &nbsp;vs.&nbsp; mes anterior</p>"
    tabla = _tabla_html(
        ["Entrenador", "Clase", "% Ocup. mes en curso", "Asistencia mes en curso",
         "% Ocup. mes anterior", "Variación"],
        filas,
    )
    return titulo + _imagen_html(cid, "Comportamiento mensual por entrenador") + tabla


def bloque_comparativa_yoy(df: pd.DataFrame, fecha_referencia: pd.Timestamp, imagenes: dict) -> str:
    """
    Compara el mes en curso con el mismo mes del año anterior, a nivel
    global y desglosado por entrenador, con gráfico comparativo.
    """
    mes_actual, año_actual = fecha_referencia.month, fecha_referencia.year
    año_anterior = año_actual - 1

    df_mes_actual = df[(df["Año"] == año_actual) & (df["Mes"] == mes_actual)]
    df_mes_anterior = df[(df["Año"] == año_anterior) & (df["Mes"] == mes_actual)]

    nombre_mes = fecha_referencia.strftime("%B").capitalize()

    if df_mes_anterior.empty:
        return f"<p>No hay datos de {nombre_mes} {año_anterior} todavía para comparar.</p>"

    asistencia_actual = int(df_mes_actual["Asistentes_Reales"].sum())
    asistencia_anterior = int(df_mes_anterior["Asistentes_Reales"].sum())
    ocupacion_actual = df_mes_actual["Pct_Ocupacion"].mean() if len(df_mes_actual) else np.nan
    ocupacion_anterior = df_mes_anterior["Pct_Ocupacion"].mean()

    resumen_global = f"""
    <p><b>Asistencia total:</b> {asistencia_actual} ({nombre_mes} {año_actual})
       vs. {asistencia_anterior} ({nombre_mes} {año_anterior})<br>
       <b>Ocupación media:</b> {_fmt_pct(ocupacion_actual)} ({año_actual})
       vs. {_fmt_pct(ocupacion_anterior)} ({año_anterior})</p>
    """

    # --- Desglose y gráfico por entrenador --------------------------------------
    resumen_actual_monitor = df_mes_actual.groupby("Nombre_Monitor")["Pct_Ocupacion"].mean().round(1)
    resumen_anterior_monitor = df_mes_anterior.groupby("Nombre_Monitor")["Pct_Ocupacion"].mean().round(1)
    monitores = sorted(set(resumen_actual_monitor.index) | set(resumen_anterior_monitor.index))

    cid = "grafico_yoy_entrenadores"
    imagenes[cid] = _grafico_barras_comparativo(
        monitores,
        [resumen_actual_monitor.get(m, 0) for m in monitores],
        [resumen_anterior_monitor.get(m, 0) for m in monitores],
        f"Ocupación por entrenador — {nombre_mes}: {año_actual} vs. {año_anterior}",
        f"{nombre_mes} {año_actual}", f"{nombre_mes} {año_anterior}",
    )

    filas = []
    for monitor in monitores:
        ocup_actual = resumen_actual_monitor.get(monitor, np.nan)
        ocup_anterior = resumen_anterior_monitor.get(monitor, np.nan)
        if pd.notna(ocup_actual) and pd.notna(ocup_anterior):
            delta = round(ocup_actual - ocup_anterior, 1)
            flecha = "🔺" if delta >= 0 else "🔻"
            delta_txt = f"{flecha} {delta:+.1f} p.p."
        else:
            delta_txt = "—"
        filas.append((monitor, _fmt_pct(ocup_actual), _fmt_pct(ocup_anterior), delta_txt))

    tabla_monitor = _tabla_html(
        ["Entrenador", f"% Ocup. {nombre_mes} {año_actual}", f"% Ocup. {nombre_mes} {año_anterior}", "Variación"],
        filas,
    )
    return resumen_global + _imagen_html(cid, "Comparativa interanual por entrenador") + tabla_monitor


def construir_informe_html(df: pd.DataFrame, fecha_referencia: pd.Timestamp):
    """
    Ensambla el email semanal completo a partir de los cinco bloques.
    Devuelve (html, imagenes) — imagenes es un dict {cid: bytes_png} que
    hay que incrustar en el email junto al HTML.
    """
    imagenes: dict = {}

    inicio_actual, fin_actual = _limites_semana(fecha_referencia)
    df_semana_actual = df[(df["Fecha_Hora"] >= inicio_actual) & (df["Fecha_Hora"] <= fin_actual)]

    html = f"""
    <html>
    <body style="font-family: Arial, sans-serif; color:#1f2937;">
        <h2>🏋️ Informe semanal del gimnasio — semana del {inicio_actual.strftime('%d/%m')} al {fin_actual.strftime('%d/%m/%Y')}</h2>

        <h3>🥇 Ranking de clases por asistencia (esta semana)</h3>
        {bloque_ranking_clases(df_semana_actual, imagenes)}

        <h3>🏆 Ranking de ocupación por entrenador (esta semana)</h3>
        {bloque_ranking_entrenadores(df_semana_actual, imagenes)}

        <h3>📊 Comparativa por entrenador y clase — esta semana vs. semana pasada</h3>
        {bloque_comparativa_wow_entrenador_clase(df, fecha_referencia, imagenes)}

        <h3>📅 Comportamiento mensual por entrenador y clase</h3>
        {bloque_comportamiento_mensual(df, fecha_referencia, imagenes)}

        <h3>📆 Comparativa interanual (mismo mes, año anterior)</h3>
        {bloque_comparativa_yoy(df, fecha_referencia, imagenes)}

        <p style="color:#6b7280; font-size:12px;">Informe generado automáticamente.</p>
    </body>
    </html>
    """
    return html, imagenes


# ==============================================================================
# 4. ENVÍO DEL EMAIL (Gmail SMTP con contraseña de aplicación)
# ==============================================================================
def enviar_email(asunto: str, html: str, imagenes: dict) -> None:
    """
    Envía el email en HTML con las imágenes de `imagenes` ({cid: bytes_png})
    incrustadas inline (no como adjuntos sueltos), referenciadas en el HTML
    como <img src="cid:NOMBRE">.
    """
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
# 5. PUNTO DE ENTRADA
# ==============================================================================
def main() -> None:
    df_raw = cargar_datos()
    df = calcular_metricas(df_raw)

    fecha_referencia = pd.Timestamp(datetime.now().date())
    inicio_semana, fin_semana = _limites_semana(fecha_referencia)

    html, imagenes = construir_informe_html(df, fecha_referencia)
    asunto = (
        f"🏋️ Informe semanal del gimnasio — "
        f"{inicio_semana.strftime('%d/%m')} al {fin_semana.strftime('%d/%m/%Y')}"
    )

    enviar_email(asunto, html, imagenes)
    print(f"Email enviado correctamente a {os.environ['EMAIL_TO']}")


if __name__ == "__main__":
    main()
