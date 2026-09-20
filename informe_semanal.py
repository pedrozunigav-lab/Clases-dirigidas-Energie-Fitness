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
# 2. CÁLCULO DE MÉTRICAS (idéntico al del dashboard)
# ==============================================================================
def calcular_metricas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Reservas_Totales"] = df["Asistentes_Reales"] + df["Cancelaciones_Última_Hora"]

    # % Ocupación = Presentes / Plazas disponibles (capacidad máxima) * 100
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


MESES_ES = {
    1: "enero", 2: "febrero", 3: "marzo", 4: "abril", 5: "mayo", 6: "junio",
    7: "julio", 8: "agosto", 9: "septiembre", 10: "octubre", 11: "noviembre", 12: "diciembre",
}


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


def _imagen_html(cid: str, alt: str) -> str:
    return f'<img src="cid:{cid}" alt="{alt}" style="max-width:100%; height:auto; margin: 8px 0;">'


def _caption_html(formula: str) -> str:
    """Subtítulo pequeño, en cursiva, que indica qué fórmula/parámetros usa el gráfico."""
    return f'<p style="color:#6b7280; font-size:12px; margin: 0 0 4px 0;"><i>{formula}</i></p>'


def _grafico_barras_horizontal_agrupado(categorias, series: dict, titulo: str, xlabel: str = "") -> bytes:
    """
    Barras horizontales agrupadas: para cada categoría (p. ej. un monitor),
    dibuja una barra por cada serie de `series` (dict {nombre_serie: lista
    de valores, en el mismo orden que `categorias`}).
    """
    n_series = len(series)
    n_categorias = len(categorias)
    alto = max(3, 0.5 * n_categorias + 1.5)
    fig, ax = plt.subplots(figsize=(9, alto))

    y = np.arange(n_categorias)
    alto_barra = 0.8 / max(n_series, 1)
    colores = ["#93c5fd", "#3b82f6", "#1e3a8a", "#0f172a"]

    for i, (nombre_serie, valores) in enumerate(series.items()):
        offset = (i - (n_series - 1) / 2) * alto_barra
        ax.barh(y + offset, valores, height=alto_barra, label=nombre_serie, color=colores[i % len(colores)])

    ax.set_yticks(y)
    ax.set_yticklabels(categorias)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(titulo)
    ax.legend(loc="lower right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110)
    plt.close(fig)
    buffer.seek(0)
    return buffer.read()


def bloque_ranking_clases(df_semana_actual: pd.DataFrame, imagenes: dict) -> str:
    """
    Ranking de clases por asistencia media por sesión en la semana en curso
    (presentes totales ÷ nº de sesiones de esa actividad esa semana), en
    barras horizontales, solo imagen.
    """
    if df_semana_actual.empty:
        return "<p>No hay clases registradas esta semana.</p>"

    resumen_sesiones = df_semana_actual.groupby("Nombre_Clase").agg(
        Presentes_Totales=("Asistentes_Reales", "sum"),
        Num_Sesiones=("Nombre_Clase", "count"),
    )
    resumen = (
        (resumen_sesiones["Presentes_Totales"] / resumen_sesiones["Num_Sesiones"])
        .round(1)
        .sort_values(ascending=False)
    )

    cid = "grafico_ranking_clases"
    imagenes[cid] = _grafico_barras_horizontal(
        list(resumen.index), list(resumen.values),
        "Ranking de clases por asistencia (esta semana)", xlabel="Asistentes por sesión",
    )
    caption = _caption_html("Asistencia por sesión = Presentes totales ÷ Nº de sesiones (esta semana)")
    return caption + _imagen_html(cid, "Ranking de clases por asistencia")


def bloque_ranking_entrenadores(df_semana_actual: pd.DataFrame, imagenes: dict) -> str:
    """
    Ranking de instructores por % de ocupación media, calculado sobre la
    semana en curso (la que se acaba de subir), de más a menos. Solo
    imagen — sin tabla ni otros indicadores.
    """
    if df_semana_actual.empty:
        return "<p>No hay clases registradas esta semana.</p>"

    ranking = (
        df_semana_actual.groupby("Nombre_Monitor")["Pct_Ocupacion"]
        .mean()
        .round(1)
        .sort_values(ascending=False)
    )

    cid = "grafico_ranking_entrenadores"
    imagenes[cid] = _grafico_barras_horizontal(
        list(ranking.index), list(ranking.values),
        "Ranking de instructores por % de ocupación (esta semana)", xlabel="% Ocupación",
    )
    caption = _caption_html("% Ocupación = Presentes ÷ Plazas disponibles × 100")
    return caption + _imagen_html(cid, "Ranking de instructores por % de ocupación")


def _promedio_ocupacion_monitor(df: pd.DataFrame, monitor: str, condicion) -> float:
    """Media de % Ocupación para un monitor concreto, sobre el subconjunto df[condicion]."""
    subconjunto = df[condicion & (df["Nombre_Monitor"] == monitor)]
    return subconjunto["Pct_Ocupacion"].mean() if len(subconjunto) else np.nan


def bloque_comparativa_monitor_combinada(
    df: pd.DataFrame, fecha_referencia: pd.Timestamp, monitores_activos: list, imagenes: dict,
) -> str:
    """
    Un único gráfico horizontal con tres barras por monitor: % ocupación
    media del mes en curso, del mes anterior y del mismo mes del año
    anterior. Solo imagen, sin tabla. Incluye únicamente a los monitores con
    actividades no excluidas en la semana en curso (`monitores_activos`).
    """
    if not monitores_activos:
        return "<p>No hay monitores con clases dirigidas esta semana.</p>"

    mes_actual, año_actual = fecha_referencia.month, fecha_referencia.year
    fecha_mes_anterior = fecha_referencia.replace(day=1) - pd.Timedelta(days=1)
    mes_anterior_num, año_mes_anterior = fecha_mes_anterior.month, fecha_mes_anterior.year
    año_anterior = año_actual - 1

    cond_mes_actual = (df["Año"] == año_actual) & (df["Mes"] == mes_actual)
    cond_mes_anterior = (df["Año"] == año_mes_anterior) & (df["Mes"] == mes_anterior_num)
    cond_año_anterior = (df["Año"] == año_anterior) & (df["Mes"] == mes_actual)

    valores_mes_actual = {m: _promedio_ocupacion_monitor(df, m, cond_mes_actual) for m in monitores_activos}
    monitores_ordenados = sorted(
        monitores_activos,
        key=lambda m: valores_mes_actual[m] if pd.notna(valores_mes_actual[m]) else -1,
        reverse=True,
    )

    serie_mes_actual = [
        valores_mes_actual[m] if pd.notna(valores_mes_actual[m]) else 0 for m in monitores_ordenados
    ]
    serie_mes_anterior = [
        (lambda v: v if pd.notna(v) else 0)(_promedio_ocupacion_monitor(df, m, cond_mes_anterior))
        for m in monitores_ordenados
    ]
    serie_año_anterior = [
        (lambda v: v if pd.notna(v) else 0)(_promedio_ocupacion_monitor(df, m, cond_año_anterior))
        for m in monitores_ordenados
    ]

    nombre_mes_actual = MESES_ES[mes_actual].capitalize()
    nombre_mes_anterior = MESES_ES[mes_anterior_num].capitalize()

    cid = "grafico_comparativa_monitor"
    imagenes[cid] = _grafico_barras_horizontal_agrupado(
        monitores_ordenados,
        {
            f"{nombre_mes_actual} {año_actual} (mes actual)": serie_mes_actual,
            f"{nombre_mes_anterior} {año_mes_anterior} (mes anterior)": serie_mes_anterior,
            f"{nombre_mes_actual} {año_anterior} (año anterior)": serie_año_anterior,
        },
        "% Ocupación por monitor — mes actual, mes anterior y año anterior",
        xlabel="% Ocupación",
    )
    caption = _caption_html("% Ocupación = Presentes ÷ Plazas disponibles × 100")
    return caption + _imagen_html(cid, "Comparativa por monitor: mes actual, mes anterior y año anterior")


def bloque_comparativa_semanal_actividades(df: pd.DataFrame, fecha_referencia: pd.Timestamp) -> str:
    """
    Tabla: % de ocupación por monitor + actividad (ya excluidas las no
    deseadas), semana anterior vs. semana actual, con las fechas de cada
    semana y la variación en puntos porcentuales. Ordenada por monitor y,
    dentro de cada monitor, por ocupación de esta semana descendente.
    """
    inicio_actual, fin_actual = _limites_semana(fecha_referencia)
    inicio_anterior, fin_anterior = _limites_semana(fecha_referencia - pd.Timedelta(days=7))

    df_actual = df[(df["Fecha_Hora"] >= inicio_actual) & (df["Fecha_Hora"] <= fin_actual)]
    df_anterior = df[(df["Fecha_Hora"] >= inicio_anterior) & (df["Fecha_Hora"] <= fin_anterior)]

    if df_actual.empty:
        return "<p>No hay clases registradas esta semana.</p>"

    resumen_actual = df_actual.groupby(["Nombre_Monitor", "Nombre_Clase"])["Pct_Ocupacion"].mean().round(1)
    resumen_anterior = df_anterior.groupby(["Nombre_Monitor", "Nombre_Clase"])["Pct_Ocupacion"].mean().round(1)

    # Orden: por monitor (alfabético) y, dentro de cada uno, por ocupación
    # de esta semana de mayor a menor.
    indice_ordenado = sorted(
        resumen_actual.index,
        key=lambda clave: (clave[0], -resumen_actual[clave]),
    )

    filas = []
    for monitor, clase in indice_ordenado:
        ocupacion_actual = resumen_actual[(monitor, clase)]
        ocupacion_anterior = resumen_anterior.get((monitor, clase), np.nan)
        if pd.notna(ocupacion_anterior):
            delta = round(ocupacion_actual - ocupacion_anterior, 1)
            flecha = "🔺" if delta >= 0 else "🔻"
            delta_txt = f"{flecha} {delta:+.1f} p.p."
        else:
            delta_txt = "— (sin datos la semana pasada)"
        filas.append((monitor, clase, _fmt_pct(ocupacion_anterior), _fmt_pct(ocupacion_actual), delta_txt))

    encabezado_anterior = f"% Ocup. semana anterior ({inicio_anterior.strftime('%d/%m')}–{fin_anterior.strftime('%d/%m')})"
    encabezado_actual = f"% Ocup. semana actual ({inicio_actual.strftime('%d/%m')}–{fin_actual.strftime('%d/%m')})"

    return _tabla_html(
        ["Entrenador", "Actividad", encabezado_anterior, encabezado_actual, "Variación"],
        filas,
    )


# Frases de apertura, formales pero cercanas, que rotan semana a semana para
# que el email no empiece siempre exactamente igual. La rotación se basa en
# la semana ISO, así que es reproducible (no aleatoria en cada ejecución).
SALUDOS_INICIALES = [
    "Les dejo los resultados de asistencia a las actividades dirigidas de esta semana.",
    "Comparto con vosotros el resumen de asistencia a las clases dirigidas de esta semana.",
    "Aquí tenéis el balance semanal de asistencia a las actividades dirigidas.",
    "Os hago llegar los datos de asistencia a las clases dirigidas de esta semana.",
    "Adjunto el resumen semanal de ocupación y asistencia a las actividades dirigidas.",
    "Como cada semana, os comparto los resultados de asistencia a las clases dirigidas.",
]


def _saludo_inicial(fecha_referencia: pd.Timestamp) -> str:
    semana_iso = int(fecha_referencia.isocalendar()[1])
    return SALUDOS_INICIALES[semana_iso % len(SALUDOS_INICIALES)]


def construir_narrativa(df: pd.DataFrame, fecha_referencia: pd.Timestamp, monitores_activos: list) -> str:
    """
    Párrafo breve, en tono formal pero cercano, con los resultados por
    monitor: semana actual vs. semana pasada, mes en curso vs. mes anterior,
    y comparativa interanual. No menciona el ranking de actividades ni el
    de instructores (esos hablan por sí solos en las imágenes).
    """
    if not monitores_activos:
        return "<p>Esta semana no se registraron clases dirigidas fuera de las actividades excluidas.</p>"

    inicio_actual, fin_actual = _limites_semana(fecha_referencia)
    inicio_anterior, fin_anterior = _limites_semana(fecha_referencia - pd.Timedelta(days=7))

    mes_actual, año_actual = fecha_referencia.month, fecha_referencia.year
    fecha_mes_anterior = fecha_referencia.replace(day=1) - pd.Timedelta(days=1)
    mes_anterior_num, año_mes_anterior = fecha_mes_anterior.month, fecha_mes_anterior.year
    año_anterior = año_actual - 1

    cond_semana_actual = (df["Fecha_Hora"] >= inicio_actual) & (df["Fecha_Hora"] <= fin_actual)
    cond_semana_anterior = (df["Fecha_Hora"] >= inicio_anterior) & (df["Fecha_Hora"] <= fin_anterior)
    cond_mes_actual = (df["Año"] == año_actual) & (df["Mes"] == mes_actual)
    cond_mes_anterior = (df["Año"] == año_mes_anterior) & (df["Mes"] == mes_anterior_num)
    cond_año_anterior = (df["Año"] == año_anterior) & (df["Mes"] == mes_actual)

    valores_semana = {m: _promedio_ocupacion_monitor(df, m, cond_semana_actual) for m in monitores_activos}
    valores_semana_pasada = {m: _promedio_ocupacion_monitor(df, m, cond_semana_anterior) for m in monitores_activos}
    valores_mes = {m: _promedio_ocupacion_monitor(df, m, cond_mes_actual) for m in monitores_activos}
    valores_mes_anterior = {m: _promedio_ocupacion_monitor(df, m, cond_mes_anterior) for m in monitores_activos}
    valores_año_anterior = {m: _promedio_ocupacion_monitor(df, m, cond_año_anterior) for m in monitores_activos}

    media_semana = np.nanmean(list(valores_semana.values()))
    hay_semana_pasada = any(pd.notna(v) for v in valores_semana_pasada.values())
    media_semana_pasada = np.nanmean(list(valores_semana_pasada.values())) if hay_semana_pasada else np.nan
    media_mes = np.nanmean(list(valores_mes.values()))
    hay_mes_anterior = any(pd.notna(v) for v in valores_mes_anterior.values())
    media_mes_anterior = np.nanmean(list(valores_mes_anterior.values())) if hay_mes_anterior else np.nan
    hay_año_anterior = any(pd.notna(v) for v in valores_año_anterior.values())
    media_año_anterior = np.nanmean(list(valores_año_anterior.values())) if hay_año_anterior else np.nan

    monitor_top = max(valores_semana, key=lambda m: valores_semana[m] if pd.notna(valores_semana[m]) else -1)

    deltas_semana = {
        m: valores_semana[m] - valores_semana_pasada[m]
        for m in monitores_activos
        if pd.notna(valores_semana.get(m)) and pd.notna(valores_semana_pasada.get(m))
    }

    frases = [
        f"Esta semana, la ocupación media de los instructores con clases dirigidas fue del "
        f"{media_semana:.1f} %, con {monitor_top} a la cabeza del ranking."
    ]

    if pd.notna(media_semana_pasada):
        delta_semana = media_semana - media_semana_pasada
        direccion = "una subida" if delta_semana >= 0 else "un descenso"
        frase_semana = (
            f"Respecto a la semana anterior, la ocupación media registra {direccion} de "
            f"{abs(delta_semana):.1f} puntos porcentuales."
        )
        if deltas_semana:
            monitor_sube = max(deltas_semana, key=deltas_semana.get)
            monitor_baja = min(deltas_semana, key=deltas_semana.get)
            if deltas_semana[monitor_sube] > 0:
                frase_semana += f" {monitor_sube} es quien más mejora respecto a la semana pasada."
            if deltas_semana[monitor_baja] < 0 and monitor_baja != monitor_sube:
                frase_semana += f" {monitor_baja} es quien más retrocede."
        frases.append(frase_semana)

    if pd.notna(media_mes_anterior):
        delta_mes = media_mes - media_mes_anterior
        direccion_mes = "por encima" if delta_mes >= 0 else "por debajo"
        frases.append(
            f"En lo que va de {MESES_ES[mes_actual]}, la ocupación media se sitúa en el "
            f"{media_mes:.1f} %, {direccion_mes} del {media_mes_anterior:.1f} % del mes anterior."
        )
    else:
        frases.append(
            f"En lo que va de {MESES_ES[mes_actual]}, la ocupación media se sitúa en el {media_mes:.1f} %."
        )

    if pd.notna(media_año_anterior):
        delta_año = media_mes - media_año_anterior
        direccion_año = "por encima" if delta_año >= 0 else "por debajo"
        frases.append(
            f"En la comparativa interanual, el mes en curso queda {direccion_año} del mismo mes de "
            f"{año_anterior}, que registró un {media_año_anterior:.1f} % de ocupación media."
        )

    return "<p>" + " ".join(frases) + "</p>"


def construir_informe_html(df: pd.DataFrame, fecha_referencia: pd.Timestamp):
    """
    Ensambla el email semanal completo. Devuelve (html, imagenes) —
    imagenes es un dict {cid: bytes_png} que hay que incrustar en el email
    junto al HTML.
    """
    imagenes: dict = {}

    inicio_actual, fin_actual = _limites_semana(fecha_referencia)
    df_semana_actual = df[(df["Fecha_Hora"] >= inicio_actual) & (df["Fecha_Hora"] <= fin_actual)]

    # Muy importante: solo se consideran los monitores que aparecen en las
    # actividades no excluidas de la semana en curso (df ya viene sin esas
    # actividades desde normalizar_columnas_csv).
    monitores_activos = sorted(df_semana_actual["Nombre_Monitor"].dropna().unique())

    narrativa = construir_narrativa(df, fecha_referencia, monitores_activos)
    saludo = _saludo_inicial(fecha_referencia)

    html = f"""
    <html>
    <body style="font-family: Arial, sans-serif; color:#1f2937;">
        <h2>🏋️ Informe semanal — semana del {inicio_actual.strftime('%d/%m')} al {fin_actual.strftime('%d/%m/%Y')}</h2>

        <p>Hola Equipo!!!</p>
        <p>{saludo}</p>

        {narrativa}

        <h3>🏆 Ranking de instructores por % de ocupación (esta semana)</h3>
        {bloque_ranking_entrenadores(df_semana_actual, imagenes)}

        <h3>🥇 Ranking de clases por asistencia (esta semana)</h3>
        {bloque_ranking_clases(df_semana_actual, imagenes)}

        <h3>📊 Comparativa por monitor — mes actual, mes anterior y año anterior</h3>
        {bloque_comparativa_monitor_combinada(df, fecha_referencia, monitores_activos, imagenes)}

        <h3>📅 Comparativa semanal por actividad</h3>
        {bloque_comparativa_semanal_actividades(df, fecha_referencia)}

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
        f"🏋️ Informe semanal — "
        f"{inicio_semana.strftime('%d/%m')} al {fin_semana.strftime('%d/%m/%Y')}"
    )

    enviar_email(asunto, html, imagenes)
    print(f"Email enviado correctamente a {os.environ['EMAIL_TO']}")


if __name__ == "__main__":
    main()
