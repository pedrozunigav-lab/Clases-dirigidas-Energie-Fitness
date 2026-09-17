"""
===============================================================================
 INFORME DIARIO POR EMAIL — RENDIMIENTO DE CLASES DEL GIMNASIO
===============================================================================
Script standalone (NO depende de Streamlit) pensado para ejecutarse una vez
al día de forma automática — por ejemplo con GitHub Actions (workflow
incluido en .github/workflows/informe-diario.yml) o con un cron.

Qué hace:
    1. Descarga el CSV más reciente de la carpeta de Google Drive (misma
       cuenta de servicio que usa el dashboard).
    2. Calcula las mismas métricas que el dashboard (% ocupación, tasa de
       cancelación).
    3. Construye un email en HTML con: resumen del día anterior, comparativa
       semana a semana (WoW) y el top 3 de clases y monitores.
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
    python informe_diario.py
===============================================================================
"""

import io
import json
import os
import smtplib
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

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
# 3. CONSTRUCCIÓN DEL INFORME (día anterior + WoW + tops históricos)
# ==============================================================================
def _fmt_pct(x) -> str:
    return f"{x:.1f} %" if pd.notna(x) else "—"


def construir_informe_html(df: pd.DataFrame, fecha_referencia: pd.Timestamp) -> str:
    """
    Construye el HTML del email a partir de tres bloques:
      - Resumen y detalle del día anterior a `fecha_referencia`.
      - Comparativa semana a semana (WoW) de la semana que contiene esa fecha.
      - Top 3 histórico de clases y monitores por % de ocupación.
    """
    ayer = (fecha_referencia - timedelta(days=1)).date()
    df_ayer = df[df["Fecha_Hora"].dt.date == ayer]

    # --- Bloque: resumen de ayer ----------------------------------------------
    if df_ayer.empty:
        bloque_ayer = "<p>No hubo clases registradas ayer (o los datos aún no se han actualizado).</p>"
    else:
        asistencia_total = int(df_ayer["Asistentes_Reales"].sum())
        ocupacion_media = df_ayer["Pct_Ocupacion"].mean()
        cancelacion_media = df_ayer["Tasa_Cancelacion"].mean()
        n_clases = len(df_ayer)

        filas_clase = "".join(
            f"<tr><td>{fila.Fecha_Hora.strftime('%H:%M')}</td><td>{fila.Nombre_Clase}</td>"
            f"<td>{fila.Nombre_Monitor}</td>"
            f"<td>{fila.Asistentes_Reales}/{fila.Capacidad_Máxima_Clase}</td>"
            f"<td>{_fmt_pct(fila.Pct_Ocupacion)}</td></tr>"
            for fila in df_ayer.sort_values("Fecha_Hora").itertuples()
        )

        bloque_ayer = f"""
        <p><b>Clases impartidas:</b> {n_clases} &nbsp;|&nbsp;
           <b>Asistencia total:</b> {asistencia_total} &nbsp;|&nbsp;
           <b>Ocupación media:</b> {_fmt_pct(ocupacion_media)} &nbsp;|&nbsp;
           <b>Cancelación media:</b> {_fmt_pct(cancelacion_media)}</p>
        <table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse; width:100%;">
            <tr style="background:#1f2937;color:#ffffff;">
                <th>Hora</th><th>Clase</th><th>Monitor</th><th>Asistentes</th><th>% Ocupación</th>
            </tr>
            {filas_clase}
        </table>
        """

    # --- Bloque: comparativa WoW ----------------------------------------------
    inicio_semana_actual = fecha_referencia - pd.Timedelta(days=fecha_referencia.weekday())
    fin_semana_actual = inicio_semana_actual + pd.Timedelta(days=6, hours=23, minutes=59)
    inicio_semana_anterior = inicio_semana_actual - pd.Timedelta(days=7)
    fin_semana_anterior = inicio_semana_actual - pd.Timedelta(seconds=1)

    df_actual = df[(df["Fecha_Hora"] >= inicio_semana_actual) & (df["Fecha_Hora"] <= fin_semana_actual)]
    df_anterior = df[(df["Fecha_Hora"] >= inicio_semana_anterior) & (df["Fecha_Hora"] <= fin_semana_anterior)]

    asistencia_actual = int(df_actual["Asistentes_Reales"].sum())
    asistencia_anterior = int(df_anterior["Asistentes_Reales"].sum())
    ocupacion_actual = df_actual["Pct_Ocupacion"].mean() if len(df_actual) else np.nan
    ocupacion_anterior = df_anterior["Pct_Ocupacion"].mean() if len(df_anterior) else np.nan

    delta_asistencia = asistencia_actual - asistencia_anterior
    flecha = "🔺" if delta_asistencia >= 0 else "🔻"

    bloque_wow = f"""
    <p><b>Asistencia semana en curso:</b> {asistencia_actual}
       ({flecha} {delta_asistencia:+d} vs. semana anterior: {asistencia_anterior})<br>
       <b>Ocupación media semana en curso:</b> {_fmt_pct(ocupacion_actual)}
       (semana anterior: {_fmt_pct(ocupacion_anterior)})</p>
    """

    # --- Bloque: top histórico de clases y monitores ---------------------------
    top_clases = (
        df.groupby("Nombre_Clase")["Pct_Ocupacion"].mean().round(1)
        .sort_values(ascending=False).head(3)
    )
    top_monitores = (
        df.groupby("Nombre_Monitor")["Pct_Ocupacion"].mean().round(1)
        .sort_values(ascending=False).head(3)
    )
    filas_top_clases = "".join(f"<li>{c}: {_fmt_pct(v)}</li>" for c, v in top_clases.items())
    filas_top_monitores = "".join(f"<li>{m}: {_fmt_pct(v)}</li>" for m, v in top_monitores.items())

    return f"""
    <html>
    <body style="font-family: Arial, sans-serif; color:#1f2937;">
        <h2>🏋️ Informe diario del gimnasio — {ayer.strftime('%d/%m/%Y')}</h2>

        <h3>📅 Resumen de ayer</h3>
        {bloque_ayer}

        <h3>📊 Comparativa semana a semana (WoW)</h3>
        {bloque_wow}

        <h3>🏆 Top 3 clases por ocupación (histórico)</h3>
        <ul>{filas_top_clases}</ul>

        <h3>🧑‍🏫 Top 3 monitores por ocupación (histórico)</h3>
        <ul>{filas_top_monitores}</ul>

        <p style="color:#6b7280; font-size:12px;">Informe generado automáticamente.</p>
    </body>
    </html>
    """


# ==============================================================================
# 4. ENVÍO DEL EMAIL (Gmail SMTP con contraseña de aplicación)
# ==============================================================================
def enviar_email(asunto: str, html: str) -> None:
    gmail_user = os.environ["GMAIL_USER"]
    gmail_password = os.environ["GMAIL_APP_PASSWORD"]
    destinatarios = [d.strip() for d in os.environ["EMAIL_TO"].split(",") if d.strip()]

    mensaje = MIMEMultipart("alternative")
    mensaje["Subject"] = asunto
    mensaje["From"] = gmail_user
    mensaje["To"] = ", ".join(destinatarios)
    mensaje.attach(MIMEText(html, "html", "utf-8"))

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
    ayer = (fecha_referencia - timedelta(days=1)).date()

    html = construir_informe_html(df, fecha_referencia)
    asunto = f"🏋️ Informe diario del gimnasio — {ayer.strftime('%d/%m/%Y')}"

    enviar_email(asunto, html)
    print(f"Email enviado correctamente a {os.environ['EMAIL_TO']}")


if __name__ == "__main__":
    main()
