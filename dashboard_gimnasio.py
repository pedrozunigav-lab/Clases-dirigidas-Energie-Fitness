"""
===============================================================================
 DASHBOARD ANALÍTICO — RENDIMIENTO DE CLASES DE GIMNASIO
===============================================================================
Aplicación Streamlit + Plotly + Pandas para analizar la ocupación de clases,
el rendimiento de los monitores y la retención de clientes, con el objetivo
de apoyar decisiones sobre optimización de horarios.

Cómo ejecutarlo:
    pip install streamlit plotly pandas numpy google-api-python-client google-auth
    streamlit run dashboard_gimnasio.py

El dataset es 100% sintético (se genera al vuelo) para que el script sea
ejecutable "tal cual", sin necesidad de ficheros externos. Basta con sustituir
la función `generar_datos_sinteticos()` por una carga real (CSV, base de
datos, API, etc.) para usarlo en producción — el resto del pipeline no
necesita cambios porque trabaja siempre sobre el mismo esquema de columnas.
===============================================================================

Fuente de datos:
    El dashboard puede alimentarse de dos formas (elegibles desde la barra
    lateral):
      1. Un CSV alojado en una carpeta PRIVADA de Google Drive, leído a
         través de la Google Drive API con una cuenta de servicio (no hace
         falta compartir la carpeta públicamente).
      2. El dataset sintético de demostración (útil para probar el dashboard
         sin credenciales, o como fallback si falla la conexión a Drive).

    Para usar la opción de Drive necesitas:
      a) Crear un proyecto en Google Cloud y habilitar la "Google Drive API".
      b) Crear una cuenta de servicio y descargar su clave en JSON.
      c) Compartir la carpeta de Drive que contiene el CSV con el email de
         esa cuenta de servicio (algo como
         nombre-cuenta@proyecto.iam.gserviceaccount.com), como Lector.
      d) Pegar el contenido del JSON en `.streamlit/secrets.toml` bajo la
         clave [gcp_service_account] (ver plantilla al final de este fichero),
         o bien dejar el JSON como fichero local y ajustar SERVICE_ACCOUNT_FILE.
"""

import io
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ==============================================================================
# CONFIGURACIÓN GENERAL DE LA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Dashboard Gimnasio · Rendimiento de Clases",
    page_icon="🏋️",
    layout="wide",
)


# ==============================================================================
# 1. CONEXIÓN A GOOGLE DRIVE (CUENTA DE SERVICIO)
# ==============================================================================
SCOPES_DRIVE = ["https://www.googleapis.com/auth/drive.readonly"]

# Ruta al JSON de la cuenta de servicio SOLO para desarrollo local, si no usas
# st.secrets. En producción (Streamlit Cloud, por ejemplo) usa siempre
# st.secrets["gcp_service_account"] — nunca subas el JSON a un repo público.
SERVICE_ACCOUNT_FILE = "credenciales_drive.json"

# Columnas mínimas que el CSV de Drive debe traer para que el resto del
# pipeline funcione sin cambios.
COLUMNAS_ESPERADAS = {
    "Fecha_Hora",
    "Nombre_Clase",
    "Nombre_Monitor",
    "Capacidad_Máxima_Clase",
    "Asistentes_Reales",
    "Cancelaciones_Última_Hora",
}


@st.cache_resource(show_spinner=False)
def obtener_servicio_drive():
    """
    Crea y devuelve el cliente autenticado de la Google Drive API v3 usando
    una cuenta de servicio.

    Prioriza las credenciales guardadas en `st.secrets["gcp_service_account"]`
    (forma recomendada al desplegar la app); si no existen, cae a un fichero
    JSON local (`SERVICE_ACCOUNT_FILE`) para desarrollo en local.
    """
    if "gcp_service_account" in st.secrets:
        info = dict(st.secrets["gcp_service_account"])
        credenciales = service_account.Credentials.from_service_account_info(
            info, scopes=SCOPES_DRIVE
        )
    else:
        credenciales = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES_DRIVE
        )

    return build("drive", "v3", credentials=credenciales, cache_discovery=False)


def buscar_csv_en_carpeta(servicio, carpeta_id: str, nombre_archivo: str = ""):
    """
    Busca un CSV dentro de una carpeta de Drive (por ID de carpeta).

    Si se indica `nombre_archivo`, filtra por ese nombre exacto; si no, coge
    el CSV modificado más recientemente dentro de la carpeta (útil cuando el
    fichero se sobrescribe/actualiza periódicamente con el mismo nombre).
    """
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
        raise FileNotFoundError(
            f"No se encontró ningún CSV{detalle} en la carpeta de Drive '{carpeta_id}'. "
            "Comprueba el ID de carpeta y que la has compartido con el email "
            "de la cuenta de servicio."
        )
    return archivos[0]  # el más reciente


def descargar_csv_drive(servicio, file_id: str) -> io.BytesIO:
    """Descarga el contenido binario de un fichero de Drive a un buffer en memoria."""
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
    Valida que el CSV traiga las columnas mínimas esperadas y convierte
    Fecha_Hora a datetime. Si faltan columnas o IDs, los reconstruye o lanza
    un error explicativo para que el usuario corrija el CSV de origen.
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    faltantes = COLUMNAS_ESPERADAS - set(df.columns)
    if faltantes:
        raise ValueError(
            "Al CSV le faltan columnas obligatorias: "
            f"{sorted(faltantes)}. Columnas encontradas: {sorted(df.columns)}"
        )

    df["Fecha_Hora"] = pd.to_datetime(df["Fecha_Hora"], errors="raise")

    # ID_Clase / ID_Monitor son opcionales: si no vienen en el CSV, se generan
    # a partir de los nombres para mantener el mismo esquema que el resto del
    # pipeline espera.
    if "ID_Clase" not in df.columns:
        df["ID_Clase"] = df["Nombre_Clase"].astype("category").cat.codes + 1
    if "ID_Monitor" not in df.columns:
        df["ID_Monitor"] = df["Nombre_Monitor"].astype("category").cat.codes + 1

    for col in ["Capacidad_Máxima_Clase", "Asistentes_Reales", "Cancelaciones_Última_Hora"]:
        df[col] = pd.to_numeric(df[col], errors="raise")

    return df.sort_values("Fecha_Hora").reset_index(drop=True)


def leer_csv_desde_buffer(buffer: io.BytesIO) -> pd.DataFrame:
    """
    Lee un CSV probando varias codificaciones habituales, en orden: UTF-8
    (con y sin BOM), Windows-1252 y Latin-1. Evita que falle cuando el CSV
    viene exportado desde Excel en español (tildes y "ñ" en Windows-1252/
    Latin-1 en vez de UTF-8).
    """
    codificaciones = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]
    ultimo_error = None
    for codificacion in codificaciones:
        try:
            buffer.seek(0)
            return pd.read_csv(buffer, encoding=codificacion)
        except UnicodeDecodeError as error:
            ultimo_error = error
            continue
    raise ultimo_error


@st.cache_data(ttl=600, show_spinner="Descargando CSV desde Google Drive…")
def cargar_datos_desde_drive(carpeta_id: str, nombre_archivo: str = "") -> pd.DataFrame:
    """
    Pipeline completo: autentica con la cuenta de servicio, localiza el CSV
    en la carpeta de Drive indicada, lo descarga y lo normaliza al esquema
    que usa el resto del dashboard.

    Cacheada durante 10 minutos (`ttl=600`) para no golpear la API de Drive
    en cada interacción del usuario con los filtros.
    """
    servicio = obtener_servicio_drive()
    archivo = buscar_csv_en_carpeta(servicio, carpeta_id, nombre_archivo)
    buffer = descargar_csv_drive(servicio, archivo["id"])
    df = leer_csv_desde_buffer(buffer)
    return normalizar_columnas_csv(df)


# ==============================================================================
# 2. GENERACIÓN DE DATOS SINTÉTICOS (DEMO / FALLBACK)
# ==============================================================================
@st.cache_data(show_spinner="Generando dataset sintético…")
def generar_datos_sinteticos(
    fecha_inicio: str = "2024-09-01",
    fecha_fin: str = "2026-09-17",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Genera un dataset sintético y realista de clases de gimnasio.

    Simula un horario semanal recurrente (varias clases por día, de distintos
    tipos y monitores), y para cada sesión real en el rango de fechas calcula
    una asistencia coherente con:
      - la popularidad propia de cada tipo de clase,
      - la "habilidad de retención" propia de cada monitor,
      - un factor estacional (bajón en verano, repunte en enero/septiembre),
      - una tendencia de crecimiento del gimnasio a lo largo del tiempo.

    Devuelve un DataFrame con las columnas pedidas:
        Fecha_Hora, ID_Clase, Nombre_Clase, ID_Monitor, Nombre_Monitor,
        Capacidad_Máxima_Clase, Asistentes_Reales, Cancelaciones_Última_Hora
    """
    rng = np.random.default_rng(seed)

    # --- Catálogo de clases: capacidad típica y popularidad base (0-1) -------
    clases_info = {
        "Yoga":     {"capacidad": 20, "popularidad": 0.75},
        "CrossFit": {"capacidad": 15, "popularidad": 0.85},
        "Spinning": {"capacidad": 25, "popularidad": 0.70},
        "Zumba":    {"capacidad": 30, "popularidad": 0.80},
        "Pilates":  {"capacidad": 18, "popularidad": 0.65},
        "Boxeo":    {"capacidad": 16, "popularidad": 0.60},
    }

    # --- Catálogo de monitores: "rating" que suma/resta a la popularidad -----
    monitores_info = {
        "Laura Gómez": {"rating": 0.10},
        "Marc Ferrer":  {"rating": 0.05},
        "Anna Puig":    {"rating": 0.15},
        "David Soler":  {"rating": -0.05},
        "Núria Vidal":  {"rating": 0.00},
        "Jordi Martí":  {"rating": -0.10},
    }

    nombres_clases = list(clases_info.keys())
    nombres_monitores = list(monitores_info.keys())

    # --- Plantilla semanal (el gimnasio cierra los domingos) -----------------
    dias_semana = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado"]
    dia_semana_a_weekday = {
        "Lunes": 0, "Martes": 1, "Miércoles": 2, "Jueves": 3,
        "Viernes": 4, "Sábado": 5, "Domingo": 6,
    }
    horas_pico = [9, 10, 18, 19, 20]
    horas_valle = [7, 8, 12, 13, 16, 17]
    horas_posibles = horas_pico + horas_valle

    plantilla = []
    for dia in dias_semana:
        n_clases_dia = rng.integers(4, 7)  # entre 4 y 6 sesiones ese día
        for _ in range(n_clases_dia):
            plantilla.append({
                "dia_semana": dia,
                "clase": rng.choice(nombres_clases),
                "monitor": rng.choice(nombres_monitores),
                "hora": int(rng.choice(horas_posibles)),
            })

    fechas = pd.date_range(start=fecha_inicio, end=fecha_fin, freq="D")
    total_dias = max(len(fechas), 1)
    primer_dia = fechas[0]

    registros = []
    for fecha in fechas:
        weekday_actual = fecha.weekday()
        clases_del_dia = [
            p for p in plantilla
            if dia_semana_a_weekday[p["dia_semana"]] == weekday_actual
        ]
        if not clases_del_dia:
            continue  # domingo: gimnasio cerrado

        for sesion in clases_del_dia:
            info_clase = clases_info[sesion["clase"]]
            info_monitor = monitores_info[sesion["monitor"]]
            capacidad = info_clase["capacidad"]

            # Factor estacional: caída en verano, repunte en enero/septiembre
            mes = fecha.month
            if mes in (7, 8):
                factor_estacional = 0.65
            elif mes in (1, 9):
                factor_estacional = 1.15
            else:
                factor_estacional = 1.0

            # Tendencia de crecimiento suave a lo largo de todo el periodo
            dias_desde_inicio = (fecha - primer_dia).days
            factor_crecimiento = 1 + (dias_desde_inicio / total_dias) * 0.15

            popularidad_efectiva = float(np.clip(
                info_clase["popularidad"] + info_monitor["rating"], 0.05, 0.98
            ))
            ocupacion_esperada = float(np.clip(
                popularidad_efectiva * factor_estacional * factor_crecimiento,
                0.05, 1.0,
            ))

            asistentes = int(np.clip(
                rng.binomial(capacidad, ocupacion_esperada), 0, capacidad
            ))

            # Reservas que se cancelan en el último momento (ruido acotado)
            cancelaciones = int(rng.integers(0, 4))
            # No tiene sentido cancelar más reservas de las que "cabrían"
            cancelaciones = min(cancelaciones, capacidad - asistentes + cancelaciones)

            registros.append({
                "Fecha_Hora": fecha.replace(hour=sesion["hora"], minute=0, second=0),
                "ID_Clase": nombres_clases.index(sesion["clase"]) + 1,
                "Nombre_Clase": sesion["clase"],
                "ID_Monitor": nombres_monitores.index(sesion["monitor"]) + 1,
                "Nombre_Monitor": sesion["monitor"],
                "Capacidad_Máxima_Clase": capacidad,
                "Asistentes_Reales": asistentes,
                "Cancelaciones_Última_Hora": cancelaciones,
            })

    df = pd.DataFrame(registros)
    df = df.sort_values("Fecha_Hora").reset_index(drop=True)
    return df


# ==============================================================================
# 3. CÁLCULO DE MÉTRICAS DERIVADAS
# ==============================================================================
def calcular_metricas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade al DataFrame original todas las columnas derivadas necesarias para
    los análisis: % de ocupación, tasa de cancelación y columnas temporales
    auxiliares (año, mes, semana ISO, día de la semana, hora).
    """
    df = df.copy()

    # Reservas totales = quien acabó asistiendo + quien canceló a última hora
    df["Reservas_Totales"] = df["Asistentes_Reales"] + df["Cancelaciones_Última_Hora"]

    # % de Ocupación = Asistentes_Reales / Capacidad_Máxima_Clase
    df["Pct_Ocupacion"] = (
        df["Asistentes_Reales"] / df["Capacidad_Máxima_Clase"] * 100
    ).round(2)

    # Tasa de cancelación = cancelaciones / reservas totales
    df["Tasa_Cancelacion"] = np.where(
        df["Reservas_Totales"] > 0,
        (df["Cancelaciones_Última_Hora"] / df["Reservas_Totales"] * 100).round(2),
        0.0,
    )

    # Columnas temporales auxiliares (fundamentales para WoW / YoY)
    df["Año"] = df["Fecha_Hora"].dt.year
    df["Mes"] = df["Fecha_Hora"].dt.month
    df["Nombre_Mes"] = df["Fecha_Hora"].dt.strftime("%B")
    df["Dia_Semana"] = df["Fecha_Hora"].dt.day_name()
    df["Hora"] = df["Fecha_Hora"].dt.hour

    iso = df["Fecha_Hora"].dt.isocalendar()  # año/semana ISO -> comparaciones WoW robustas
    df["Año_ISO"] = iso["year"]
    df["Semana_ISO"] = iso["week"]

    return df


# ==============================================================================
# 4. FILTROS INTERACTIVOS (BARRA LATERAL)
# ==============================================================================
def aplicar_filtros_sidebar(df: pd.DataFrame):
    """
    Dibuja los filtros en la barra lateral (rango de fechas, tipo de clase,
    monitor) y devuelve el DataFrame filtrado junto con las fechas elegidas.
    """
    st.sidebar.header("🔍 Filtros")

    fecha_min = df["Fecha_Hora"].min().date()
    fecha_max = df["Fecha_Hora"].max().date()

    rango_fechas = st.sidebar.date_input(
        "Rango de fechas",
        value=(fecha_max - timedelta(days=90), fecha_max),
        min_value=fecha_min,
        max_value=fecha_max,
    )
    # date_input devuelve una tupla solo cuando el usuario ha elegido las dos
    # fechas; mientras tanto puede devolver un único valor.
    if isinstance(rango_fechas, tuple) and len(rango_fechas) == 2:
        f_inicio, f_fin = rango_fechas
    else:
        f_inicio, f_fin = fecha_min, fecha_max

    clases_disponibles = sorted(df["Nombre_Clase"].unique())
    clases_sel = st.sidebar.multiselect(
        "Tipo de clase", clases_disponibles, default=clases_disponibles
    )

    monitores_disponibles = sorted(df["Nombre_Monitor"].unique())
    monitores_sel = st.sidebar.multiselect(
        "Monitor", monitores_disponibles, default=monitores_disponibles
    )

    df_filtrado = df[
        (df["Fecha_Hora"].dt.date >= f_inicio)
        & (df["Fecha_Hora"].dt.date <= f_fin)
        & (df["Nombre_Clase"].isin(clases_sel))
        & (df["Nombre_Monitor"].isin(monitores_sel))
    ]

    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"📌 {len(df_filtrado):,} clases seleccionadas".replace(",", ".")
    )

    return df_filtrado, f_inicio, f_fin


# ==============================================================================
# 5. RENDERIZADO — KPIs GENERALES
# ==============================================================================
def render_kpis(df: pd.DataFrame) -> None:
    st.subheader("📊 Resumen general")

    col1, col2, col3, col4 = st.columns(4)

    total_asistentes = int(df["Asistentes_Reales"].sum())
    ocupacion_media = df["Pct_Ocupacion"].mean() if len(df) else 0.0
    cancelacion_media = df["Tasa_Cancelacion"].mean() if len(df) else 0.0
    total_clases = len(df)

    col1.metric("👥 Asistencia total", f"{total_asistentes:,}".replace(",", "."))
    col2.metric("📈 Ocupación media", f"{ocupacion_media:.1f} %")
    col3.metric("❌ Cancelación media", f"{cancelacion_media:.1f} %")
    col4.metric("🗓️ Clases impartidas", f"{total_clases:,}".replace(",", "."))


# ==============================================================================
# 6. RENDERIZADO — ANÁLISIS CLASE POR CLASE
# ==============================================================================
def render_analisis_clases(df: pd.DataFrame) -> None:
    st.subheader("🧘 Análisis clase por clase")

    resumen_clase = (
        df.groupby("Nombre_Clase")
        .agg(
            Asistencia_Total=("Asistentes_Reales", "sum"),
            Asistencia_Media=("Asistentes_Reales", "mean"),
            Ocupacion_Media=("Pct_Ocupacion", "mean"),
            Cancelacion_Media=("Tasa_Cancelacion", "mean"),
            Num_Clases=("Nombre_Clase", "count"),
        )
        .round(1)
        .sort_values("Ocupacion_Media", ascending=False)
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            resumen_clase.reset_index(),
            x="Nombre_Clase",
            y="Ocupacion_Media",
            color="Ocupacion_Media",
            color_continuous_scale="Blues",
            title="Ocupación media por tipo de clase (%)",
            labels={"Nombre_Clase": "Clase", "Ocupacion_Media": "% Ocupación"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            resumen_clase.reset_index(),
            x="Nombre_Clase",
            y="Asistencia_Total",
            color="Asistencia_Total",
            color_continuous_scale="Greens",
            title="Asistencia total por tipo de clase",
            labels={"Nombre_Clase": "Clase", "Asistencia_Total": "Asistentes totales"},
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Mapa de calor: horarios pico (día de la semana x hora) --------------
    orden_dias_en = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    dias_en_a_es = {
        "Monday": "Lunes", "Tuesday": "Martes", "Wednesday": "Miércoles",
        "Thursday": "Jueves", "Friday": "Viernes", "Saturday": "Sábado",
    }
    heat = df.groupby(["Dia_Semana", "Hora"])["Pct_Ocupacion"].mean().reset_index()
    heat["Dia_ES"] = heat["Dia_Semana"].map(dias_en_a_es)
    heat_pivot = heat.pivot(index="Dia_ES", columns="Hora", values="Pct_Ocupacion")
    orden_es = [dias_en_a_es[d] for d in orden_dias_en if dias_en_a_es[d] in heat_pivot.index]
    heat_pivot = heat_pivot.reindex(orden_es)

    fig_heat = px.imshow(
        heat_pivot,
        color_continuous_scale="YlOrRd",
        aspect="auto",
        title="Mapa de calor — ocupación media por día y hora (identifica horarios pico)",
        labels={"x": "Hora del día", "y": "Día de la semana", "color": "% Ocupación"},
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # --- Clases más y menos populares ----------------------------------------
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**🏆 Clases más populares** (por % ocupación)")
        st.dataframe(resumen_clase.head(3), use_container_width=True)
    with col_b:
        st.markdown("**📉 Clases menos concurridas** (por % ocupación)")
        st.dataframe(
            resumen_clase.sort_values("Ocupacion_Media").head(3),
            use_container_width=True,
        )


# ==============================================================================
# 7. RENDERIZADO — ANÁLISIS MONITOR POR MONITOR
# ==============================================================================
def render_analisis_monitores(df: pd.DataFrame) -> None:
    st.subheader("🧑‍🏫 Análisis monitor por monitor")

    resumen_monitor = (
        df.groupby("Nombre_Monitor")
        .agg(
            Asistencia_Media=("Asistentes_Reales", "mean"),
            Ocupacion_Media=("Pct_Ocupacion", "mean"),
            Cancelacion_Media=("Tasa_Cancelacion", "mean"),
            Num_Clases=("Nombre_Monitor", "count"),
        )
        .round(1)
        .sort_values("Ocupacion_Media", ascending=False)
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            resumen_monitor.reset_index(),
            x="Nombre_Monitor",
            y="Ocupacion_Media",
            color="Cancelacion_Media",
            color_continuous_scale="RdYlGn_r",
            title="Ocupación media por monitor (color = tasa de cancelación)",
            labels={"Nombre_Monitor": "Monitor", "Ocupacion_Media": "% Ocupación media"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # La "retención" se aproxima aquí como el complemento de la cancelación:
        # a menor tasa de cancelación de sus clases, mejor retiene el monitor
        # a sus alumnos reservados.
        fig2 = px.scatter(
            resumen_monitor.reset_index(),
            x="Asistencia_Media",
            y="Cancelacion_Media",
            size="Num_Clases",
            color="Nombre_Monitor",
            text="Nombre_Monitor",
            title="Retención (cancelación) vs. asistencia media por monitor",
            labels={
                "Asistencia_Media": "Asistencia media por clase",
                "Cancelacion_Media": "Tasa de cancelación (%)",
            },
        )
        fig2.update_traces(textposition="top center")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**📋 Tabla comparativa de monitores**")
    st.dataframe(resumen_monitor, use_container_width=True)


# ==============================================================================
# 8. RENDERIZADO — COMPARATIVA SEMANA A SEMANA (WoW)
# ==============================================================================
def render_comparativa_wow(df_completo: pd.DataFrame, f_fin) -> None:
    """
    Compara la semana ISO que contiene `f_fin` con la semana ISO inmediatamente
    anterior. Se usa siempre el dataset COMPLETO (no el filtrado por fechas),
    porque la comparativa necesita ver ambas semanas aunque el usuario haya
    acotado el rango de fechas en la barra lateral.
    """
    st.subheader("📅 Comparativa semana a semana (WoW)")

    fecha_ref = pd.Timestamp(f_fin)

    inicio_semana_actual = fecha_ref - pd.Timedelta(days=fecha_ref.weekday())
    fin_semana_actual = inicio_semana_actual + pd.Timedelta(days=6)
    inicio_semana_anterior = inicio_semana_actual - pd.Timedelta(days=7)
    fin_semana_anterior = inicio_semana_actual - pd.Timedelta(days=1)

    df_actual = df_completo[
        (df_completo["Fecha_Hora"] >= inicio_semana_actual)
        & (df_completo["Fecha_Hora"] <= fin_semana_actual + pd.Timedelta(hours=23, minutes=59))
    ]
    df_anterior = df_completo[
        (df_completo["Fecha_Hora"] >= inicio_semana_anterior)
        & (df_completo["Fecha_Hora"] <= fin_semana_anterior + pd.Timedelta(hours=23, minutes=59))
    ]

    if df_actual.empty and df_anterior.empty:
        st.info("No hay datos suficientes para comparar estas dos semanas.")
        return

    def _resumen(d: pd.DataFrame) -> dict:
        return {
            "Asistencia total": int(d["Asistentes_Reales"].sum()),
            "Ocupación media (%)": round(d["Pct_Ocupacion"].mean(), 1) if len(d) else 0.0,
            "Cancelación media (%)": round(d["Tasa_Cancelacion"].mean(), 1) if len(d) else 0.0,
        }

    actual = _resumen(df_actual)
    anterior = _resumen(df_anterior)

    cols = st.columns(3)
    for col, etiqueta in zip(cols, actual.keys()):
        valor_actual = actual[etiqueta]
        valor_anterior = anterior[etiqueta]
        delta = round(valor_actual - valor_anterior, 1)
        col.metric(etiqueta, valor_actual, delta=delta)

    st.caption(
        f"Semana actual: {inicio_semana_actual.date()} → {fin_semana_actual.date()}  "
        f"·  Semana anterior: {inicio_semana_anterior.date()} → {fin_semana_anterior.date()}"
    )

    comp_df = pd.DataFrame({
        "Periodo": ["Semana anterior", "Semana actual"],
        "Asistencia total": [anterior["Asistencia total"], actual["Asistencia total"]],
        "Ocupación media (%)": [anterior["Ocupación media (%)"], actual["Ocupación media (%)"]],
    })

    fig = go.Figure()
    fig.add_bar(name="Asistencia total", x=comp_df["Periodo"], y=comp_df["Asistencia total"])
    fig.update_layout(title="Asistencia total: semana actual vs. semana anterior")
    st.plotly_chart(fig, use_container_width=True)


# ==============================================================================
# 9. RENDERIZADO — COMPARATIVA INTERANUAL (YoY)
# ==============================================================================
def render_comparativa_yoy(df_completo: pd.DataFrame, f_fin) -> None:
    """
    Compara el mes del calendario al que pertenece `f_fin` con el mismo mes
    del año anterior (p. ej. septiembre 2026 vs. septiembre 2025). Igual que
    en el WoW, se usa el dataset completo para poder ver ambos periodos.
    """
    st.subheader("📆 Comparativa interanual (YoY)")

    fecha_ref = pd.Timestamp(f_fin)
    mes_actual = fecha_ref.month
    año_actual = fecha_ref.year
    año_anterior = año_actual - 1

    df_mes_actual = df_completo[
        (df_completo["Año"] == año_actual) & (df_completo["Mes"] == mes_actual)
    ]
    df_mes_anterior = df_completo[
        (df_completo["Año"] == año_anterior) & (df_completo["Mes"] == mes_actual)
    ]

    if df_mes_anterior.empty:
        st.info("No hay datos del año anterior para este mes todavía.")
        return

    def _resumen(d: pd.DataFrame) -> dict:
        return {
            "Asistencia total": int(d["Asistentes_Reales"].sum()),
            "Ocupación media (%)": round(d["Pct_Ocupacion"].mean(), 1) if len(d) else 0.0,
            "Cancelación media (%)": round(d["Tasa_Cancelacion"].mean(), 1) if len(d) else 0.0,
        }

    actual = _resumen(df_mes_actual)
    anterior = _resumen(df_mes_anterior)
    nombre_mes = fecha_ref.strftime("%B").capitalize()

    cols = st.columns(3)
    for col, etiqueta in zip(cols, actual.keys()):
        valor_actual = actual[etiqueta]
        valor_anterior = anterior[etiqueta]
        delta_pct = (
            (valor_actual - valor_anterior) / valor_anterior * 100
            if valor_anterior else 0.0
        )
        col.metric(etiqueta, valor_actual, delta=f"{delta_pct:.1f} %")

    st.caption(f"{nombre_mes} {año_actual}  vs.  {nombre_mes} {año_anterior}")

    comp_df = pd.DataFrame({
        "Periodo": [f"{nombre_mes} {año_anterior}", f"{nombre_mes} {año_actual}"],
        "Asistencia total": [anterior["Asistencia total"], actual["Asistencia total"]],
    })
    fig = px.bar(
        comp_df,
        x="Periodo",
        y="Asistencia total",
        color="Periodo",
        title=f"Asistencia total — {nombre_mes}: {año_actual} vs. {año_anterior}",
    )
    st.plotly_chart(fig, use_container_width=True)


# ==============================================================================
# 10. FUNCIÓN PRINCIPAL
# ==============================================================================
def seleccionar_fuente_datos() -> pd.DataFrame:
    """
    Dibuja en la barra lateral el selector de fuente de datos y devuelve el
    DataFrame crudo (sin métricas calculadas todavía) según la elección:
      - CSV en Google Drive (cuenta de servicio), o
      - Dataset sintético de demostración.

    Si falla la conexión a Drive (credenciales, carpeta o CSV mal
    configurados), se muestra el error y se cae automáticamente al dataset
    sintético para que el dashboard nunca se quede en blanco.
    """
    st.sidebar.header("🗂️ Fuente de datos")
    fuente = st.sidebar.radio(
        "Origen de los datos",
        ["CSV en Google Drive", "Dataset sintético (demo)"],
        index=0,
    )

    if fuente == "Dataset sintético (demo)":
        return generar_datos_sinteticos()

    carpeta_id = st.sidebar.text_input(
        "ID de la carpeta de Drive",
        help="Es el fragmento de la URL de la carpeta: "
             "https://drive.google.com/drive/folders/《ESTE ID》",
    )
    nombre_archivo = st.sidebar.text_input(
        "Nombre exacto del CSV (opcional)",
        help="Déjalo vacío para usar automáticamente el CSV modificado "
             "más recientemente dentro de la carpeta.",
    )

    if not carpeta_id:
        st.info("👈 Introduce el ID de la carpeta de Drive en la barra lateral para cargar los datos reales.")
        st.stop()

    try:
        return cargar_datos_desde_drive(carpeta_id, nombre_archivo)
    except Exception as error:
        st.sidebar.error(f"No se pudo cargar el CSV de Drive: {error}")
        st.warning(
            "⚠️ No se pudo conectar con Google Drive — mostrando el dataset "
            "sintético de demostración mientras tanto."
        )
        return generar_datos_sinteticos()


def main() -> None:
    st.title("🏋️ Dashboard de Rendimiento de Clases del Gimnasio")
    st.markdown(
        "Análisis interactivo de **ocupación**, **cancelaciones** y "
        "**rendimiento de instructores** para optimizar horarios y mejorar "
        "la retención de clientes."
    )

    # --- Pipeline de datos: elegir fuente -> cargar -> calcular métricas ------
    df_raw = seleccionar_fuente_datos()
    df = calcular_metricas(df_raw)

    # --- Filtros interactivos --------------------------------------------------
    df_filtrado, f_inicio, f_fin = aplicar_filtros_sidebar(df)

    if df_filtrado.empty:
        st.warning("No hay datos para los filtros seleccionados. Ajusta el rango o las selecciones.")
        return

    # --- Secciones del dashboard ------------------------------------------------
    render_kpis(df_filtrado)
    st.divider()

    render_analisis_clases(df_filtrado)
    st.divider()

    render_analisis_monitores(df_filtrado)
    st.divider()

    # WoW y YoY se calculan sobre el dataset COMPLETO (no el filtrado por
    # fechas) para poder comparar siempre dos periodos completos, tomando
    # como referencia la fecha final seleccionada en el filtro.
    render_comparativa_wow(df, f_fin)
    st.divider()

    render_comparativa_yoy(df, f_fin)
    st.divider()

    with st.expander("📄 Ver datos crudos del periodo filtrado"):
        st.dataframe(df_filtrado, use_container_width=True)
        st.download_button(
            "⬇️ Descargar CSV",
            data=df_filtrado.to_csv(index=False).encode("utf-8"),
            file_name="clases_gimnasio_filtrado.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()


# ==============================================================================
# PLANTILLA: .streamlit/secrets.toml
# ==============================================================================
# Copia el contenido del JSON de tu cuenta de servicio dentro de este bloque
# (respetando las comillas triples para private_key, que tiene saltos de
# línea). Streamlit carga automáticamente este fichero como `st.secrets`.
#
# [gcp_service_account]
# type = "service_account"
# project_id = "tu-proyecto"
# private_key_id = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
# private_key = """-----BEGIN PRIVATE KEY-----
# ...
# -----END PRIVATE KEY-----
# """
# client_email = "nombre-cuenta@tu-proyecto.iam.gserviceaccount.com"
# client_id = "xxxxxxxxxxxxxxxxxxxxx"
# token_uri = "https://oauth2.googleapis.com/token"
#
# Recuerda: comparte la carpeta de Drive con el `client_email` de arriba
# (como Lector) para que la cuenta de servicio pueda leer el CSV.
# ==============================================================================
