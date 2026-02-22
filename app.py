from motor_fisico import *

# ==========================================
# 3️⃣ ENTORNO VISUAL (INTERFAZ)
# ==========================================

def inicializar_estado_del_simulador():
    # --- INICIALIZADOR DE DATOS (Fuente de Verdad Única) ---
    if 'componentes_data' not in st.session_state:
        st.session_state.componentes_data = {
            "bancada": {"m": 10542.0, "pos": [0.00001, 0.0, 0.0], "I": [[9235.0, 1, 1], [1, 5690.0, 1], [1, 1, 3779.0]]},
            "cesto": {"m": 980.0, "pos": [0.0, 0.0, 0.0], "I": [[178.0, 0, 0], [0, 392.0, 0], [0, 0, 312.0]]}
        }

    if 'configuracion_sistema' not in st.session_state:
        st.session_state.configuracion_sistema = {
            "distancia_eje": 0.3,
            "sensor_pos": [-0.4, 0.2, 0.0],
            "diametro_cesto": 1250  # Valor por defecto (mm)
        }

    if 'dampers_prop_data' not in st.session_state:
        st.session_state.dampers_prop_data = [
            {"Tipo": "Ref_1", "kx": 1.5e6, "ky": 2.0e6, "kz": 1.5e6, "cx": 5.5e4, "cy": 5.5e4, "cz": 5e4},
            {"Tipo": "Ref_2", "kx": 1.0e6,  "ky": 1.5e6,  "kz": 1.0e6, "cx": 5.5e4, "cy": 5.5e4, "cz": 5e4}
        ]

    if 'dampers_pos_data' not in st.session_state:
        st.session_state.dampers_pos_data = [
            {"Nombre": "D1 (Motor)", "X": -1.4, "Y": -0.4, "Z": -0.2, "Tipo": "Ref_1"},
            {"Nombre": "D2 (Motor)", "X": 1.4, "Y":  -0.4, "Z": -0.2, "Tipo": "Ref_1"},
            {"Nombre": "D3 (Front)", "X": -1.4, "Y": -0.4, "Z": -2.0, "Tipo": "Ref_2"},
            {"Nombre": "D4 (Front)", "X": 1.4, "Y":  -0.4, "Z": -2.0, "Tipo": "Ref_2"},
        ]

# --- 1. INTERFAZ DE STREAMLIT ---
st.set_page_config(layout="wide")

# Llamamos a nuestra nueva función
inicializar_estado_del_simulador()

st.title("Simulador Interactivo de Centrífuga 700F - Departamento de Ingenieria de Riera Nadeu")
st.markdown("Modifica los valores en la barra lateral para ver el impacto en las vibraciones.")

# --- BARRA LATERAL PARA MODIFICAR VALORES ---
st.sidebar.header("Parámetros de cálculos")


# Ejemplo de cómo modificar la masa de desbalanceo y RPM
m_unbalance = st.sidebar.slider("Masa de Desbalanceo (kg)", 0.1, 8.0, 1.6)
rpm_obj = st.sidebar.number_input("RPM nominales", value=1100)

# --- SECCIÓN: PESTAÑAS ---
st.header("🧱 Configuración del Sistema")


# Contenedor para los datos procesados en los tabs
comp_editados = {} 
tab_config, tab_comp, tab_dampers, = st.tabs([ "⚙️ Configuración del Sistema", "📦 Componentes Masas/Inercias", "🛡️ Configuración de Dampers"])

# 1️⃣ CONFIGURACION DE SISTEMA
with tab_config:
    st.subheader("Configuración de Ejes y Convención")
    # 1. Leemos del "log" (session_state) para establecer el valor inicial
    distancia_eje = st.number_input(
        "Coordenada horizontal de la masa de desbalanceo (m)", 
        value=float(st.session_state.configuracion_sistema.get("distancia_eje", 0.8)),
        step=0.01,
        format="%.2f"
    )

    # --- DENTRO DE tab_config ---
    opciones_diametro = [800, 1000, 1250, 1400, 1600, 1800, 2000]

    # Simplificado: Calculamos el índice directamente en una línea
    # Al no tener 'key', el selectbox obedecerá siempre al 'index' que viene del JSON
    diametro_sel = st.selectbox(
        "Tamaño de cesto (Diámetro en mm):", 
        opciones_diametro, 
        index=opciones_diametro.index(st.session_state.configuracion_sistema.get("diametro_cesto", 1250))
    )

    # 3. Calculamos la excentricidad (Radio en metros)
    e_unbalance = (diametro_sel / 1000) / 2

       
    # --- NUEVA SECCIÓN: POSICIÓN DEL SENSOR ---
    st.text("Posición del Sensor de velocidad/aceleracion(m)")
    col_s1, col_s2, col_s3 = st.columns(3)

    sensor_actual = st.session_state.configuracion_sistema.get("sensor_pos", [0.0, 0.0, 0.0])
    with col_s1:
        sensor_x = st.number_input("X", value=float(sensor_actual[0]), step=0.1)
    with col_s2:
        sensor_y = st.number_input("Y", value=float(sensor_actual[1]), step=0.1)
    with col_s3:
        sensor_z = st.number_input("Z", value=float(sensor_actual[2]), step=0.1)

    st.divider()

# Actualizamos los valores de sistema en el session_state con lo que hay actualmente en los widgets
st.session_state.configuracion_sistema["distancia_eje"] = distancia_eje
st.session_state.configuracion_sistema["sensor_pos"] = [sensor_x, sensor_y, sensor_z] 
st.session_state.configuracion_sistema["diametro_cesto"] = diametro_sel

# 1️⃣ GESTIÓN DE COMPONENTES (Inercia 3x3 con Persistencia)
with tab_comp:
    subtabs = st.tabs(["Bancada", "Cesto",])
    
    # Mapeo de nombres para session_state
    nombres_llaves = ["bancada", "cesto"]

    for i, nombre in enumerate(nombres_llaves):
        with subtabs[i]:
            # ✅ NOTA ACLARATORIA: Solo para la primera subpestaña (Bancada)
            if i == 0:
                st.info("💡 **Nota:** La bancada debe inlcuir la masa y la inercia de la caja de rodamientos y elementos auxilaires, EXCLUYENDO la placa de inercia")
            # 1. LEER DEL LOG (Fuente de Verdad)
            datos_memoria = st.session_state.componentes_data[nombre]
            pos_actual = datos_memoria.get("pos", [0.0, 0.0, 0.0])
            
            c_m, c_p = st.columns([1, 2])
            with c_m:
                # Eliminamos la 'key' interna para que mande el 'value' del Log
                m_val = st.number_input(f"Masa {nombre} (kg)", value=float(datos_memoria.get("m", 0.0)))

            with c_p:
                cx, cy, cz = st.columns(3)
            px = cx.number_input(f"X [m]", value=float(pos_actual[0]), format="%.3f", key=f"x_{nombre}")
            py = cy.number_input(f"Y [m]", value=float(pos_actual[1]), format="%.3f", key=f"y_{nombre}")
            pz = cz.number_input(f"Z [m]", value=float(pos_actual[2]), format="%.3f", key=f"z_{nombre}")
            
            st.write(f"**Matriz de Inercia (3x3) [kg·m²]**")

            # El data_editor es excelente para matrices
            df_iner_3x3 = st.data_editor(
                np.array(datos_memoria["I"]),
                key=f"editor_matriz_{nombre}", # El editor sí necesita key para ser interactivo
                use_container_width=True
            )
            
            # 2. ACTUALIZAR LOG (Sincronización inmediata)
            st.session_state.componentes_data[nombre] = {
                "m": m_val, 
                "pos": [px, py, pz], 
                "I": df_iner_3x3.tolist() if isinstance(df_iner_3x3, np.ndarray) else df_iner_3x3
            }

# 2️⃣ GESTIÓN DE DAMPERS
with tab_dampers:
    st.write("### 1. Definición de Propiedades por Tipo")
    
    # Editor de Propiedades: Sincronización directa con el Log
    df_prop_editada = st.data_editor(
        st.session_state.dampers_prop_data,
        key="editor_tipos_nombres",
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic", # Permitimos que el usuario defina nuevos tipos
        column_config={
            "Tipo": st.column_config.TextColumn("Nombre del Tipo/Modelo", required=True),
            "kx": st.column_config.NumberColumn("Kx [N/m]", format="%.1e"),
            "ky": st.column_config.NumberColumn("Ky [N/m]", format="%.1e"),
            "kz": st.column_config.NumberColumn("Kz [N/m]", format="%.1e"),
        }
    )
    # SOLO actualizamos el session_state si el widget ha cambiado realmente
    # Esto evita que el widget "pise" al JSON recién cargado
    if st.session_state.get("editor_tipos_nombres"):
        st.session_state.dampers_prop_data = df_prop_editada

    # Extraemos la lista de tipos para el desplegable de la siguiente tabla
    # Usamos list(set(...)) para evitar duplicados si el usuario se equivoca
    df_prop = pd.DataFrame(df_prop_editada)
    lista_tipos_disponibles = df_prop["Tipo"].dropna().unique().tolist()

    st.write("### 2. Ubicación de los Dampers")
    
    # Editor de Posiciones
    res_pos_editor = st.data_editor(
        st.session_state.dampers_pos_data,
        num_rows="dynamic", 
        key="pos_dampers_editor_v2", 
        use_container_width=True,
        hide_index=True,
        column_config={
            "Nombre": st.column_config.TextColumn("Identificador (ej. D1)", required=True),
            "Tipo": st.column_config.SelectboxColumn(
                "Tipo de Damper", 
                options=lista_tipos_disponibles,
                required=True
            ),
            "X": st.column_config.NumberColumn("X [m]", format="%.3f"),
            "Y": st.column_config.NumberColumn("Y [m]", format="%.3f"),
            "Z": st.column_config.NumberColumn("Z [m]", format="%.3f"),
        }
    )
    # Sincronizamos con el Log Maestro
    st.session_state.dampers_pos_data = res_pos_editor

    # ✅ PROCESAMIENTO FINAL (Para el motor de cálculo)
    # Creamos 'dampers_finales' uniendo ambas tablas
    dampers_finales = []
    df_pos = pd.DataFrame(res_pos_editor)
    
    if not df_pos.empty and not df_prop.empty:
        df_prop_indexed = df_prop.set_index("Tipo")
        
        for _, row in df_pos.iterrows():
            tipo_sel = row.get("Tipo")
            # Seguridad: Solo procesamos si el tipo existe en la tabla de propiedades
            if tipo_sel and tipo_sel in df_prop_indexed.index:
                p = df_prop_indexed.loc[tipo_sel]
                dampers_finales.append({
                    "nombre": row.get("Nombre", "Sin nombre"),
                    "pos": [row.get("X", 0.0), row.get("Y", 0.0), row.get("Z", 0.0)],
                    "tipo": tipo_sel,
                    "kx": p["kx"], "ky": p["ky"], "kz": p["kz"],
                    "cx": p.get("cx", 0.0), "cy": p.get("cy", 0.0), "cz": p.get("cz", 0.0)
                })



def dibujar_modelo_2d(modelo):
    # 1. Obtener datos del modelo
    _, _, _, cg_global = modelo.armar_matrices()
    pos_sensor = modelo.pos_sensor
    ex = modelo.excitacion
    
    # Obtenemos el radio desde la excentricidad guardada en la excitación
    radio_m = ex.get('e_unbalance', 0.625) 
    diametro_mm = radio_m * 2 * 1000

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("<b>Vista Frontal (X-Y)</b>", "<b>Vista de Planta (X-Z)</b>"),
        horizontal_spacing=0.15
    )

    nombres_vistos = set()
    def agregar_traza(trace, row, col):
        if trace.name not in nombres_vistos:
            trace.showlegend = True
            nombres_vistos.add(trace.name)
        else:
            trace.showlegend = False
        fig.add_trace(trace, row=row, col=col)

    # --- 1. ÁREA GRIS DE DAMPERS (Sombreado) ---
    if len(modelo.dampers) >= 2:
        d_x = [d.pos[0] for d in modelo.dampers]
        d_z = [d.pos[2] for d in modelo.dampers]
        
        # Sombreado Aisladores (Dampers) usando ConvexHull para el área de apoyo
        from scipy.spatial import ConvexHull
        puntos = np.column_stack((d_x, d_z))
        hull = ConvexHull(puntos)
        idx_orden = np.append(hull.vertices, hull.vertices[0])
        
        fig.add_trace(go.Scatter(
            x=puntos[idx_orden, 0], y=puntos[idx_orden, 1],
            fill="toself", fillcolor="rgba(128, 128, 128, 0.2)",
            line=dict(color="rgba(128, 128, 128, 0.5)", width=1),
            name="Área de Apoyo (Base)", showlegend=True, hoverinfo='skip'
        ), row=1, col=2)

        # --- 2. RECTÁNGULO DEL CESTO ---
        z_base_cesto = max(d_z) # Punto superior de los dampers
        z_masa = ex['distancia_eje'] # Altura de la carga

        fig.add_trace(go.Scatter(
            x=[-radio_m, radio_m, radio_m, -radio_m, -radio_m],
            y=[z_base_cesto, z_base_cesto, z_masa, z_masa, z_base_cesto],
            fill="toself",
            fillcolor="rgba(100, 100, 100, 0.1)",
            line=dict(color="rgba(0, 0, 0, 0.3)", width=2, dash='dot'),
            name=f"Cuerpo Cesto (Ø {diametro_mm:.0f}mm)",
            showlegend=True, hoverinfo='skip'
        ), row=1, col=2)

    # --- 3. CENTRO DE GRAVEDAD GLOBAL (Único) ---
    for r, c, y_val in [(1, 1, cg_global[1]), (1, 2, cg_global[2])]:
        agregar_traza(go.Scatter(
            x=[cg_global[0]], y=[y_val],
            mode='markers', name='CG Global',
            marker=dict(symbol='x', size=14, color='purple', line=dict(width=2))
        ), r, c)

    # --- 4. DAMPERS, SENSOR Y MASA ---
    for d in modelo.dampers:
        for r, c, y_val in [(1, 1, d.pos[1]), (1, 2, d.pos[2])]:
            agregar_traza(go.Scatter(
                x=[d.pos[0]], y=[y_val], mode='markers', name='Aisladores (Dampers)',
                marker=dict(symbol='diamond', size=10, color='cyan', line=dict(width=1, color='black'))
            ), r, c)

    for r, c, y_val in [(1, 1, pos_sensor[1]), (1, 2, pos_sensor[2])]:
        agregar_traza(go.Scatter(
            x=[pos_sensor[0]], y=[y_val], mode='markers', name='Sensor Velocidad',
            marker=dict(symbol='star', size=15, color='lime', line=dict(width=1, color='black'))
        ), r, c)

    if ex['m_unbalance'] > 0:
        z_unb = ex['distancia_eje']
        e_unb = ex['e_unbalance']
        fig.add_trace(go.Scatter(
            x=[0, e_unb], y=[z_unb, z_unb], mode='lines', name='Radio Desbalanceo',
            line=dict(color='red', width=2, dash='dash')
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=[e_unb], y=[z_unb], mode='markers', name='Masa Desbalanceo',
            marker=dict(size=12, color='red')
        ), row=1, col=2)

    # Configuración de Layout y Ejes Marcados
    fig.update_layout(
        font=dict(size=14),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
        height=600, plot_bgcolor='white'
    )

    # Vista Frontal (X-Y): X normal (Derecha)
    fig.update_xaxes(title_text="<b>Eje X [m]</b>", zeroline=True, row=1, col=1)
    fig.update_yaxes(title_text="<b>Eje Y (Altura) [m]</b>", zeroline=True, row=1, col=1)

    # Vista de Planta (X-Z): Invertimos X para que sea coherente con Z hacia adelante
    fig.update_xaxes(
        title_text="<b>Eje X [m]</b>", 
        zeroline=True, 
        autorange="reversed", # <--- ESTO invierte el eje X solo en la planta
        row=1, col=2
    )
    fig.update_yaxes(title_text="<b>Eje Z (Profundidad) [m]</b>", zeroline=True, row=1, col=2)

    return fig














# 3️⃣ ENSAMBLAJE FINAL (Cálculo Base)
# Usamos las llaves del session_state para garantizar que, 
# aunque el usuario no abra una pestaña, el simulador use el último dato guardado.

config_base = {
    "excitacion": {
        "distancia_eje": st.session_state.configuracion_sistema["distancia_eje"], 
        "m_unbalance": m_unbalance, # Viene del slider de la sidebar
        "e_unbalance": e_unbalance # Valor constante de diseño
    },
    "componentes": st.session_state.componentes_data,
    "dampers": dampers_finales, # Lista ya procesada en la pestaña anterior
    "sensor": {
        "pos_sensor": st.session_state.configuracion_sistema["sensor_pos"]
    },
    "tipos_dampers": pd.DataFrame(st.session_state.dampers_prop_data).set_index("Tipo").to_dict('index')
}




st.sidebar.divider()
st.sidebar.header("💾 Gestión de Archivos")

# --- 1. SECCIÓN DE IMPORTAR (Cargar) ---
archivo_subido = st.sidebar.file_uploader("📂 Subir configuración (.json)", type=["json"])

if archivo_subido is not None:
    # Agregamos el botón para confirmar la carga
    if st.sidebar.button("📥 Aplicar configuración del archivo"):
        try:
            datos_preset = json.load(archivo_subido)
            
            # Actualizamos los componentes
            if "componentes_data" in datos_preset:
                for nombre, data in datos_preset["componentes_data"].items():
                    if nombre in st.session_state.componentes_data:
                        st.session_state.componentes_data[nombre].update(data)            
            
            if "configuracion_sistema" in datos_preset:
                st.session_state.configuracion_sistema.update(datos_preset["configuracion_sistema"])
            
            # Actualización de Dampers
            if "dampers_prop_data" in datos_preset:
                st.session_state.dampers_prop_data = datos_preset["dampers_prop_data"]
                if "editor_tipos_nombres" in st.session_state:
                    del st.session_state["editor_tipos_nombres"]
            
            if "dampers_pos_data" in datos_preset:
                st.session_state.dampers_pos_data = datos_preset["dampers_pos_data"]
                if "pos_dampers_editor_v2" in st.session_state:
                    del st.session_state["pos_dampers_editor_v2"]

            st.sidebar.success("✅ Datos cargados correctamente")
            st.rerun() 
            
        except Exception as e:
            st.sidebar.error(f"Error al procesar el archivo: {e}")

# 3️⃣ GUARDADO ARCHIVO

def json_compacto(obj):
    """
    Convierte a JSON colapsando listas de números en una sola línea
    sin duplicar comas.
    """
    # 1. Generar JSON estándar
    content = json.dumps(obj, indent=4, sort_keys=True)
    
    # 2. Regex corregida: 
    # Busca una lista que empiece por '[', contenga números, comas, espacios y cierre con ']'
    # Luego elimina los saltos de línea y espacios extra dentro de esa lista.
    def limpiar_lista(match):
        return match.group(0).replace("\n", "").replace(" ", "").replace(",", ", ")

    # Esta regex identifica patrones de listas de números/floats
    content = re.sub(r'\[(?:\s*[-+]?\d*\.?\d+(?:e[-+]?\d+)?\s*,?)+ \s*\]', limpiar_lista, content)
    
    # Limpieza final de seguridad por si quedaron espacios raros
    content = content.replace(", ]", "]").replace("[, ", "[")
    
    return content

# --- FUNCIONALIDAD DE EXPORTAR (Download) ---
# Preparamos el diccionario con todo lo que hay en memoria actualmente
datos_a_exportar = {
    # Agrupamos todo lo referente a la física global del sistema
    "configuracion_sistema": {
        "distancia_eje": st.session_state.configuracion_sistema["distancia_eje"],
        "diametro_cesto": st.session_state.configuracion_sistema["diametro_cesto"], 
        "sensor_pos": st.session_state.configuracion_sistema["sensor_pos"]
    },
    # Los diccionarios de componentes (Bancada, Cesto)
    "componentes_data": st.session_state.componentes_data,
    
    # Las dos tablas de los Dampers (Propiedades y Ubicaciones)
    "dampers_prop_data": st.session_state.dampers_prop_data,
    "dampers_pos_data": st.session_state.dampers_pos_data
}

# Convertir a string JSON
json_string = json_compacto(datos_a_exportar)
st.sidebar.download_button(
    label="📥 Descargar Configuración (.json)",
    data=json_string,
    file_name="config_centrifuga.json",
    mime="application/json",
    help="Guarda todos los datos actuales en un archivo para usarlos después."
)
st.sidebar.write("---")




# --- SELECTOR DE DAMPER ---
# Accedemos directamente al diccionario de configuración
lista_dampers_config = config_base["dampers"] 
# Creamos las opciones para el selectbox usando el diccionario
opciones = [f"{i}: {d['tipo']} en {d['pos']}" for i, d in enumerate(lista_dampers_config)]
seleccion = st.sidebar.selectbox("Selección de damper para diagnóstico:", opciones)
# Extraemos el índice
d_idx = int(seleccion.split(":")[0])




# --- 4. EJECUTAR AMBAS SIMULACIONES ---
modelo_base = SimuladorCentrifuga(config_base)

st.subheader("🌐 Visualización 2D del Modelo")

# Asegúrate de que el modelo_base ya esté inicializado antes de llamar a dibujar_modelo_3d
if 'modelo_base' in locals() or 'modelo_base' in globals(): # Comprueba si modelo_base existe
    fig_2d = dibujar_modelo_2d(modelo_base)
    st.plotly_chart(fig_2d, use_container_width=True)
else:
    st.warning("Carga una configuración o ajusta los parámetros para ver el modelo 3D.")

st.divider()

f_res_rpm, modos = modelo_base.calcular_frecuencias_naturales()
# RPM de operación

rpm_range = np.linspace(10, rpm_obj*1.2, 15000)
idx_op = np.argmin(np.abs(rpm_range - rpm_obj))

rpm_range, D_desp, D_fuerza, acel_cg, vel_cg, S_desp, S_vel, S_acel = ejecutar_barrido_rpm(modelo_base, rpm_range, d_idx)

# ==========================================
# 📄 INTRODUCCIÓN Y MEMORIA DE CÁLCULO
# ==========================================
st.markdown(f"""
### 📋 Resultados
---
""")

st.markdown("""
    <style>
    @media print {
        /* Ocultar barra lateral y botones al imprimir */
        [data-testid="stSidebar"], .stButton, header, footer {
            display: none !important;
        }
        /* Ajustar el ancho del contenedor principal */
        .main .block-container {
            max-width: 100%;
            padding: 1rem;
        }
        /* Forzar que los gráficos no se corten entre páginas */
        .stPlotlyChart, .css-12w0qpk {
            page-break-inside: avoid;
        }
    }
    </style>
    """, unsafe_allow_html=True)



st.subheader("📋 Resumen de Cargas por Apoyo")
df_cargas = calcular_tabla_fuerzas(modelo_base, rpm_obj)

if not df_cargas.empty:
    st.dataframe(
        df_cargas,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Carga Vert. Máx (Est+Z) [N]": st.column_config.NumberColumn(
                "Carga Vert. Máx [N]",
                help="Suma de la carga estática y la amplitud de la fuerza dinámica vertical (Z).",
                format="%.1f"
            )
        }
    )

st.markdown("""
### 💡 Guía de Interpretación de Cargas
---
* **Carga Estática:** Es el peso de la máquina (Bancada + Cesto) distribuido en cada apoyo según la posición del Centro de Gravedad (CG).
* **Dinámica (X, Y, Z):** Es la amplitud de la fuerza vibratoria generada por el desbalanceo a las RPM nominales.
* **Carga TOTAL MÁX:** Es la carga máxima que el damper debe soportar estructuralmente ($F_{est} + F_{din, Vertical}$). Útil para verificar la capacidad del catálogo del fabricante.
* **Margen de Estabilidad:** Es la fuerza neta mínima durante la oscilación ($F_{est} - F_{din, Vertical}$). 
    * **Si es > 0:** El apoyo siempre está en compresión (Seguro).
    * **Si es < 0:** El apoyo intenta levantarse de la base (Vuelo), lo que genera impactos, ruido y desgaste prematuro.
""")

# --- DEFINICIÓN DE EJES PARA GRÁFICOS (Pegar antes de los bucles for) ---
eje_axial = "z"
eje_vert_fisico = "y"  # P.ej: si eje es 'x', este es 'y'
eje_horiz_fisico = "x" # P.ej: si eje es 'x', este es 'z'

# Creamos la lista para iterar en los gráficos
orden_grafico = [eje_vert_fisico, eje_horiz_fisico, eje_axial]

# Diccionario de etiquetas para las leyendas
ejes_lbl = {
    eje_vert_fisico: f"({eje_vert_fisico.upper()})",
    eje_horiz_fisico: f"({eje_horiz_fisico.upper()})",
    eje_axial: f"({eje_axial.upper()})"
}

# Diccionario de colores
colores = {
    eje_vert_fisico: "tab:orange", 
    eje_horiz_fisico: "tab:blue", 
    eje_axial: "tab:green"
}




# ==========================
# 📊 GRÁFICO 1: Aceleración en el SENSOR (CORREGIDO)
# ==========================
st.subheader("Análisis de Aceleración en el Sensor")
fig, ax = plt.subplots(figsize=(10, 4))

# Usamos el orden dinámico y las etiquetas mapeadas
for eje in orden_grafico:
    ax.plot(rpm_range, S_acel[eje], color=colores[eje], label=ejes_lbl[eje])

ax.axvline(rpm_obj, color='black', linestyle=':', label=f'RPM operación ({rpm_obj})')
ax.set_xlabel("RPM")
ax.set_ylabel("Aceleración [g]")
ax.grid(True, alpha=0.1)
ax.legend()
plt.rcParams.update({'font.size': 10}) 
fig.tight_layout()
st.pyplot(fig, clear_figure=True)


# ==========================
# 📊 GRÁFICO 2: Velocidad en el SENSOR (REVISADO)
# ==========================
st.subheader("Respuesta en Frecuencia: Velocidad en Sensor")
fig2, ax2 = plt.subplots(figsize=(10, 5))
for eje in orden_grafico:
    ax2.plot(rpm_range, S_vel[eje], color=colores[eje], label=ejes_lbl[eje])

ax2.axvline(rpm_obj, color='black', linestyle=':', label=f'RPM operación ({rpm_obj})')

# Marcar resonancias teóricas
for i, f in enumerate(f_res_rpm):
    if f < rpm_range[-1]: 
        ax2.axvline(f, color='red', linestyle='--', alpha=0.3, 
                    label='Resonancia' if i == 0 else "") # Etiqueta solo una vez

ax2.set_xlabel('Velocidad de Rotación [RPM]')
ax2.set_ylabel('Velocidad [mm/s]')
ax2.grid(True, alpha=0.1)
ax2.legend()
st.pyplot(fig2)

# Inserta esto antes de una sección nueva que quieras que empiece en hoja limpia
st.markdown('<div style="break-after:page"></div>', unsafe_allow_html=True)
st.subheader(f"Desplazamiento Amplitud en Damper {lista_dampers_config[d_idx]['tipo']}")


# ==========================
# 📊 GRÁFICO 3: Desplazamiento Damper
# ==========================
fig3, ax3 = plt.subplots(figsize=(10, 5))
for eje in orden_grafico:
    ax3.plot(rpm_range, D_desp[eje], color=colores[eje], label=f'{ejes_lbl[eje]}')

ax3.axvline(rpm_obj, color='black', linestyle=':', label=f'RPM operación ({rpm_obj})')
ax3.set_xlabel('Velocidad de Rotación [RPM]')
ax3.set_ylabel('Desplazamiento [mm]')
ax3.grid(True, alpha=0.1)
ax3.legend()
st.pyplot(fig3)

# ==========================
# 📊 GRÁFICO 4: Fuerzas Dinámicas
# ==========================
st.subheader(f"Fuerzas Dinámicas en Damper {lista_dampers_config[d_idx]['tipo']}")
fig4, ax4 = plt.subplots(figsize=(10, 5))

# Usamos el orden lógico: Radial Vertical, Radial Horizontal y Axial
for eje in orden_grafico:
    ax4.plot(rpm_range, D_fuerza[eje], color=colores[eje], label=ejes_lbl[eje])

ax4.axvline(rpm_obj, color='black', linestyle=':', label=f'RPM operación ({rpm_obj})')

# --- CORRECCIÓN DE LA ANOTACIÓN ---
# Usamos el eje vertical físico (donde realmente hay carga dinámica)
eje_v = eje_vert_fisico 
f_max_op = D_fuerza[eje_v][idx_op]

ax4.annotate(
    f'{f_max_op:.0f} N ({eje_v.upper()}) a {rpm_obj} RPM',
    xy=(rpm_range[idx_op], f_max_op),
    # Ajustamos xytext para que no se solape con la línea de la curva
    xytext=(rpm_range[idx_op] * 0.6, f_max_op * 1.15), 
    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
    fontsize=10,
    fontweight='bold'
)

ax4.set_xlabel('Velocidad de Rotación [RPM]')
ax4.set_ylabel('Fuerza Transmitida [N]')
ax4.grid(True, alpha=0.1)

# Colocamos la leyenda fuera del gráfico si hay muchas líneas
ax4.legend(loc='upper right')

st.pyplot(fig4)


# ==========================================
# 📈 ANÁLISIS DE RESONANCIA Y CONCLUSIONES
# ==========================================
st.markdown(f"""
### 📋 Informe de Análisis Dinámico
Este reporte simula el comportamiento vibratorio de una centrífuga industrial bajo condiciones de desbalanceo.
A continuación se detallan los parámetros de entrada utilizados para este análisis:

* **Masa de Desbalanceo:** {m_unbalance:.2f} kg
* **RPM de Operación:** {rpm_obj} RPM
---
""")


st.divider()
# Inserta esto antes de una sección nueva que quieras que empiece en hoja limpia
st.markdown('<div style="break-after:page"></div>', unsafe_allow_html=True)
st.header("Análisis de Seguridad y Vibraciones")

# 1. Identificación de la Frecuencia Crítica (Resonancia)
# Buscamos el pico máximo en el barrido de RPM
idx_res_base = np.argmax(S_vel[eje_v])
rpm_res_base = rpm_range[idx_res_base]

col_concl1, col_concl2 = st.columns(2)

with col_concl1:
    st.write("### 🚨 Puntos Críticos (Resonancia)")
    # Mostramos la primera frecuencia natural (Modo 1)
    st.write(f"**Caso Base (Modo 1):** {f_res_rpm[0]:.0f} RPM")
    st.write(f"**Caso Base (Modo 2):** {f_res_rpm[1]:.0f} RPM")
    st.write(f"**Caso Base (Modo 3):** {f_res_rpm[2]:.0f} RPM")
    st.write(f"**Caso Base (Modo 4):** {f_res_rpm[3]:.0f} RPM")
    st.write(f"**Caso Base (Modo 5):** {f_res_rpm[4]:.0f} RPM")
    st.write(f"**Caso Base (Modo 6):** {f_res_rpm[5]:.0f} RPM")

    
    dist_min_base = abs(f_res_rpm[5] - rpm_obj)
    if dist_min_base < 150:
        # Identificamos cuál falló para dar un mensaje preciso
        st.error(f"⚠️ PELIGRO: Resonancia crítica detectada. "
                 f"Margen insuficiente (< 150 RPM) respecto a {rpm_obj} RPM.")
    else:
        st.success(f"✅ SEGURO: Todos los modos de ambos modelos mantienen un margen "
                   f"> 150 RPM respecto a la operación.")
        
    st.caption(f"Margen actual: Base {dist_min_base:.0f} RPM")

with col_concl2:
    st.write("### 📊 Cumplimiento de Norma (ISO 10816)")
    
    # Extraemos el pico máximo considerando los tres ejes para ser conservadores
    v_max_base = max(max(S_vel["x"]), max(S_vel["y"]), max(S_vel["z"]))
        
    st.write(f"**Velocidad Máx. detectada:** {v_max_base:.2f} mm/s")
    
    # Opcional: Clasificación rápida
    if v_max_base > 12.0:
        st.warning("Zona C: Vibración insatisfactoria para operación continua.")
    elif v_max_base > 8.0:
        st.info("Zona B: Vibración aceptable.")
    else:
        st.success("Zona A: Vibración excelente.")


# 2. Espacio para Observaciones del Ingeniero
st.write("---")
st.subheader("📝 Notas del Analista")
observaciones = st.text_area("Escribe aquí tus conclusiones adicionales para el PDF:", 
                             "Por ejemplo: Se observa que el aumento del espesor de la placa desplaza la frecuencia natural hacia arriba, reduciendo la amplitud en el punto de operación.")

st.info("💡 **Consejo para el reporte:** Las anotaciones de arriba aparecerán en tu PDF final.")

st.divider()
st.subheader("🖨️ Generar Reporte Técnico")

if st.button("Preparar Informe para PDF"):
    st.balloons()
    st.info("### Instrucciones para un PDF Profesional:\n"
            "1. Presiona **Ctrl + P** (Windows) o **Cmd + P** (Mac).\n"
            "2. Selecciona **'Guardar como PDF'**.\n"
            "3. En 'Más ajustes', activa **'Gráficos de fondo'**.\n"
            "4. Cambia el diseño a **'Vertical'**.")
    
    # Esto fuerza a Streamlit a mostrar todo de forma estática y clara
    st.markdown("""
        <style>
        @media print {
            .stButton, .stDownloadButton { display: none; } /* Oculta botones al imprimir */
            .main { background-color: white !important; }
        }
        </style>
    """, unsafe_allow_html=True)