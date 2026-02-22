import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import math
import copy
import json
import pandas as pd
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# 1️⃣ TUS CLASES
# ==========================================

class Damper:
    def __init__(self, nombre, pos, kx, ky, kz, cx, cy, cz):
        self.nombre = nombre
        self.pos = np.array(pos, dtype=float)
        self.kx, self.ky, self.kz = kx, ky, kz
        self.cx, self.cy, self.cz = cx, cy, cz

    def get_matriz_T(self, cg):
        d = self.pos - np.array(cg)
        lx, ly, lz = d
        return np.array([
            [1, 0, 0, 0,  lz, -ly],
            [0, 1, 0, -lz, 0,  lx],
            [0, 0, 1,  ly, -lx, 0]
        ])

    def get_matriz_K(self): return np.diag([self.kx, self.ky, self.kz])
    def get_matriz_C(self): return np.diag([self.cx, self.cy, self.cz])

class SimuladorCentrifuga:
    def __init__(self, config):
        self.pos_sensor = np.array(config["sensor"]["pos_sensor"])

        # --- Componentes ---
        self.componentes = {
            "cesto": config['componentes']['cesto'],
            "bancada": config['componentes']['bancada'],
        }

        # --- Excitación ---
        self.excitacion = config['excitacion']

        # --- Dampers ---
        self.dampers = []
        for d_conf in config['dampers']:
            nombre_instancia = d_conf.get('nombre', 'unnamed')
            
            # Buscamos las propiedades (kx, ky, etc.)
            # Prioridad 1: Que ya vengan en el diccionario del damper
            # Prioridad 2: Buscarlas en tipos_dampers usando el campo 'tipo'
            if 'kx' in d_conf:
                self.dampers.append(Damper(nombre_instancia, d_conf['pos'], 
                                           d_conf['kx'], d_conf['ky'], d_conf['kz'], 
                                           d_conf['cx'], d_conf['cy'], d_conf['cz']))
            else:
                tipo_nombre = d_conf['tipo']
                tipo_vals = config['tipos_dampers'][tipo_nombre]
                self.dampers.append(Damper(tipo_nombre, d_conf['pos'], **tipo_vals))

    def obtener_matriz_sensor(self, cg_global):
        r_p = self.pos_sensor - cg_global
        return np.array([
            [1, 0, 0, 0,  r_p[2], -r_p[1]],
            [0, 1, 0, -r_p[2], 0,  r_p[0]],
            [0, 0, 1,  r_p[1], -r_p[0], 0]
        ])

    def armar_matrices(self):

        if not self.componentes:
            st.error("No hay componentes cargados en el simulador.")
            return np.zeros((6,6)), np.zeros((6,6)), np.zeros((6,6)), np.array([0,0,0])

        # 2. Validación de contenido (evita el TypeError)
        componentes_validos = {k: v for k, v in self.componentes.items() if v is not None}
        
        if len(componentes_validos) < len(self.componentes):
            st.warning(f"Se detectaron {len(self.componentes) - len(componentes_validos)} componentes nulos.")

        m_total = sum(c["m"] for c in self.componentes.values())
        cg_global = sum(c["m"] * np.array(c["pos"]) for c in self.componentes.values()) / m_total

        M, I_global = np.zeros((6, 6)), np.zeros((3, 3))
        
        for nombre, c in self.componentes.items():
            m_c = c["m"]
            p_c = np.array(c["pos"])
            # Inercia local (convertir a matriz 3x3 si es lista)
            #I_local = np.diag(c["I"]) if isinstance(c["I"], list) else np.array(c["I"])

            I_local = np.array(c["I"], dtype=float)

            if I_local.shape != (3, 3):
                st.error(f"Error en '{nombre}': La inercia debe ser matriz 3x3. Recibido: {I_local.shape}")
                continue

            # Vector desde el CG global al CG del componente
            d = p_c - cg_global
        
            # Teorema de Steiner (Ejes Paralelos) en forma matricial
            # I_global = sum( I_local + m * [ (d·d)diag(1) - (d ⊗ d) ] )
            term_steiner = m_c * (np.dot(d, d) * np.eye(3) - np.outer(d, d))

            # Verificación de simetría real antes de sumar
            matriz_c = I_local + term_steiner
            if not np.allclose(matriz_c, matriz_c.T, atol=1e-5):
                st.error(f"Asimetría detectada en componente: {nombre}")
                # Esto nos dirá si es I_local o Steiner el culpable
                st.write(f"Asimetría I_local: {np.max(np.abs(I_local - I_local.T))}")
                st.write(f"Asimetría Steiner: {np.max(np.abs(term_steiner - term_steiner.T))}")

            I_global += (I_local + term_steiner)

        M[0:3, 0:3], M[3:6, 3:6] = np.eye(3) * m_total, I_global

        K, C = np.zeros((6, 6)), np.zeros((6, 6))
        K += np.eye(6) * 1e-6
        for damper in self.dampers:
            T = damper.get_matriz_T(cg_global)
            K += T.T @ damper.get_matriz_K() @ T
            C += T.T @ damper.get_matriz_C() @ T

        # 1. Masa total
        if m_total <= 0:
            st.error("❌ Error Crítico: La masa total es cero o negativa.")

        # 2. Determinante de M (Corregido el error de sintaxis)
        det_M = np.linalg.det(M)
        if abs(det_M) < 1e-9:
            st.warning(f"⚠️ Determinante de M muy bajo ({det_M:.2e}): El sistema es físicamente imposible o singular.")

        # 3. Inercia Definida Positiva (Cholesky)
        try:
            np.linalg.cholesky(I_global) 
        except np.linalg.LinAlgError:
            st.error("🚨 ¡Inestabilidad Numérica en I_global!")
            evs = np.linalg.eigvals(I_global)
            st.write("Autovalores de I_global (deben ser todos > 0):", evs)

        # 4. Condicionamiento
        cond_M = np.linalg.cond(M)
        if cond_M > 1e12:
            st.warning(f"⚠️ Matriz de Masa mal condicionada (Cond: {cond_M:.2e}).")

        return M, K, C, cg_global



    def calcular_frecuencias_naturales(self):
        # Todo este bloque debe tener la misma sangría inicial (4 espacios)
        M, K, C, _ = self.armar_matrices()

        # Cálculo del problema de autovalores generalizado
        # K * v = λ * M * v
        evals, evecs = linalg.eigh(K, M)
        
        # Limpieza de valores por precisión numérica
        evals = np.maximum(evals, 0)
        
        # Frecuencias angulares (rad/s)
        w_n = np.sqrt(evals)
        
        # Convertir a Hz y a RPM
        f_hz = w_n / (2 * np.pi)
        f_rpm = f_hz * 60

        return f_rpm, evecs



# ==========================================
# 2️⃣ LÓGICA DE CÁLCULO
# ==========================================

def ejecutar_barrido_rpm(modelo, rpm_range, d_idx):

    M, K, C, cg_global = modelo.armar_matrices()
    T_sensor = modelo.obtener_matriz_sensor(cg_global)

    # --- Preparación damper específico ---
    damper_d = modelo.dampers[d_idx]
    T_damper = damper_d.get_matriz_T(cg_global)
    ks = [damper_d.kx, damper_d.ky, damper_d.kz]
    cs = [damper_d.cx, damper_d.cy, damper_d.cz]

    ex = modelo.excitacion
    dist = ex['distancia_eje']

    acel_cg = {"x": [], "y": [], "z": []}
    D_fuerza = {"x": [], "y": [], "z": []}
    vel_cg  = {"x": [], "y": [], "z": []}
    D_desp  = {"x": [], "y": [], "z": []}
    S_desp = {"x": [], "y": [], "z": []}
    S_vel  = {"x": [], "y": [], "z": []}
    S_acel = {"x": [], "y": [], "z": []}

    for rpm in rpm_range:
        w = rpm * 2 * np.pi / 60
        F0 = ex['m_unbalance'] * ex['e_unbalance'] * w**2
        
        # 2. Inicialización del vector de excitación F (6 DOFs: Fx, Fy, Fz, Mx, My, Mz)
        F = np.zeros(6, dtype=complex)
            
        arm = ex['distancia_eje'] - cg_global[2]
        # Fuerzas en X e Y
        #F[0], F[1] = F0, F0 * 1j
        # Momentos: My debido a Fx, Mx debido a Fy
        # Mx = Fy * brazo | My = -Fx * brazo
        #F[3] = (F0 * 1j) * arm  # Momento en X
        #F[4] = -F0 * arm        # Momento en Y

        # =========================================================================
        # NOTA TÉCNICA SOBRE LA EXCITACIÓN (EJE Z HORIZONTAL)
        # =========================================================================
        # Para garantizar la simetría dinámica en los apoyos, la fuerza debe 
        # aplicarse respecto al EJE DE ROTACIÓN REAL (0, 0 en el plano X-Y).
        #
        # Si el CG_global está desplazado de este eje (excentricidad lateral), 
        # la fuerza centrífuga genera momentos adicionales (Mx, My, Mz) 
        # referidos al CG que el simulador debe resolver.
        #
        # Brazos de palanca desde el CG al punto de aplicación (en el eje):
        # lx = 0 - cg_global[0] 
        # ly = 0 - cg_global[1]
        # lz = dist - cg_global[2]
        #
        # Esto corrige el "conflicto de fases" y restaura la simetría en los 
        # resultados de los dampers cuando el sistema es geométricamente espejo.
        # =========================================================================

        # Implementación corregida en el vector F:
        lx_exc = -cg_global[0]
        ly_exc = -cg_global[1]
        lz_exc = dist - cg_global[2]

        F = np.array([
            F0,                     # Fx (Real)
            1j * F0,                # Fy (Imaginaria - Giro 90°)
            0,                      # Fz (Nula en desbalanceo radial)
            (1j * F0) * lz_exc,     # Mx = Fy*lz - Fz*ly
            -F0 * lz_exc,           # My = Fz*lx - Fx*lz
            F0 * ly_exc - (1j * F0) * lx_exc  # Mz = Fx*ly - Fy*lx (Momento Torsional)
        ])

        F[0], F[1] = F0, F0 * 1j
        # Momentos: My debido a Fx, Mx debido a Fy
        # Mx = Fy * brazo | My = -Fx * brazo
        F[3] = (F0 * 1j) * arm  # Momento en X
        F[4] = -F0 * arm        # Momento en Y


        # Resolver el sistema: Z * X = F
        Z = -w**2 * M + 1j*w * C + K
        X = linalg.solve(Z, F)
        # --- CG: aceleración y velocidad ---
        for i, eje in enumerate(["x", "y", "z"]):
          acel_cg[eje].append((w**2) * np.abs(X[i])/9.81)
          vel_cg[eje].append(w * np.abs(X[i]) * 1000)

        # --- Damper: desplazamiento y fuerza ---
        X_damper = T_damper @ X
        for i, eje in enumerate(["x", "y", "z"]):
          D_desp[eje].append(np.abs(X_damper[i]) * 1000)
          f_comp = (ks[i] + 1j * w * cs[i]) * X_damper[i]
          D_fuerza[eje].append(np.abs(f_comp))

        # --- Sensor: desplazamiento y fuerza ---
        U_sensor = T_sensor @ X
        for i, eje in enumerate(["x", "y", "z"]):
          # desplazamiento [mm]
          S_desp[eje].append(np.abs(U_sensor[i]) * 1000)
          # velocidad [mm/s]
          S_vel[eje].append(w * np.abs(U_sensor[i]) * 1000)
          # aceleración [g]
          S_acel[eje].append((w**2) * np.abs(U_sensor[i])/9.81)
    
    return rpm_range, D_desp, D_fuerza, acel_cg, vel_cg, S_desp, S_vel, S_acel


def calcular_tabla_fuerzas(modelo, rpm_obj):
    """
    Versión espejada de ejecutar_barrido_rpm para garantizar coincidencia total.
    Usa la convención: Vertical = Y, Planta = X-Z.
    """
    M, K, C, cg_global = modelo.armar_matrices()
    m_total = sum(c["m"] for c in modelo.componentes.values())
    peso_total = m_total * 9.81
    
    n_d = len(modelo.dampers)
    if n_d == 0: return pd.DataFrame()

    # --- 1. REPARTO ESTÁTICO (Consistente con Vertical = Y) ---
    A = np.zeros((3, n_d))
    b = np.array([peso_total, 0, 0])
    for i, d in enumerate(modelo.dampers):
        rx = d.pos[0] - cg_global[0]
        rz = d.pos[2] - cg_global[2]
        A[0, i] = 1        # Suma de fuerzas en Y
        A[1, i] = rz       # Momento en X (Brazo Z)
        A[2, i] = -rx      # Momento en Z (Brazo X)

    reacciones_estaticas = np.linalg.pinv(A) @ b

    # --- 2. CÁLCULO DINÁMICO (Copia exacta de la lógica del barrido) ---
    w = rpm_obj * 2 * np.pi / 60
    ex = modelo.excitacion
    F0 = ex['m_unbalance'] * ex['e_unbalance'] * w**2
    
    F = np.zeros(6, dtype=complex)
    # Brazo en Z como en tu barrido: arm = dist - cg_global[2]
    arm = ex['distancia_eje'] - cg_global[2] 
    
    # Fuerzas en X e Y como en tu barrido
    F[0], F[1] = F0, F0 * 1j
    # Momentos Mx y My como en tu barrido
    F[3] = -(F0 * 1j) * arm  # Momento en X
    F[4] = F0 * arm        # Momento en Y
    F[5] =  0

    Z = -w**2 * M + 1j*w * C + K
    X = linalg.solve(Z, F)

    resumen = []
    for i, d in enumerate(modelo.dampers):
        T_d = d.get_matriz_T(cg_global)
        X_d = T_d @ X
        ks, cs = [d.kx, d.ky, d.kz], [d.cx, d.cy, d.cz]
        
        # Amplitudes dinámicas (idéntico al barrido)
        f_din = [np.abs((ks[j] + 1j * w * cs[j]) * X_d[j]) for j in range(3)]



        f_est_y = reacciones_estaticas[i]
        
        # En tu barrido la fuerza vertical es el eje Y (índice 1)
        resumen.append({
            "Damper": d.nombre,
            "Carga Estática [N]": round(f_est_y, 1),
            "Dinámica X [N]": round(f_din[0], 1),
            "Dinámica Y [N]": round(f_din[1], 1),
            "Dinámica Z [N]": round(f_din[2], 1),
            "Carga TOTAL MÁX [N]": round(f_est_y + f_din[1], 1),
            "Margen Estabilidad [N]": round(f_est_y - f_din[1], 1)
        })

    return pd.DataFrame(resumen)