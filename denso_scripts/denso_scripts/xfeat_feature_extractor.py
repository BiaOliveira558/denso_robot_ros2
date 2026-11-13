#!/usr/bin/env python3
"""
Extrator XFeat + KLT para IBVS (simulação Gazebo OU câmera real RealSense D405),
com banco multi-escala / multi-rotação da goal, TABELA DE TRACKS, ATIVOS ponderados e espalhados,
ERRO DE CONVERGÊNCIA (RMS ponderado, igual ao do controlador), MEDIÇÃO DE TEMPO por etapa
e DIAGNÓSTICO por frame para coleta (/ibvs/xfeat_debug).

Versões (mesmo código, só parâmetros):
  v1: n_ativos=6 grade=0                         z_tol_rel=0    z_salto_max=0
  v2: n_ativos=6 grade=4 grade_objeto=false      z_tol_rel=0.25 z_salto_max=0.30
  v3: n_ativos=6 grade=3 grade_objeto=true       z_tol_rel=0.15 z_salto_max=0.30

  grade        : escolha dos IDs da goal por células de uma grade g x g (0 = só score)
  grade_objeto : True = grade sobre a região do objeto | False = grade sobre a imagem inteira
  z_tol_rel    : na goal, descarta keypoints com Z* longe do Z típico do objeto (0 = desligado)
  z_salto_max  : no run, ignora saltos de Z entre frames maiores que essa fração (0 = desligado)
  mostrar_grade: desenha a grade usada na escolha dos IDs no lado GOAL da janela

O parâmetro 'sim' (vindo do launch) escolhe os defaults:
  sim = True  -> tópicos /basic_camera, goal em ~/xfeat_goal, z 0.05-2.0 m, callbacks separados
  sim = False -> tópicos /camera/camera/..., goal em ~/xfeat_ibvs_logs/goal, z 0.07-0.6 m, sync RGB-D

Publica em /ibvs/xfeat_features (Float64MultiArray), só os ATIVOS (e os que estão ENTRANDO):
  formato_ponderado = True (padrão):
    [N, detectado, stamp_sec, stamp_nanosec,  id1, x1, y1, z1, x1*, y1*, z1*, w1,  ...]
  formato_ponderado = False (controlador antigo):
    [N, detectado, stamp_sec, stamp_nanosec,  x1, y1, z1, x1*, y1*, z1*,  ...]

Publica em /ibvs/xfeat_debug (25 valores + 6 por ID):
  [frame, stamp_sec, stamp_nsec, intervalo_ms, idade_ms, t_total_ms, t_klt_ms, t_xfeat_ms,
   xfeat_rodou, n_matches, n_inliers, n_repostos, n_rastreados, n_elegiveis, n_ativos,
   soma_w, dist_min_px, rms, perdas_ativo, entradas,
   n_keypoints_xfeat, n_rastreados_antes, n_mantidos, n_perdidos_klt, ativos_mudaram]
  + por ID: [status, idade, u, v, w, e_px]

Teclas na janela: g = definir goal | q = sair
"""
import os
import sys
import csv
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.clock import Clock, ClockType
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer

DEFAULTS = {
    True: {   # simulação
        'rgb_topic': '/basic_camera',
        'depth_topic': '/basic_camera/depth/image_raw',
        'info_topic': '/basic_camera/camera_info',
        'goal_dir': os.path.expanduser('~/xfeat_goal'),
        'z_min': 0.05,
        'z_max': 2.0,
        'janela': 'XFeat + KLT feature extractor (sim)',
    },
    False: {  # robô real (RealSense D405)
        'rgb_topic': '/camera/camera/color/image_raw',
        'depth_topic': '/camera/camera/aligned_depth_to_color/image_raw',
        'info_topic': '/camera/camera/color/camera_info',
        'goal_dir': os.path.expanduser('~/xfeat_ibvs_logs/goal'),
        'z_min': 0.07,
        'z_max': 0.6,
        'janela': 'XFeat + KLT feature extractor (real)',
    },
}

# Status de cada ID na tabela de tracks
DESCARTADO, PERDIDO, RESERVA, ENTRANDO, ATIVO, SAINDO = -1, 0, 1, 2, 3, 4
NOME_STATUS = {DESCARTADO: 'descartado', PERDIDO: 'perdido', RESERVA: 'reserva',
               ENTRANDO: 'entrando', ATIVO: 'ativo', SAINDO: 'saindo'}
CORES = {DESCARTADO: (0, 0, 255), PERDIDO: (70, 70, 70), RESERVA: (170, 170, 170),
         ENTRANDO: (255, 0, 255), ATIVO: (0, 255, 0), SAINDO: (0, 165, 255)}
ORDEM_TEMPOS = ['intervalo', 'idade_frame', 'callback', 'copia', 'klt', 'xfeat', 'z',
                'ativos', 'tabela', 'erro_pub', 'log', 'janela', 'total']


def top_n_pontos(pts, scores, n, raio):
    """Índices dos n pontos de maior score. Se raio > 0, ignora pontos a menos de
    'raio' px de um já escolhido (evita IDs amontoados)."""
    ordem = np.argsort(-scores)
    if raio <= 0:
        return ordem[:n]
    escolhidos = []
    for i in ordem:
        if escolhidos and np.min(np.linalg.norm(pts[escolhidos] - pts[i], axis=1)) < raio:
            continue
        escolhidos.append(i)
        if len(escolhidos) == n:
            break
    return np.array(escolhidos, dtype=int)


def top_n_grade(pts, scores, n, raio, shape, g, sobre_objeto=True):
    """n pontos espalhados por uma grade g x g.
    sobre_objeto=True : grade sobre o retângulo ocupado pelos candidatos (o objeto)
    sobre_objeto=False: grade sobre a imagem inteira (versão v2)
    Melhor score de cada célula primeiro, respeitando a distância mínima 'raio'."""
    if sobre_objeto:
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
    else:
        x0, y0 = 0.0, 0.0
        x1, y1 = float(shape[1]), float(shape[0])
    w = max(float(x1 - x0), 1e-6)
    h = max(float(y1 - y0), 1e-6)
    cx = np.clip(((pts[:, 0] - x0) / w * g).astype(int), 0, g - 1)
    cy = np.clip(((pts[:, 1] - y0) / h * g).astype(int), 0, g - 1)
    cel = cy * g + cx
    filas = {}
    for i in np.argsort(-scores):
        filas.setdefault(int(cel[i]), []).append(i)
    escolhidos = []
    while len(escolhidos) < n and any(filas.values()):
        for c in list(filas):
            while filas[c]:
                i = filas[c].pop(0)
                if not escolhidos or raio <= 0 or \
                        np.min(np.linalg.norm(pts[escolhidos] - pts[i], axis=1)) >= raio:
                    escolhidos.append(i)
                    break
            if len(escolhidos) == n:
                break
    return np.array(escolhidos, dtype=int)


class XFeatKLTExtractor(Node):
    def __init__(self):
        super().__init__('xfeat_feature_extractor')

        # ---- Modo (precisa ser lido antes dos outros parâmetros) ----
        self.declare_parameter('sim', False)
        self.sim = bool(self.get_parameter('sim').value)
        d = DEFAULTS[self.sim]
        self.janela = d['janela']

        self.declare_parameter('rgb_topic', d['rgb_topic'])
        self.declare_parameter('depth_topic', d['depth_topic'])
        self.declare_parameter('info_topic', d['info_topic'])
        self.declare_parameter('features_topic', '/ibvs/xfeat_features')
        self.declare_parameter('debug_topic', '/ibvs/xfeat_debug')
        self.declare_parameter('xfeat_path', os.path.expanduser('~/accelerated_features'))
        self.declare_parameter('goal_dir', d['goal_dir'])
        self.declare_parameter('top_k', 2048)          # keypoints do XFeat no frame atual e na goal original
        self.declare_parameter('n_pontos', 20)         # IDs fixos escolhidos na goal
        self.declare_parameter('raio_nms', 20.0)       # px entre IDs na goal (0 = só score)
        # ---- Correções (espalhamento e profundidade) — configuração final (v2) ----
        self.declare_parameter('grade', 4)              # grade 4x4 na escolha dos IDs
        self.declare_parameter('grade_objeto', True)   # grade sobre a IMAGEM inteira
        self.declare_parameter('z_tol_rel', 0.25)       # descarta na goal Z* > 25% longe do Z típico
        self.declare_parameter('z_salto_max', 0.30)     # no run, rejeita saltos de Z > 30% entre frames
        # ---- Banco multi-escala / multi-rotação ----
        self.declare_parameter('escalas', [1.0, 0.7, 0.5, 0.35, 0.25])
        self.declare_parameter('angulos', [0.0, -25.0, 25.0])   # graus
        self.declare_parameter('top_k_nivel', 1024)    # keypoints detectados por versão (reforço do RANSAC)
        self.declare_parameter('min_lado', 64)         # px: ignora versões menores que isso
        self.declare_parameter('dup_px', 3.0)          # px: remove keypoint nativo colado num ID
        # ---- XFeat (reposição) ----
        self.declare_parameter('min_pontos', 6)        # abaixo disso: XFeat em todo frame
        self.declare_parameter('min_rastreados', 10)   # abaixo disso: XFeat a cada redetect_every frames
        self.declare_parameter('redetect_every', 10)
        # ---- Ativos ----
        self.declare_parameter('n_ativos', 6)          # pontos que controlam o robô
        self.declare_parameter('idade_min', 10)        # frames rastreado para ser elegível
        self.declare_parameter('ramp_frames', 15)      # frames para o peso do substituto ir de 0 a 1
        self.declare_parameter('min_soma_w', 4.0)      # abaixo disso publica detectado = 0
        self.declare_parameter('peso_score', 20.0)     # px por unidade de score no desempate do espalhamento
        self.declare_parameter('formato_ponderado', True)   # True: publica id e w por ponto
        # ---- Convergência (mesmos valores do controlador) ----
        self.declare_parameter('convergence_threshold', 0.035)
        self.declare_parameter('convergence_frames', 5)
        # ---------------------------------------------
        self.declare_parameter('rate', 30.0)
        self.declare_parameter('max_dim', 640)
        self.declare_parameter('min_cossim', 0.82)
        self.declare_parameter('ransac_px', 2.0)
        self.declare_parameter('klt_win', 21)
        self.declare_parameter('klt_niveis', 3)
        self.declare_parameter('klt_fb_px', 1.0)       # checagem ida-e-volta: acima disso o ID é perdido
        self.declare_parameter('depth_scale', 0.001)
        self.declare_parameter('sync_slop', 0.03)
        self.declare_parameter('z_min', d['z_min'])
        self.declare_parameter('z_max', d['z_max'])
        self.declare_parameter('mostrar_janela', True)
        self.declare_parameter('mostrar_grade', True)   # desenha a grade da escolha dos IDs no lado GOAL
        self.declare_parameter('log_terminal', True)
        self.declare_parameter('log_tabela_every', 30)   # imprime a tabela a cada N frames (0 = nunca)
        self.declare_parameter('log_tempo_every', 30)   # imprime o resumo de tempos a cada N frames (0 = nunca)
        gp = lambda n: self.get_parameter(n).value

        self.top_k = gp('top_k')
        self.n_pontos = gp('n_pontos')
        self.raio_nms = float(gp('raio_nms'))
        self.grade = int(gp('grade'))
        self.grade_objeto = bool(gp('grade_objeto'))
        self.z_tol_rel = float(gp('z_tol_rel'))
        self.z_salto_max = float(gp('z_salto_max'))
        self.escalas = [float(s) for s in gp('escalas')]
        self.angulos = [float(a) for a in gp('angulos')]
        self.top_k_nivel = int(gp('top_k_nivel'))
        self.min_lado = int(gp('min_lado'))
        self.dup_px = float(gp('dup_px'))
        self.min_pontos = max(gp('min_pontos'), 4)
        self.min_rastreados = int(gp('min_rastreados'))
        self.redetect_every = max(gp('redetect_every'), 1)
        self.n_ativos = int(gp('n_ativos'))
        self.idade_min = int(gp('idade_min'))
        self.ramp_frames = max(int(gp('ramp_frames')), 1)
        self.min_soma_w = float(gp('min_soma_w'))
        self.peso_score = float(gp('peso_score'))
        self.formato_ponderado = bool(gp('formato_ponderado'))
        self.conv_thr = float(gp('convergence_threshold'))
        self.conv_frames = int(gp('convergence_frames'))
        self.max_dim = gp('max_dim')
        self.min_cossim = gp('min_cossim')
        self.ransac_px = gp('ransac_px')
        w = gp('klt_win')
        self.klt_params = dict(winSize=(w, w), maxLevel=gp('klt_niveis'),
                               criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self.klt_fb_px = gp('klt_fb_px')
        self.depth_scale = gp('depth_scale')
        self.z_min, self.z_max = float(gp('z_min')), float(gp('z_max'))
        self.goal_dir = gp('goal_dir')
        self.mostrar = gp('mostrar_janela')
        self.mostrar_grade = bool(gp('mostrar_grade'))
        self.log_terminal = gp('log_terminal')
        self.log_tabela_every = int(gp('log_tabela_every'))
        self.log_tempo_every = int(gp('log_tempo_every'))
        os.makedirs(self.goal_dir, exist_ok=True)

        # ---- XFeat ----
        sys.path.insert(0, gp('xfeat_path'))
        from modules.xfeat import XFeat
        torch.set_grad_enabled(False)
        self.xfeat = XFeat()

        # ---- Estado ----
        self.bridge = CvBridge()
        self.frame = None
        self.frame_stamp = None
        self.depth = None
        self.novo = False
        self.K = None
        self.goal = None
        self.banco = None
        self.versoes = []
        self.encoding_logado = False
        self.n_frame = 0
        self.ultima_versao = '-'
        self._dbg = None
        self._ult_n_kp = 0
        # ---- Medição de tempo ----
        self.tempos = {}                   # nome -> lista de ms
        self.t_ultimo_frame = None         # perf_counter do último frame processado
        self.n_rx = 0                      # frames recebidos da câmera desde o último resumo
        self.t_rx = time.perf_counter()
        self.ultimo_resumo = -1
        self.resetar_rastreio()

        # ---- Goal salva em disco ----
        rgb_path = os.path.join(self.goal_dir, 'rgb.png')
        depth_path = os.path.join(self.goal_dir, 'depth.npy')
        if os.path.exists(rgb_path):
            depth = np.load(depth_path) if os.path.exists(depth_path) else None
            self.definir_goal(cv2.imread(rgb_path), depth, salvar=False)
        else:
            self.get_logger().warn(f'Sem goal salva em {self.goal_dir}: posicione o robo e aperte g na janela')

        # ---- ROS ----
        if self.sim:
            self.create_subscription(Image, gp('rgb_topic'), self.rgb_cb, qos_profile_sensor_data)
            self.create_subscription(Image, gp('depth_topic'), self.depth_cb, qos_profile_sensor_data)
        else:
            self.sub_rgb = Subscriber(self, Image, gp('rgb_topic'), qos_profile=qos_profile_sensor_data)
            self.sub_depth = Subscriber(self, Image, gp('depth_topic'), qos_profile=qos_profile_sensor_data)
            self.sync = ApproximateTimeSynchronizer(
                [self.sub_rgb, self.sub_depth], queue_size=10, slop=gp('sync_slop'))
            self.sync.registerCallback(self.rgbd_cb)
        self.create_subscription(CameraInfo, gp('info_topic'), self.info_cb, 10)
        self.pub = self.create_publisher(Float64MultiArray, gp('features_topic'), 10)
        self.pub_dbg = self.create_publisher(Float64MultiArray, gp('debug_topic'), 50)
        self.create_timer(1.0 / gp('rate'), self.processar,
                          clock=Clock(clock_type=ClockType.STEADY_TIME))

        if self.mostrar:
            cv2.namedWindow(self.janela, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow(self.janela, 1600, 640)

        self.get_logger().info(
            f'XFeat+KLT iniciado [{"SIM" if self.sim else "REAL"}] ({self.xfeat.dev}) | '
            f'rgb={gp("rgb_topic")} | goal_dir={self.goal_dir} | n_pontos={self.n_pontos} | '
            f'grade={self.grade} ({"objeto" if self.grade_objeto else "imagem"}) | '
            f'z_tol_rel={self.z_tol_rel} | z_salto_max={self.z_salto_max} | '
            f'n_ativos={self.n_ativos} | idade_min={self.idade_min} | ramp={self.ramp_frames} | '
            f'peso_score={self.peso_score} | formato_ponderado={self.formato_ponderado} | '
            f'limiar={self.conv_thr} ({self.conv_frames} frames) | rate={gp("rate")} Hz | '
            f'teclas: g = goal, q = sair')

    # ================================================================== callbacks
    def converter_depth(self, msg):
        d = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough').astype(np.float32)
        if msg.encoding == '16UC1':
            d *= self.depth_scale
        d[~np.isfinite(d)] = 0.0
        return d

    def rgbd_cb(self, rgb_msg, depth_msg):
        t0 = time.perf_counter()
        self.frame = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
        self.frame_stamp = rgb_msg.header.stamp
        self.depth = self.converter_depth(depth_msg)
        self.novo = True
        self.n_rx += 1
        self.marcar('callback', t0)
        if not self.encoding_logado:
            self.encoding_logado = True
            self.get_logger().info(
                f'RGB {rgb_msg.width}x{rgb_msg.height} ({rgb_msg.encoding}) | '
                f'depth {depth_msg.width}x{depth_msg.height} ({depth_msg.encoding})')
            if (rgb_msg.width, rgb_msg.height) != (depth_msg.width, depth_msg.height):
                self.get_logger().warn(
                    'RGB e depth com resolucoes diferentes: confira align_depth.enable:=true')

    def rgb_cb(self, msg):
        t0 = time.perf_counter()
        self.frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.frame_stamp = msg.header.stamp
        self.novo = True
        self.n_rx += 1
        self.marcar('callback', t0)

    def depth_cb(self, msg):
        self.depth = self.converter_depth(msg)
        if not self.encoding_logado:
            self.encoding_logado = True
            self.get_logger().info(f'depth {msg.width}x{msg.height} ({msg.encoding})')
            if self.frame is not None and self.frame.shape[:2] != self.depth.shape[:2]:
                self.get_logger().warn('RGB e depth com resolucoes diferentes: Z sera reescalado')

    def info_cb(self, msg):
        if self.K is None:
            self.get_logger().info('camera_info recebido')
        self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)

    # ================================================================== XFeat
    def extrair(self, img):
        """Keypoints (pixels da imagem original), descritores e score de detecção."""
        h, w = img.shape[:2]
        escala = 1.0
        img_in = img
        if self.max_dim > 0 and max(h, w) > self.max_dim:
            escala = self.max_dim / max(h, w)
            img_in = cv2.resize(img, (int(w * escala), int(h * escala)), interpolation=cv2.INTER_AREA)
        out = self.xfeat.detectAndCompute(img_in, top_k=self.top_k)[0]
        kp = out['keypoints'].cpu().numpy() / escala
        scores = out['scores'].cpu().numpy()
        return kp, out['descriptors'], scores

    def descrever_em(self, img, pts):
        """Descritores XFeat lidos EXATAMENTE nas posições pts (px de img), sem detector."""
        if len(pts) == 0:
            return torch.zeros((0, 64), device=self.xfeat.dev)
        x, rh, rw = self.xfeat.preprocess_tensor(img)
        M1, _, _ = self.xfeat.net(x)
        M1 = F.normalize(M1, dim=1)
        _, _, H, W = x.shape
        p = torch.tensor(pts / np.array([rw, rh]), dtype=torch.float32, device=x.device)[None]
        d = self.xfeat.interpolator(M1, p, H=H, W=W)[0]
        return F.normalize(d, dim=-1)

    def construir_banco(self, img, pts_id):
        """Banco de descritores da goal em várias escalas e ângulos.
        Cada linha: posição na goal ORIGINAL, ID (-1 = não é ID), descritor e versão."""
        H, W = img.shape[:2]
        base = self.max_dim / max(H, W) if (self.max_dim > 0 and max(H, W) > self.max_dim) else 1.0
        pos_l, id_l, desc_l, ver_l = [], [], [], []
        self.versoes = []
        margem = 6.0

        for s in self.escalas:
            fator = base * s
            w_s, h_s = int(round(W * fator)), int(round(H * fator))
            if min(w_s, h_s) < self.min_lado:
                continue
            img_s = cv2.resize(img, (w_s, h_s), interpolation=cv2.INTER_AREA)

            for a in self.angulos:
                R = cv2.getRotationMatrix2D((w_s / 2.0, h_s / 2.0), a, 1.0)
                img_v = img_s if a == 0.0 else cv2.warpAffine(
                    img_s, R, (w_s, h_s), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                A = np.vstack([R, [0.0, 0.0, 1.0]])         # reduzida -> versão
                Ainv = np.linalg.inv(A)                     # versão -> reduzida
                v = len(self.versoes)
                self.versoes.append((s, a))

                # (a) IDs: descritor lido na posição exata de cada ID nesta versão
                pid_v = (A[:2, :2] @ (pts_id * fator).T).T + A[:2, 2]
                dentro = ((pid_v[:, 0] >= margem) & (pid_v[:, 0] < w_s - margem) &
                          (pid_v[:, 1] >= margem) & (pid_v[:, 1] < h_s - margem))
                ids_ok = np.where(dentro)[0]
                if len(ids_ok):
                    pos_l.append(pts_id[ids_ok])
                    id_l.append(ids_ok)
                    desc_l.append(self.descrever_em(img_v, pid_v[ids_ok]))
                    ver_l.append(np.full(len(ids_ok), v))

                # (b) keypoints detectados nesta versão (reforçam o RANSAC; não são IDs)
                out = self.xfeat.detectAndCompute(img_v, top_k=self.top_k_nivel)[0]
                kp_v = out['keypoints'].cpu().numpy()
                if len(kp_v) == 0:
                    continue
                kp_s = (Ainv[:2, :2] @ kp_v.T).T + Ainv[:2, 2]       # coordenadas na reduzida
                ok = ((kp_s[:, 0] >= margem) & (kp_s[:, 0] < w_s - margem) &
                      (kp_s[:, 1] >= margem) & (kp_s[:, 1] < h_s - margem))   # fora das bordas pretas da rotação
                if len(ids_ok):
                    dmin = np.min(np.linalg.norm(kp_v[:, None, :] - pid_v[ids_ok][None, :, :], axis=2), axis=1)
                    ok &= dmin > self.dup_px                          # não competir com o próprio ID
                idx = np.where(ok)[0]
                if len(idx):
                    pos_l.append((kp_s[idx] / fator).astype(np.float32))
                    id_l.append(np.full(len(idx), -1))
                    desc_l.append(out['descriptors'][torch.as_tensor(idx, device=out['descriptors'].device)])
                    ver_l.append(np.full(len(idx), v))

        if not desc_l:
            return None
        return {'pos': np.vstack(pos_l).astype(np.float32),
                'id': np.concatenate(id_l).astype(int),
                'desc': torch.cat([dd.to(self.xfeat.dev) for dd in desc_l], 0),
                'ver': np.concatenate(ver_l).astype(int)}

    def definir_goal(self, img, depth, salvar=True):
        t0 = time.time()
        kp, desc, scores = self.extrair(img)

        # 1) filtro de profundidade dos candidatos (remove pontos no fundo / borda)
        if depth is not None:
            z_kp = self.amostrar_z(depth, kp, img.shape)
            ok = z_kp > 0
            n_sem_z = int((~ok).sum())
            z_ref = float('nan')
            if self.z_tol_rel > 0 and ok.any():
                top = np.argsort(-scores)[:50]
                top = top[ok[top]]
                if len(top):
                    z_ref = float(np.median(z_kp[top]))       # Z típico do objeto (pontos mais fortes)
                    ok &= np.abs(z_kp - z_ref) <= self.z_tol_rel * z_ref
        else:
            z_kp = np.zeros(len(kp))
            ok = np.ones(len(kp), dtype=bool)
            n_sem_z, z_ref = len(kp), float('nan')
            self.get_logger().warn('Goal SEM profundidade: filtro de Z desligado')
        cand = np.where(ok)[0]

        # 2) escolha dos IDs entre os candidatos (grade sobre objeto/imagem, ou só score)
        grade_bbox = None
        if self.grade > 0 and len(cand) > 0:
            s_loc = top_n_grade(kp[cand], scores[cand], self.n_pontos, self.raio_nms, img.shape,
                                self.grade, self.grade_objeto)
            # mesmos limites usados dentro de top_n_grade (para desenhar a grade na janela)
            if self.grade_objeto:
                x0, y0 = kp[cand].min(axis=0)
                x1, y1 = kp[cand].max(axis=0)
            else:
                x0, y0 = 0.0, 0.0
                x1, y1 = float(img.shape[1]), float(img.shape[0])
            grade_bbox = (float(x0), float(y0), float(x1), float(y1))
        else:
            s_loc = top_n_pontos(kp[cand], scores[cand], self.n_pontos, self.raio_nms)
        sel = cand[s_loc]
        pts_id = kp[sel].astype(np.float32)
        zs_id = z_kp[sel]

        self.goal = {'img': img.copy(), 'pts_id': pts_id,
                     'score': scores[sel].astype(np.float64),
                     'zs': zs_id,
                     'grade_bbox': grade_bbox,
                     'depth': None if depth is None else depth.copy()}
        self.n_ids = len(sel)
        self.banco = self.construir_banco(img, pts_id)
        self.resetar_rastreio()      # monta a tabela de tracks (todos perdidos; sem Z* = descartado)

        os.makedirs(self.goal_dir, exist_ok=True)
        if salvar:
            cv2.imwrite(os.path.join(self.goal_dir, 'rgb.png'), img)
            if depth is not None:
                np.save(os.path.join(self.goal_dir, 'depth.npy'), depth)
            self.get_logger().info(f'Goal salva em {self.goal_dir}')
        # goal_ids.csv sempre reflete os IDs em uso (muda com grade / filtro de Z)
        with open(os.path.join(self.goal_dir, 'goal_ids.csv'), 'w', newline='') as f:
            wr = csv.writer(f)
            wr.writerow(['id', 'u', 'v', 'score', 'z_star'])
            for i in range(self.n_ids):
                wr.writerow([i, f'{pts_id[i, 0]:.3f}', f'{pts_id[i, 1]:.3f}',
                             f'{self.goal["score"][i]:.4f}', f'{zs_id[i]:.4f}'])

        n_banco = 0 if self.banco is None else len(self.banco['id'])
        n_id_rows = 0 if self.banco is None else int((self.banco['id'] >= 0).sum())
        onde = 'objeto' if self.grade_objeto else 'imagem'
        self.get_logger().info(
            f'Goal definida: IDs fixos {self.n_ids} | grade {self.grade} (sobre {onde}) | '
            f'filtro Z: ref {z_ref:.3f} m, tol {self.z_tol_rel:.0%}, candidatos {len(cand)}/{len(kp)} '
            f'(sem Z {n_sem_z}) | versoes {len(self.versoes)} | banco {n_banco} ({n_id_rows} de IDs) | '
            f'{time.time() - t0:.1f} s')
        sem_z = np.where(zs_id <= 0)[0]
        if len(sem_z):
            self.get_logger().warn(f'IDs sem Z* valido (descartados): {sem_z.tolist()}')
        self.imprimir_tabela()

    def casar(self, desc_cur):
        """Vizinho mais próximo mútuo entre banco e frame atual, com min_cossim.
        Devolve (linhas do banco, índices no frame, similaridade)."""
        B = self.banco['desc']
        if B.shape[0] == 0 or desc_cur.shape[0] == 0:
            return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
        sim = B @ desc_cur.T                         # (M, N)
        melhor_c = sim.argmax(dim=1)                 # para cada linha do banco
        melhor_b = sim.argmax(dim=0)                 # para cada ponto do frame
        ib = torch.arange(B.shape[0], device=B.device)
        mutuo = melhor_b[melhor_c] == ib
        val = sim[ib, melhor_c]
        ok = mutuo & (val > self.min_cossim)
        return ib[ok].cpu().numpy(), melhor_c[ok].cpu().numpy(), val[ok].cpu().numpy()

    def buscar_xfeat(self, img):
        """Frame atual x banco + RANSAC (posições na goal original).
        Devolve {id: posição atual} dos IDs inliers, n_matches, n_inliers."""
        achados = {}
        if self.banco is None:
            return achados, 0, 0
        kp, desc, _ = self.extrair(img)
        self._ult_n_kp = len(kp)
        ib, ic, sv = self.casar(desc)
        if len(ib) < 8:
            return achados, len(ib), 0
        pg = self.banco['pos'][ib]
        pc = kp[ic].astype(np.float32)
        _, mask = cv2.findFundamentalMat(pg, pc, cv2.FM_RANSAC, self.ransac_px, 0.999)
        if mask is None:
            return achados, len(ib), 0
        inl = mask.ravel().astype(bool)

        # Versão (escala, ângulo) que mais contribuiu: indica a "distância aparente" atual
        vers = self.banco['ver'][ib[inl]]
        if len(vers):
            s, a = self.versoes[int(np.bincount(vers).argmax())]
            self.ultima_versao = f'escala {s:g} ang {a:+.0f}'

        melhor = {}
        for b, c, sim_v in zip(ib[inl], ic[inl], sv[inl]):
            id_ = int(self.banco['id'][b])
            if id_ >= 0 and (id_ not in melhor or sim_v > melhor[id_][1]):
                melhor[id_] = (kp[c], sim_v)
        achados = {id_: p for id_, (p, _) in melhor.items()}
        return achados, len(ib), int(inl.sum())

    # ================================================================== KLT + tabela de tracks
    def resetar_rastreio(self):
        n = getattr(self, 'n_ids', self.n_pontos)
        self.trk_pts = np.zeros((n, 2), dtype=np.float32)   # ponto atual (s)
        self.trk_ok = np.zeros(n, dtype=bool)               # rastreado pelo KLT
        # ---- tabela de tracks ----
        self.trk_status = np.full(n, PERDIDO, dtype=int)
        self.trk_idade = np.zeros(n, dtype=int)             # frames desde a (re)aquisição
        self.trk_z = np.zeros(n)                            # último Z válido no frame atual (0 = nunca)
        self.trk_zrej = np.zeros(n, dtype=int)              # frames seguidos com salto de Z rejeitado
        self.trk_w = np.zeros(n)                            # peso no controle
        if self.goal is not None:
            self.trk_status[self.goal['zs'] <= 0] = DESCARTADO
        self.iniciado = False                               # True depois da escolha dos ativos iniciais
        self.prev_gray = None
        self.frames_sem_xfeat = 0
        # ---- convergência ----
        self.conv_count = 0
        self.convergiu_logado = False
        # ---- diagnóstico ----
        self._ativos_ant = None

    def rastrear_klt(self, gray):
        idx = np.where(self.trk_ok)[0]
        if self.prev_gray is None or len(idx) == 0:
            return
        p0 = self.trk_pts[idx].reshape(-1, 1, 2)
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p0, None, **self.klt_params)
        p0r, st2, _ = cv2.calcOpticalFlowPyrLK(gray, self.prev_gray, p1, None, **self.klt_params)
        fb = np.linalg.norm(p0 - p0r, axis=2).ravel()
        h, w = gray.shape
        p1 = p1.reshape(-1, 2)
        dentro = (p1[:, 0] >= 0) & (p1[:, 0] < w) & (p1[:, 1] >= 0) & (p1[:, 1] < h)
        ok = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb < self.klt_fb_px) & dentro
        if self._dbg is not None:
            self._dbg['n_antes'] = int(len(idx))
            self._dbg['mantidos'] = int(ok.sum())
            self._dbg['perdidos_klt'] = int((~ok).sum())

        # quem continua: posição e idade
        seg = idx[ok]
        self.trk_pts[seg] = p1[ok]
        self.trk_idade[seg] += 1

        # quem foi perdido (avisa se era ativo ou estava entrando)
        perd = idx[~ok]
        for i in perd:
            if self.trk_status[i] in (ATIVO, ENTRANDO):
                self.get_logger().warn(
                    f'>>> frame {self.n_frame} | ID {i} PERDIDO pelo KLT '
                    f'(era {NOME_STATUS[int(self.trk_status[i])]}, w={self.trk_w[i]:.2f})')
                if self._dbg is not None:
                    self._dbg['perdas_ativo'] += 1
        self.trk_ok[perd] = False
        self.trk_status[perd] = PERDIDO
        self.trk_idade[perd] = 0
        self.trk_z[perd] = 0.0
        self.trk_zrej[perd] = 0
        self.trk_w[perd] = 0.0

    # ================================================================== ativos
    def escolher_espalhados(self, cand, fixos, n):
        """Escolhe até n IDs de cand, um por vez: o que fica mais longe (px, frame atual)
        dos já escolhidos (fixos + escolhidos). O score desempata."""
        score = self.goal['score']
        cand = list(cand)
        escolhidos = []
        while cand and len(escolhidos) < n:
            ref = list(fixos) + escolhidos
            if not ref:
                melhor = max(cand, key=lambda c: score[c])
            else:
                P = self.trk_pts[ref]
                melhor = max(cand, key=lambda c: np.min(np.linalg.norm(P - self.trk_pts[c], axis=1))
                             + self.peso_score * score[c])
            escolhidos.append(melhor)
            cand.remove(melhor)
        return escolhidos

    def dist_min(self, ids):
        """Menor distância (px) entre dois pontos do conjunto, no frame atual."""
        if len(ids) < 2:
            return 0.0
        P = self.trk_pts[ids]
        D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
        D[np.eye(len(ids), dtype=bool)] = np.inf
        return float(D.min())

    def gerenciar_ativos(self):
        """Escolhe os ativos iniciais (espalhados) e repõe ativos perdidos com rampa de peso."""
        score = self.goal['score']
        elegivel = ((self.trk_status == RESERVA) & (self.trk_idade >= self.idade_min) &
                    (self.trk_z > 0) & (self.goal['zs'] > 0))
        cand = np.where(elegivel)[0]

        # 1. Escolha inicial: espera ter n_ativos elegíveis; entram direto com w = 1
        if not self.iniciado:
            if len(cand) >= self.n_ativos:
                ini = np.sort(np.array(self.escolher_espalhados(cand, [], self.n_ativos), dtype=int))
                self.trk_status[ini] = ATIVO
                self.trk_w[ini] = 1.0
                self.iniciado = True
                self.get_logger().warn(
                    f'>>> frame {self.n_frame} | ATIVOS INICIAIS (w=1): {ini.tolist()} | '
                    f'scores {[round(float(score[i]), 3) for i in ini]} | '
                    f'Z* {[round(float(self.goal["zs"][i]), 3) for i in ini]} | '
                    f'dist min entre eles {self.dist_min(ini):.0f} px')
            else:
                self.get_logger().info(
                    f'aguardando ativos: {len(cand)}/{self.n_ativos} elegiveis', throttle_duration_sec=2.0)
            return

        # 2. Rampa de quem está entrando
        ent = np.where(self.trk_status == ENTRANDO)[0]
        if len(ent):
            self.trk_w[ent] = np.minimum(1.0, self.trk_w[ent] + 1.0 / self.ramp_frames)
            for i in ent[self.trk_w[ent] >= 1.0]:
                self.trk_status[i] = ATIVO
                self.get_logger().warn(f'>>> frame {self.n_frame} | ID {i} agora ATIVO (w=1)')

        # 3. Repõe ativos perdidos com o elegível mais afastado dos que ficaram
        em_campo = np.where(np.isin(self.trk_status, [ATIVO, ENTRANDO]))[0]
        faltam = self.n_ativos - len(em_campo)
        if faltam > 0 and len(cand):
            for i in self.escolher_espalhados(cand, em_campo, faltam):
                self.trk_status[i] = ENTRANDO
                self.trk_w[i] = 0.0
                if self._dbg is not None:
                    self._dbg['entradas'] += 1
                d = np.min(np.linalg.norm(self.trk_pts[em_campo] - self.trk_pts[i], axis=1)) \
                    if len(em_campo) else 0.0
                self.get_logger().warn(
                    f'>>> frame {self.n_frame} | ID {i} ENTRANDO como substituto '
                    f'(score {score[i]:.3f}, idade {self.trk_idade[i]}, a {d:.0f} px do ativo mais proximo) | '
                    f'w sobe 0 -> 1 em {self.ramp_frames} frames')
        if faltam > len(cand):
            self.get_logger().warn(
                f'frame {self.n_frame} | faltam {faltam - len(cand)} ativo(s) e nao ha reserva elegivel',
                throttle_duration_sec=2.0)

    # ================================================================== auxiliares
    def amostrar_z(self, depth, pts, img_shape, r=2):
        z = np.zeros(len(pts))
        if depth is None:
            return z
        sy = depth.shape[0] / img_shape[0]
        sx = depth.shape[1] / img_shape[1]
        h, w = depth.shape[:2]
        for i, (x, y) in enumerate(pts):
            u, v = int(round(x * sx)), int(round(y * sy))
            jan = depth[max(v - r, 0):min(v + r + 1, h), max(u - r, 0):min(u + r + 1, w)]
            val = jan[(jan > self.z_min) & (jan < self.z_max)]
            if val.size > 0:
                z[i] = np.median(val)
        return z

    def imprimir_tabela(self):
        if self.goal is None:
            return
        cont = {s: int((self.trk_status == s).sum()) for s in NOME_STATUS}
        resumo = ' | '.join(f'{NOME_STATUS[s]} {cont[s]}' for s in NOME_STATUS if cont[s] > 0)
        linhas = [f'TABELA DE TRACKS (frame {self.n_frame}) -> {resumo}',
                  'ID | status     | idade | score | Z*    | Z     | w    | atual (px)']
        for i in range(self.n_ids):
            atual = f'({self.trk_pts[i, 0]:.0f},{self.trk_pts[i, 1]:.0f})' if self.trk_ok[i] else '-'
            z = f'{self.trk_z[i]:.3f}' if self.trk_z[i] > 0 else '  -  '
            linhas.append(
                f'{i:2d} | {NOME_STATUS[int(self.trk_status[i])]:<10} | {self.trk_idade[i]:5d} | '
                f'{self.goal["score"][i]:.3f} | {self.goal["zs"][i]:.3f} | {z} | {self.trk_w[i]:.2f} | {atual}')
        self.get_logger().info('\n' + '\n'.join(linhas))

    def publicar(self, stamp, linhas=None, detectado=False):
        n = 0 if linhas is None else len(linhas)
        dados = [float(n), 1.0 if (detectado and n > 0) else 0.0, float(stamp.sec), float(stamp.nanosec)]
        if n > 0:
            dados += linhas.flatten().tolist()
        msg = Float64MultiArray()
        msg.data = dados
        self.pub.publish(msg)

    def publicar_debug(self, intervalo_ms):
        """Diagnóstico por frame em /ibvs/xfeat_debug (layout no cabeçalho do arquivo)."""
        if self.goal is None or self._dbg is None:
            return
        d = self._dbg
        nan = float('nan')
        tot = self.tempos['total'][-1] if self.tempos.get('total') else nan
        n_eleg = int(((self.trk_status == RESERVA) & (self.trk_idade >= self.idade_min) &
                      (self.trk_z > 0)).sum())
        ativos = tuple(np.where(np.isin(self.trk_status, [ATIVO, ENTRANDO]))[0].tolist())
        mudou = 1.0 if (self._ativos_ant is not None and ativos != self._ativos_ant) else 0.0
        self._ativos_ant = ativos
        dados = [float(self.n_frame), float(d['stamp'].sec), float(d['stamp'].nanosec),
                 float(intervalo_ms), float(d['idade']), float(tot),
                 float(d['t_klt']), float(d['t_xfeat']), float(d['xfeat']),
                 float(d['n_m']), float(d['n_in']), float(d['rep']),
                 float(self.trk_ok.sum()), float(n_eleg), float(len(ativos)),
                 float(d['soma_w']), float(d['dmin']), float(d['rms']),
                 float(d['perdas_ativo']), float(d['entradas']),
                 float(d['n_kp']), float(d['n_antes']), float(d['mantidos']),
                 float(d['perdidos_klt']), mudou]
        for i in range(self.n_ids):
            if self.trk_ok[i]:
                u, v = float(self.trk_pts[i, 0]), float(self.trk_pts[i, 1])
                epx = float(np.linalg.norm(self.trk_pts[i] - self.goal['pts_id'][i]))
            else:
                u = v = epx = nan
            dados += [float(self.trk_status[i]), float(self.trk_idade[i]), u, v,
                      float(self.trk_w[i]), epx]
        msg = Float64MultiArray()
        msg.data = dados
        self.pub_dbg.publish(msg)
        self._dbg = None

    # ---- medição de tempo ----
    def marcar(self, nome, t0):
        """Guarda o tempo (ms) desde t0 com o nome dado e devolve o instante atual."""
        t = time.perf_counter()
        self.tempos.setdefault(nome, []).append((t - t0) * 1000.0)
        return t

    def imprimir_tempos(self):
        agora = time.perf_counter()
        hz_rx = self.n_rx / max(agora - self.t_rx, 1e-6)
        v_int = self.tempos.get('intervalo', [])
        hz_proc = 1000.0 / np.mean(v_int) if v_int else 0.0
        linhas = [f'TEMPOS (ultimos {self.log_tempo_every} frames) em ms -> media | max | n   '
                  f'| camera {hz_rx:.1f} Hz | processamento {hz_proc:.1f} Hz']
        for nome in ORDEM_TEMPOS + [k for k in self.tempos if k not in ORDEM_TEMPOS]:
            v = self.tempos.get(nome)
            if v:
                linhas.append(f'  {nome:<12} {np.mean(v):7.1f} | {np.max(v):7.1f} | {len(v)}')
        self.get_logger().info('\n'.join(linhas))
        self.tempos = {}
        self.n_rx = 0
        self.t_rx = agora

    # ================================================================== laço principal
    def processar(self):
        if self.frame is None or not self.novo:
            self.teclado()
            return
        t_ini = time.perf_counter()
        intervalo = float('nan')
        if self.t_ultimo_frame is not None:
            intervalo = (t_ini - self.t_ultimo_frame) * 1000.0
            self.tempos.setdefault('intervalo', []).append(intervalo)
        self.t_ultimo_frame = t_ini

        self.processar_frame(t_ini)

        self.marcar('total', t_ini)
        self.publicar_debug(intervalo)
        if (self.log_tempo_every > 0 and self.n_frame > 0 and
                self.n_frame % self.log_tempo_every == 0 and self.n_frame != self.ultimo_resumo):
            self.ultimo_resumo = self.n_frame
            self.imprimir_tempos()

    def processar_frame(self, t_ini):
        self.novo = False
        img = self.frame.copy()
        stamp = self.frame_stamp
        depth_atual = None if self.depth is None else self.depth.copy()
        idade = (self.get_clock().now() - Time.from_msg(stamp)).nanoseconds * 1e-6
        self.tempos.setdefault('idade_frame', []).append(idade)
        nan = float('nan')
        self._dbg = {'stamp': stamp, 'idade': idade, 't_klt': nan, 't_xfeat': nan, 'xfeat': 0,
                     'n_m': 0, 'n_in': 0, 'rep': 0, 'perdas_ativo': 0, 'entradas': 0,
                     'rms': nan, 'soma_w': 0.0, 'dmin': nan,
                     'n_kp': 0, 'n_antes': 0, 'mantidos': 0, 'perdidos_klt': 0}
        t = self.marcar('copia', t_ini)

        if self.goal is None or self.K is None:
            self.publicar(stamp)
            self.mostrar_imagem(img, None, 'SEM GOAL (aperte g)' if self.goal is None else 'SEM camera_info', None)
            return

        self.n_frame += 1
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. KLT nos IDs rastreados (atualiza posição e idade; marca perdidos)
        self.rastrear_klt(gray)
        n_ok = int(self.trk_ok.sum())
        t = self.marcar('klt', t)
        self._dbg['t_klt'] = self.tempos['klt'][-1]

        # 2. XFeat só quando faltam IDs rastreados
        modo = 'KLT'
        precisa = (n_ok < self.min_pontos) or \
                  (n_ok < self.min_rastreados and self.frames_sem_xfeat >= self.redetect_every)
        if precisa:
            achados, n_m, n_in = self.buscar_xfeat(img)
            repostos = []
            for id_, p in achados.items():
                if not self.trk_ok[id_] and self.trk_status[id_] != DESCARTADO:
                    self.trk_pts[id_] = p
                    self.trk_ok[id_] = True
                    self.trk_status[id_] = RESERVA
                    self.trk_idade[id_] = 0
                    self.trk_z[id_] = 0.0
                    self.trk_zrej[id_] = 0
                    self.trk_w[id_] = 0.0
                    repostos.append(id_)
            self.frames_sem_xfeat = 0
            n_ok = int(self.trk_ok.sum())
            modo = (f'KLT+XFeat (matches {n_m}, inliers {n_in}, repostos {sorted(repostos)}, '
                    f'{self.ultima_versao})')
            t = self.marcar('xfeat', t)
            self._dbg.update(xfeat=1, n_m=n_m, n_in=n_in, rep=len(repostos), n_kp=self._ult_n_kp,
                             t_xfeat=self.tempos['xfeat'][-1])
        else:
            self.frames_sem_xfeat += 1
        self.prev_gray = gray

        # 3. Z atual dos rastreados (rejeita saltos de profundidade, ex.: borda objeto/fundo)
        rastreados = np.where(self.trk_ok)[0]
        if len(rastreados):
            z_now = self.amostrar_z(depth_atual, self.trk_pts[rastreados], img.shape)
            for j, i in enumerate(rastreados):
                z = z_now[j]
                if z <= 0:
                    continue                                  # sem leitura: mantém o último Z válido
                zant = self.trk_z[i]
                if zant > 0 and self.z_salto_max > 0 and abs(z - zant) > self.z_salto_max * zant:
                    self.trk_zrej[i] += 1
                    if self.trk_zrej[i] < 3:
                        continue                              # salto isolado: mantém o Z anterior
                self.trk_z[i] = z
                self.trk_zrej[i] = 0
        t = self.marcar('z', t)

        # 4. Ativos: escolha inicial (espalhados), rampa e substituição
        self.gerenciar_ativos()
        t = self.marcar('ativos', t)

        # 5. Tabela de tracks no terminal
        if self.log_tabela_every > 0 and self.n_frame % self.log_tabela_every == 0:
            self.imprimir_tabela()
            t = self.marcar('tabela', t)

        # 6. Ainda sem ativos: aguarda (robô deve ficar parado)
        sel = np.where(np.isin(self.trk_status, [ATIVO, ENTRANDO]))[0]
        if not self.iniciado or len(sel) == 0:
            n_eleg = int(((self.trk_status == RESERVA) & (self.trk_idade >= self.idade_min) &
                          (self.trk_z > 0)).sum())
            txt = f'AGUARDANDO ATIVOS (elegiveis {n_eleg}/{self.n_ativos})'
            self.publicar(stamp)
            t = self.marcar('erro_pub', t)
            if self.log_terminal:
                self.get_logger().info(f'frame {self.n_frame} | {modo} | rastreados {n_ok}/{self.n_ids} | {txt}')
            t = self.marcar('log', t)
            self.mostrar_imagem(img, None, f'{modo} | {txt}', None)
            self.marcar('janela', t)
            return

        # 7. Normalização e erro de convergência (RMS ponderado, igual ao do controlador)
        w = self.trk_w[sel]
        soma_w = float(w.sum())
        pa = self.trk_pts[sel].astype(np.float64)
        pg = self.goal['pts_id'][sel].astype(np.float64)   # alvo = posição na goal ORIGINAL
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        x, y = (pa[:, 0] - cx) / fx, (pa[:, 1] - cy) / fy
        xs, ys = (pg[:, 0] - cx) / fx, (pg[:, 1] - cy) / fy
        e2 = np.column_stack([x - xs, y - ys])
        rms = float(np.sqrt(np.sum(w * np.sum(e2 ** 2, axis=1)) / soma_w)) if soma_w > 0 else 0.0
        self._dbg.update(rms=rms, soma_w=soma_w)

        if rms < self.conv_thr and np.all(w > 0.99):
            self.conv_count += 1
        else:
            self.conv_count = 0
        convergiu = self.conv_count >= self.conv_frames
        if convergiu and not self.convergiu_logado:
            self.get_logger().warn(
                f'>>> frame {self.n_frame} | CONVERGIU! rms={rms:.4f} < {self.conv_thr} '
                f'por {self.conv_frames} frames | ativos {sel.tolist()}')
            self.convergiu_logado = True
        elif not convergiu:
            self.convergiu_logado = False
        status = 'CONVERGIU' if convergiu else f'{rms / self.conv_thr:.1f}x acima do limiar'
        erro = (rms, status, convergiu)

        # 8. Publicação
        z = self.trk_z[sel]
        zs = self.goal['zs'][sel]
        if self.formato_ponderado:
            linhas = np.column_stack([sel.astype(np.float64), x, y, z, xs, ys, zs, w])
        else:
            linhas = np.column_stack([x, y, z, xs, ys, zs])
        detectado = soma_w >= self.min_soma_w
        self.publicar(stamp, linhas, detectado)
        t = self.marcar('erro_pub', t)

        dmin = self.dist_min(sel)
        self._dbg['dmin'] = dmin
        if self.log_terminal:
            ativos_txt = ' '.join(f'{i}:{self.trk_w[i]:.2f}' for i in sel)
            self.get_logger().info(
                f'frame {self.n_frame} | {modo} | rastreados {n_ok}/{self.n_ids} | '
                f'ativos [{ativos_txt}] | dist min {dmin:.0f} px | '
                f'soma w {soma_w:.2f}{"" if detectado else " (BAIXA)"} | '
                f'erro rms {rms:.4f} (limiar {self.conv_thr}) | {status}')
        t = self.marcar('log', t)

        info = (f'{modo} | ativos: {len(sel)} | soma w: {soma_w:.2f} | '
                f'dist min: {dmin:.0f} px | rastreados: {n_ok}/{self.n_ids}')
        self.mostrar_imagem(img, sel, info, erro)
        self.marcar('janela', t)

    # ================================================================== visualização
    def mostrar_imagem(self, img, ids, texto, erro):
        if not self.mostrar:
            return
        if self.goal is not None:
            g = self.goal['img'].copy()
            # grade usada na escolha dos IDs (só no lado GOAL)
            if self.mostrar_grade and self.goal.get('grade_bbox') is not None:
                x0, y0, x1, y1 = self.goal['grade_bbox']
                n = self.grade
                cor_g = (255, 200, 0)
                W, H = g.shape[1] - 1, g.shape[0] - 1
                for k in range(n + 1):
                    xk = min(int(round(x0 + (x1 - x0) * k / n)), W)
                    yk = min(int(round(y0 + (y1 - y0) * k / n)), H)
                    cv2.line(g, (xk, min(int(y0), H)), (xk, min(int(y1), H)), cor_g, 1)
                    cv2.line(g, (min(int(x0), W), yk), (min(int(x1), W), yk), cor_g, 1)
            atual = img if img.shape[0] == g.shape[0] else cv2.resize(
                img, (int(img.shape[1] * g.shape[0] / img.shape[0]), g.shape[0]))
            for id_, p in enumerate(self.goal['pts_id']):
                cor = CORES[int(self.trk_status[id_])]
                cv2.circle(g, (int(p[0]), int(p[1])), 5, cor, 2)
                cv2.putText(g, str(id_), (int(p[0]) + 6, int(p[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 1)
            vis = np.hstack([g, atual])
            off = g.shape[1]
            cv2.putText(vis, 'GOAL', (10, g.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(vis, 'ATUAL', (off + 10, g.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            # reservas no frame atual (cinza, pequenas)
            for id_ in np.where(self.trk_status == RESERVA)[0]:
                b = self.trk_pts[id_]
                cv2.circle(vis, (int(b[0]) + off, int(b[1])), 3, CORES[RESERVA], 1)
        else:
            vis = img.copy()
            off = 0

        # ativos (verde) e entrando (magenta), com seta até o alvo
        if ids is not None:
            for id_ in ids:
                a = self.goal['pts_id'][id_]
                b = self.trk_pts[id_]
                pa_ = (int(b[0]) + off, int(b[1]))
                pg_ = (int(a[0]), int(a[1]))
                alvo = (int(a[0]) + off, int(a[1]))
                cor = CORES[int(self.trk_status[id_])]
                cv2.line(vis, pg_, pa_, cor, 1)
                cv2.arrowedLine(vis, pa_, alvo, (0, 255, 255), 1, tipLength=0.2)
                cv2.circle(vis, pa_, 5, cor, 2)
                cv2.putText(vis, f'{id_}', (pa_[0] + 6, pa_[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 1)

        # faixa superior: informações + erro usado no limiar de convergência
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 66), (0, 0, 0), -1)
        cv2.putText(vis, texto, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if erro is not None:
            rms, status, convergiu = erro
            erro_txt = f'ERRO: {rms:.4f}  (limiar {self.conv_thr})' + ('  CONVERGIU' if convergiu else '')
            cor_erro = (0, 255, 0) if convergiu else (0, 255, 255)
        else:
            erro_txt, cor_erro = 'ERRO: -', (0, 255, 255)
        cv2.putText(vis, erro_txt, (8, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.75, cor_erro, 2)

        cv2.imshow(self.janela, vis)
        self.teclado()

    def teclado(self):
        if not self.mostrar:
            return
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            raise KeyboardInterrupt
        if key == ord('g') and self.frame is not None:
            self.definir_goal(self.frame.copy(), None if self.depth is None else self.depth.copy())


def main():
    rclpy.init()
    node = XFeatKLTExtractor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()