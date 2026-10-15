#!/usr/bin/env python3
"""
Extrator de features XFeat para IBVS.

Publica em /ibvs/xfeat_features (Float64MultiArray):
  [N, detectado, stamp_sec, stamp_nanosec,
   x1, y1, z1, x1*, y1*, z1*,  ...  (N blocos)]
  x, y   : coordenadas normalizadas atuais
  z      : profundidade atual (m), 0 = inválida
  x*, y*, z* : o mesmo na goal

Seleção dos pontos (parâmetro 'selecao'):
  goal  : espalhamento feito UMA VEZ nos keypoints da goal (lista de prioridade fixa);
          a cada frame usa os n_pontos inliers de maior prioridade nessa lista.
  atual : espalhamento refeito a cada frame sobre os inliers atuais.

Teclas na janela: g = definir goal | q = sair
"""
import os
import sys
import cv2
import numpy as np
import torch
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge

JANELA = 'XFeat feature extractor'


def ordem_espalhada(pts, n=None):
    """Farthest point sampling a partir do índice 0.
    Devolve os índices de pts em ordem de escolha (n primeiros, ou todos se n=None)."""
    M = len(pts)
    if M == 0:
        return np.array([], dtype=int)
    n = M if n is None else min(n, M)
    escolhidos = [0]
    dist = np.linalg.norm(pts - pts[0], axis=1)
    for _ in range(n - 1):
        k = int(np.argmax(dist))
        escolhidos.append(k)
        dist = np.minimum(dist, np.linalg.norm(pts - pts[k], axis=1))
    return np.array(escolhidos)


class XFeatFeatureExtractor(Node):
    def __init__(self):
        super().__init__('xfeat_feature_extractor')
        self.declare_parameter('rgb_topic', '/basic_camera')
        self.declare_parameter('depth_topic', '/basic_camera/depth/image_raw')
        self.declare_parameter('info_topic', '/basic_camera/camera_info')
        self.declare_parameter('features_topic', '/ibvs/xfeat_features')
        self.declare_parameter('xfeat_path', os.path.expanduser('~/accelerated_features'))
        self.declare_parameter('goal_dir', os.path.expanduser('~/xfeat_goal'))
        self.declare_parameter('top_k', 2048)         # keypoints por frame e no catálogo da goal
        self.declare_parameter('n_pontos', 20)        # máximo de pares publicados
        self.declare_parameter('selecao', 'goal')     # goal (lista fixa) | atual (refeito a cada frame)
        self.declare_parameter('min_pontos', 6)       # abaixo disso: detectado = 0
        self.declare_parameter('rate', 15.0)
        self.declare_parameter('max_dim', 640)        # maior lado da imagem no XFeat (0 = sem reduzir)
        self.declare_parameter('min_cossim', 0.82)
        self.declare_parameter('ransac_px', 2.0)
        self.declare_parameter('z_min', 0.05)
        self.declare_parameter('z_max', 2.0)
        self.declare_parameter('mostrar_janela', True)
        gp = lambda n: self.get_parameter(n).value

        self.top_k = gp('top_k')
        self.n_pontos = gp('n_pontos')
        self.selecao = gp('selecao')
        self.min_pontos = max(gp('min_pontos'), 4)
        self.max_dim = gp('max_dim')
        self.min_cossim = gp('min_cossim')
        self.ransac_px = gp('ransac_px')
        self.z_min, self.z_max = gp('z_min'), gp('z_max')
        self.goal_dir = gp('goal_dir')
        self.mostrar = gp('mostrar_janela')
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

        # ---- Goal salva em disco ----
        rgb_path = os.path.join(self.goal_dir, 'rgb.png')
        depth_path = os.path.join(self.goal_dir, 'depth.npy')
        if os.path.exists(rgb_path):
            depth = np.load(depth_path) if os.path.exists(depth_path) else None
            self.definir_goal(cv2.imread(rgb_path), depth, salvar=False)
        else:
            self.get_logger().warn('Sem goal salva: posicione o robo e aperte g na janela')

        # ---- ROS ----
        self.create_subscription(Image, gp('rgb_topic'), self.rgb_cb, qos_profile_sensor_data)
        self.create_subscription(Image, gp('depth_topic'), self.depth_cb, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, gp('info_topic'), self.info_cb, 10)
        self.pub = self.create_publisher(Float64MultiArray, gp('features_topic'), 10)
        self.create_timer(1.0 / gp('rate'), self.processar)

        if self.mostrar:
            cv2.namedWindow(JANELA, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow(JANELA, 1600, 600)

        self.get_logger().info(
            f'XFeat feature extractor iniciado ({self.xfeat.dev}) | publicando em '
            f'{gp("features_topic")} | n_pontos={self.n_pontos} | selecao={self.selecao} | '
            f'teclas: g = goal, q = sair')

    # ================================================================== callbacks
    def rgb_cb(self, msg):
        self.frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.frame_stamp = msg.header.stamp
        self.novo = True

    def depth_cb(self, msg):
        d = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        d = d.astype(np.float32) / 1000.0 if msg.encoding == '16UC1' else d.astype(np.float32)
        d[~np.isfinite(d)] = 0.0
        self.depth = d

    def info_cb(self, msg):
        if self.K is None:
            self.get_logger().info('camera_info recebido')
        self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)

    # ================================================================== XFeat
    def extrair(self, img):
        """Keypoints (pixels da imagem original) e descritores."""
        h, w = img.shape[:2]
        escala = 1.0
        img_in = img
        if self.max_dim > 0 and max(h, w) > self.max_dim:
            escala = self.max_dim / max(h, w)
            img_in = cv2.resize(img, (int(w * escala), int(h * escala)), interpolation=cv2.INTER_AREA)
        out = self.xfeat.detectAndCompute(img_in, top_k=self.top_k)[0]
        kp = out['keypoints'].cpu().numpy() / escala
        return kp, out['descriptors']

    def definir_goal(self, img, depth, salvar=True):
        kp, desc = self.extrair(img)

        # Lista de prioridade fixa: espalhamento sobre TODOS os keypoints da goal.
        # rank[i] = posição do keypoint i da goal nessa lista (0 = maior prioridade).
        ordem = ordem_espalhada(kp)
        rank = np.empty(len(kp), dtype=int)
        rank[ordem] = np.arange(len(kp))

        self.goal = {'img': img.copy(), 'kp': kp, 'desc': desc, 'rank': rank,
                     'depth': None if depth is None else depth.copy()}
        if salvar:
            cv2.imwrite(os.path.join(self.goal_dir, 'rgb.png'), img)
            if depth is not None:
                np.save(os.path.join(self.goal_dir, 'depth.npy'), depth)
        self.get_logger().info(
            f'Goal definida: {len(kp)} keypoints ({"COM" if depth is not None else "SEM"} profundidade)')

    # ================================================================== auxiliares
    def amostrar_z(self, depth, pts, img_shape, r=2):
        """Z (m) de cada ponto: mediana numa janela 5x5. 0 se inválido."""
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

    def selecionar(self, ig, p_atual, sim, idx_in):
        """Escolhe até n_pontos inliers. Devolve posições na lista de pares."""
        if self.selecao == 'goal':
            # Ordena os inliers pela prioridade do keypoint da goal e pega os primeiros
            ranks = self.goal['rank'][ig[idx_in]]
            return idx_in[np.argsort(ranks)][:self.n_pontos]
        # 'atual': espalhamento refeito sobre os inliers, começando pelo de maior similaridade
        idx_in = idx_in[np.argsort(-sim[idx_in])]
        return idx_in[ordem_espalhada(p_atual[idx_in], self.n_pontos)]

    def publicar(self, stamp, pares=None):
        """pares: array (N, 6) com [x, y, z, x*, y*, z*] ou None."""
        n = 0 if pares is None else len(pares)
        detectado = 1.0 if n >= self.min_pontos else 0.0
        dados = [float(n), detectado, float(stamp.sec), float(stamp.nanosec)]
        if pares is not None and n > 0:
            dados += pares.flatten().tolist()
        msg = Float64MultiArray()
        msg.data = dados
        self.pub.publish(msg)

    # ================================================================== laço principal
    def processar(self):
        if self.frame is None or not self.novo:
            self.teclado()
            return
        self.novo = False
        img = self.frame.copy()
        stamp = self.frame_stamp
        depth_atual = None if self.depth is None else self.depth.copy()

        if self.goal is None or self.K is None:
            self.publicar(stamp)
            self.mostrar_imagem(img, None, None, None,
                                'SEM GOAL (aperte g)' if self.goal is None else 'SEM camera_info')
            return

        # 1. Extração no frame atual
        kp, desc = self.extrair(img)

        # 2. Matching com a goal (MNN + similaridade mínima)
        ig, ic = self.xfeat.match(self.goal['desc'], desc, min_cossim=self.min_cossim)
        sim = (self.goal['desc'][ig] * desc[ic]).sum(-1).cpu().numpy()
        ig, ic = ig.cpu().numpy(), ic.cpu().numpy()
        p_goal = self.goal['kp'][ig]
        p_atual = kp[ic]

        # 3. RANSAC (matriz fundamental)
        inl = np.zeros(len(ig), dtype=bool)
        if len(ig) >= 8:
            F, mask = cv2.findFundamentalMat(p_goal, p_atual, cv2.FM_RANSAC, self.ransac_px, 0.999)
            if mask is not None:
                inl = mask.ravel().astype(bool)
        idx_in = np.where(inl)[0]

        if len(idx_in) < self.min_pontos:
            self.publicar(stamp)
            self.mostrar_imagem(img, None, None, None, f'POUCOS INLIERS ({len(idx_in)})')
            return

        # 4. Seleção dos pontos (lista fixa da goal ou espalhamento no frame atual)
        sel = self.selecionar(ig, p_atual, sim, idx_in)
        pa, pg = p_atual[sel], p_goal[sel]
        ranks = self.goal['rank'][ig[sel]]          # ID fixo de cada ponto (para a janela)

        # 5. Normalização e profundidade
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        x, y = (pa[:, 0] - cx) / fx, (pa[:, 1] - cy) / fy
        xs, ys = (pg[:, 0] - cx) / fx, (pg[:, 1] - cy) / fy
        z = self.amostrar_z(depth_atual, pa, img.shape)
        zs = self.amostrar_z(self.goal['depth'], pg, img.shape)

        # 6. Publicação
        pares = np.column_stack([x, y, z, xs, ys, zs])
        self.publicar(stamp, pares)

        info = (f'goal: {len(self.goal["kp"])} kp | atual: {len(kp)} kp | inliers: {len(idx_in)} | '
                f'publicados: {len(sel)} | Z validos: {int((z > 0).sum())}/{len(sel)} | '
                f'selecao: {self.selecao}')
        self.mostrar_imagem(img, pg, pa, ranks, info)

    # ================================================================== visualização
    def mostrar_imagem(self, img, pg, pa, ranks, texto):
        if not self.mostrar:
            return
        if self.goal is not None:
            g = self.goal['img']
            atual = img if img.shape[0] == g.shape[0] else cv2.resize(
                img, (int(img.shape[1] * g.shape[0] / img.shape[0]), g.shape[0]))
            vis = np.hstack([g, atual])
            off = g.shape[1]
            cv2.putText(vis, 'GOAL', (10, g.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(vis, 'ATUAL', (off + 10, g.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        else:
            vis = img.copy()
            off = 0

        if pg is not None:
            for a, b, r in zip(pg, pa, ranks):
                pa_ = (int(b[0]) + off, int(b[1]))
                pg_ = (int(a[0]), int(a[1]))
                alvo = (int(a[0]) + off, int(a[1]))
                cv2.line(vis, pg_, pa_, (0, 255, 0), 1)
                cv2.arrowedLine(vis, pa_, alvo, (0, 255, 255), 1, tipLength=0.2)
                for p in (pg_, pa_):
                    cv2.circle(vis, p, 5, (0, 255, 0), 2)
                    # Número = prioridade na lista da goal: o mesmo ponto físico tem sempre o mesmo número
                    cv2.putText(vis, str(r), (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.rectangle(vis, (0, 0), (vis.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(vis, texto, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.imshow(JANELA, vis)
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
    node = XFeatFeatureExtractor()
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