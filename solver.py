import numpy as np
from matplotlib import pyplot as plt
from itertools import zip_longest
import time
import threading 
from functools import partial
import pandas as pd
from tqdm import tqdm

from mesh import Mesh2D, Node, MeshRectangular
from utils import thomas_method, tridiag

class Solver():
    def __init__(self, mesh:Mesh2D) -> None:
        self.mesh:Mesh2D = mesh

    def Gauss_seide(self, MAX_CONV_COND=1e-4, MAX_ITER_COND=None, anime=False):
        if anime:
            fig, ax = plt.subplots(1, 1)
        mesh = [[node for node in nodes if node is not None] for nodes in self.mesh.mesh]
        
        s = -1
        pbar = tqdm(total=None)
        k = 0
        while MAX_ITER_COND is None or MAX_ITER_COND > k:
            k += 1
            delta_w = []
            delta_psi = []
            #s = -1 if s == 1 else 1
            for nodes in mesh[::]:
                d_row_w = []
                d_row_psi = []
                for node in nodes:
                    d_row_w.append(node.update_w(node.calc_w(R=node.calc_I_residual_w())))
                    d_row_psi.append(node.update_psi(node.calc_psi(R=node.calc_I_residual_psi())))
                    
                delta_psi += d_row_psi
                delta_w += d_row_w
            if anime:
                self.mesh.anime_vector(ax, fig, t=0.01)
            updated_data =(f"psi = {max(delta_psi)} w = {max(delta_w)}")
            pbar.set_description(updated_data)
            if MAX_CONV_COND is not None and max([max(delta_psi), max(delta_w)]) < MAX_CONV_COND:
                break
        
    def coefficient_w_SOR(self, nodes:list[Node], omega, axis=0):
        As = [] 
        Bs = []
        Cs = []
        Ds = []
        for node in nodes:
            if axis == 0:
                a,b,c,d = node.calc_coefficient_w_line(omega)
            else:
                a,b,c,d = node.calc_coefficient_w_col(omega)
            As.append(a)
            Bs.append(b)
            Cs.append(c)
            Ds.append(d)
        return As, Bs, Cs, Ds

    def coefficient_psi_SOR(self, nodes:list[Node], omega, axis=0):
        As = [] 
        Bs = []
        Cs = []
        Ds = []
        for node in nodes:
            if axis == 0:
                a,b,c,d = node.calc_coefficient_psi_line(omega)
            else:
                a,b,c,d = node.calc_coefficient_psi_col(omega)
            As.append(a)
            Bs.append(b)
            Cs.append(c)
            Ds.append(d)
        return As, Bs, Cs, Ds
    
    def update_ws_line(self, nodes:list[Node], ws_new):
        return [node.update_w(w_new) for node, w_new in zip(nodes, ws_new)]

    def update_psis_line(self, nodes:list[Node], psis_new):
        return [node.update_psi(psi_new) for node, psi_new in zip(nodes, psis_new)]

    def SOR_by(self, omega, MAX_CONV_COND=1e-5, method=0, MAX_ITER_COND=None, anime=False, f_delta_omega=None):
        """
        SLOR combined with ADI -> method=-1 
        SOR by line            -> method= 0
        SOR by col             -> method= 1
        """
        if anime:
            fig, ax = plt.subplots(1, 1)
        mesh_i = [([node for node in nodes if node is not None], 0) for nodes in self.mesh.mesh]
        mesh_j = [([node for node in nodes if node is not None], 1) for nodes in np.array(self.mesh.mesh).T.tolist()]
        meshs = []
        if method == 0:
            meshs.append(mesh_i)
        elif method == 1:
            meshs.append(mesh_j)
        elif method == -1:
            meshs.append(mesh_i)
            meshs.append(mesh_j)
        elif method == -2:
            mesh_i_j = []
            for m_i, m_j in zip_longest(mesh_j, mesh_i):
                if m_i is not None:
                    mesh_i_j.append(m_i)
                if m_j is not None:
                    mesh_i_j.append(m_j)
            meshs.append(mesh_i_j)
        else:
            raise Exception("method is not valid.")
        s = -1
        
        pbar = tqdm(total=None)
        k = 0
        while MAX_ITER_COND is None or MAX_ITER_COND > k:
            s = 1
            #s = -1 if s == 1 else 1
            if f_delta_omega is not None:
                omega = f_delta_omega(omega)
            for mesh in meshs:
                delta_w = []
                delta_psi = []
                k += 1
                for nodes, axis in mesh[::]:
                    As, Bs, Cs, Ds = self.coefficient_w_SOR(nodes, omega, axis=axis)
                    #box = tridiag(As, Bs, Cs)
                    #ws_new = np.linalg.solve(box, Ds).flat
                    ws_new = thomas_method(As, Bs, Cs, Ds)
                    delta_w += self.update_ws_line(nodes, ws_new)
                    As, Bs, Cs, Ds = self.coefficient_psi_SOR(nodes, omega, axis=axis)
                    #box = tridiag(As, Bs, Cs)
                    #psis_new = np.linalg.solve(box, Ds).flat
                    psis_new = thomas_method(As, Bs, Cs, Ds)
                    delta_psi += self.update_psis_line(nodes, psis_new)
                    if anime:
                        self.mesh.anime_vector(ax, fig)
                updated_data = (f"psi = {max(delta_psi)} w = {max(delta_w)}")
                pbar.set_description(updated_data)
                if max([max(delta_psi), max(delta_w)]) < MAX_CONV_COND:
                    return


class MultiGridSolver():
    def __init__(self, mesh_base:Mesh2D, info_base:dict, layers:list) -> None:
        self.mesh_base:Mesh2D = mesh_base
        self.info_base = info_base
        self.solver_base:Solver = Solver(mesh=mesh_base)
        self.layers = layers
        self.meshs:list[MeshRectangular] = self.create_sub_meshs()
        self.solvers:list[Solver] = [Solver(mesh=mesh) for mesh in self.meshs]

    def create_sub_meshs(self):
        meshs = []
        for layer in self.layers:
            mesh:MeshRectangular = MeshRectangular(
                fs_x=self.mesh_base.fs_x, fs_y=self.mesh_base.fs_y, 
                ns_node_x=None, ns_node_y=None, boundrie_x=self.mesh_base.boundrie_x, boundrie_y=self.mesh_base.boundrie_y,
                info_default_psi=layer["info_default_psi"], info_default_w=layer["info_default_w"], 
                Length=self.mesh_base.Length, Re=self.mesh_base.Re
            )
            mesh.create_mesh_interpolate(self.mesh_base)
            meshs.append(mesh)
        return meshs
            

    def solve(self):
        solver, info = self.solver_base, self.info_base
        #forward
        for mesh, (solver, info) in zip([self.mesh_base]+self.meshs ,[(self.solver_base, self.info_base)] + list(zip(self.solvers,self.layers))):
            solver.Gauss_seide(
                MAX_ITER_COND=info["max_iter"],
                MAX_CONV_COND=info["max_cond_conv"],
            )
            mesh.update_Residuals_w_psi()
        #backward
        for mesh, (solver, info) in list(zip([self.mesh_base]+self.meshs ,[(self.solver_base, self.info_base)] + list(zip(self.solvers,self.layers))))[::-1][:-1:]:
            mesh.backward()
        return


def f_linear(a,b,x):
    return a*x+b

def f_sin(a,w,b,x):
    return a*np.sin(w*x)+b

def find_a(length, N_node, b):
    a2 = 1.0
    a1 = 0.0000000000001
    while True:
        x = 0
        count = 1
        while x < length:
            x += f_sin((a1+a2)/2.0,np.pi/length,b,x)
            count += 1
        if count > N_node:
            a1 = (a2+a1)/2.0
        if count < N_node:
            a2 = (a2+a1)/2.0
        if count == N_node:
            return (a2+a1)/2

def calc_AS_BS(boundries, NS):
    BS = []
    AS = []
    for bnd1,bnd2, N in zip([0.0]+boundries, boundries, NS):
        BS.append((bnd2-bnd1)/(N*2.0))
        AS.append(find_a(bnd2-bnd1, N, b=BS[-1]))
    return AS, BS

