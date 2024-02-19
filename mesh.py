from typing import Callable
from matplotlib import pyplot as plt
import seaborn as sns
from pyparsing import Any
from tqdm import tqdm
import numpy as np
import pandas as pd
from utils import nan_helper

class Mesh2D():
    def __init__(self, fs_y:Callable, fs_x:Callable, ns_node_x:list, ns_node_y:list, boundrie_x:list, boundrie_y:list, info_default_w:dict, info_default_psi:dict, Length:float, pooling=None, MAX_NODE_EACH_DIM=1000000, MIN_DELTA_Y=1.0e-3, MIN_DELTA_X=1.0e-3):
        self.fs_x = fs_x
        self.fs_y = fs_y
        self.pooling = pooling

        self.ns_node_x = ns_node_x
        self.ns_node_y = ns_node_y
        
        self.MAX_NODE_EACH_DIM = MAX_NODE_EACH_DIM
        self.MIN_DELTA_Y = MIN_DELTA_Y
        self.MIN_DELTA_X = MIN_DELTA_X
        
        self.Length = Length
        self.boundrie_x = boundrie_x
        self.boundrie_y = boundrie_y

        self.info_default_w   = info_default_w
        self.info_default_psi = info_default_psi
        self.mesh:list[list[Node]]

    def generate_guide_node(self, ns_node, fs, boundrie):
        guide_node = np.array([0.0])
        delta_node = []
        for bnd1, bnd2, n, f in zip([0]+boundrie[:-1], boundrie, ns_node, fs):
            l = bnd2 - bnd1
            batch_gn = np.vectorize(f)( ((np.linspace(0,1,n)))[1:] )*l+bnd1 
            if self.pooling is not None:
                batch_gn = batch_gn[:-1][1::2] + [batch_gn[-1]]
            guide_node = np.concatenate( (guide_node,  batch_gn))
        for x1, x2 in zip(guide_node[:-1], guide_node[1:]):
            delta_node.append(x2-x1)
        return guide_node.tolist(), delta_node

    def plot_AB(self, ax=None, fig=None, mode="U", label_on=True):
        """
        mode avilible -> "U", "V", "PSI", "W"
        """
        if fig is None or ax is None:
            fig, ax = plt.subplots(1, 1)
        if mode == "U":
            df = pd.DataFrame([[None if node is None else node.calc_U() for node in nodes] for nodes in self.mesh])
        elif mode == "V":
            df = pd.DataFrame([[None if node is None else node.calc_V() for node in nodes] for nodes in self.mesh])
        elif mode == "PSI":
            df = pd.DataFrame([[None if node is None else node.psi for node in nodes] for nodes in self.mesh])
        elif mode == "W":
            df = pd.DataFrame([[None if node is None else node.w for node in nodes] for nodes in self.mesh])
        else:
            raise Exception("mode is not valid.")
        ax.plot(self.y,df[df.shape[1]//2][::-1], label=self.Re)
        if label_on:
            ax.set_xlabel(f'{mode}')
            ax.set_ylabel('Y')
            ax.set_title(f"{mode} in AB line for Re={self.Re}")
        
    def plot_CD(self, ax=None, fig=None, mode="U", label_on=True):
        """
        mode avilible -> "U", "V", "PSI", "W"
        """
        if fig is None or ax is None:
            fig, ax = plt.subplots(1, 1)
        if mode == "U":
            df = pd.DataFrame([[None if node is None else node.calc_U() for node in nodes] for nodes in self.mesh])
        elif mode == "V":
            df = pd.DataFrame([[None if node is None else node.calc_V() for node in nodes] for nodes in self.mesh])
        elif mode == "PSI":
            df = pd.DataFrame([[None if node is None else node.psi for node in nodes] for nodes in self.mesh])
        elif mode == "W":
            df = pd.DataFrame([[None if node is None else node.w for node in nodes] for nodes in self.mesh])
        else:
            raise Exception("mode is not valid.")
        ax.plot(self.x,df.iloc[df.shape[0]//2][::-1], label=self.Re)
        if label_on:
            ax.set_xlabel(f'{mode}')
            ax.set_ylabel('Y')
            ax.set_title(f"{mode} in CD line for Re={self.Re}")
    
    def plot_mesh(self, ax=None, fig=None):
        x = []
        y = []
        x_b = []
        y_b = []
        if fig is None or ax is None:
            fig, ax = plt.subplots(1, 1)
        for nodes in self.mesh:
            for node in nodes:
                if node is None:
                    continue
                if node.N_walls[0] != -1:
                    x_b.append(node.x)
                    y_b.append(node.y)
                else:
                    x.append(node.x)
                    y.append(node.y)
        

        # نمایش تری‌مش
        ax.scatter(x,y, s=2, c="red")
        ax.scatter(x_b,y_b, s=2, c="#000000")

    def plot_contour(self, ax=None, fig=None, mode="U"):
        """
        mode avilible -> "U", "V", "PSI", "W"
        """
        if fig is None or ax is None:
            fig, ax = plt.subplots(1, 1)
        if mode == "U":
            df = pd.DataFrame([[None if node is None else node.calc_U() for node in nodes] for nodes in self.mesh])
        elif mode == "V":
            df = pd.DataFrame([[None if node is None else node.calc_V() for node in nodes] for nodes in self.mesh])
        elif mode == "PSI":
            df = pd.DataFrame([[None if node is None else node.psi for node in nodes] for nodes in self.mesh])
        elif mode == "W":
            df = pd.DataFrame([[None if node is None else node.w for node in nodes] for nodes in self.mesh])
        else:
            raise Exception("mode is not valid.")
        
        cp = ax.contour(self.x, self.y, df, colors='black', linestyles='dashed', linewidths=1)
        ax.clabel(cp, inline=1, fontsize=10)
        cp = ax.contourf(self.x, self.y, df)
        cb = fig.colorbar(cp)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f"{mode} for Re={self.Re}")

    def plot_vector(self, ax=None, fig=None):
        
        if fig is None or ax is None:
            fig, ax = plt.subplots(1, 1)

        df_U = pd.DataFrame([[None if node is None else node.calc_U() for node in nodes] for nodes in self.mesh])
        df_V = pd.DataFrame([[None if node is None else node.calc_V() for node in nodes] for nodes in self.mesh])

        ax.quiver(self.x, self.y, df_U, df_V)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f"vector U & V for Re={self.Re}")
    
    def plot_pcolor(self, ax=None, fig=None, mode="U"):
        """
        mode avilible -> "U", "V", "PSI", "W"
        """

        if fig is None or ax is None:
            fig, ax = plt.subplots(1, 1)

        if fig is None or ax is None:
            fig, ax = plt.subplots(1, 1)
        if mode == "U":
            df = pd.DataFrame([[None if node is None else node.calc_U() for node in nodes] for nodes in self.mesh])
        elif mode == "V":
            df = pd.DataFrame([[None if node is None else node.calc_V() for node in nodes] for nodes in self.mesh])
        elif mode == "PSI":
            df = pd.DataFrame([[None if node is None else node.psi for node in nodes] for nodes in self.mesh])
        elif mode == "W":
            df = pd.DataFrame([[None if node is None else node.w for node in nodes] for nodes in self.mesh])
        else:
            raise Exception("mode is not valid.")
        pc = ax.pcolormesh(self.x, self.y, df) 
        cb = fig.colorbar(pc)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f"{mode} for Re={self.Re}")
   
    def plot_residual(self, ax=None, fig=None, mode="W", n_end=None, label=None, label_on=True):
        """
        mode avilible -> "PSI", "W"
        """

        if n_end is None:
            n_end = len(self.mesh[0][0].residuals_w)
        if fig is None or ax is None:
            fig, ax = plt.subplots(1, 1)

        if fig is None or ax is None:
            fig, ax = plt.subplots(1, 1)
        if mode == "W":
            residual = []
            for nodes in self.mesh:
                for node in nodes:
                    if node is None:
                        continue
                    residual.append(node.residuals_w[-n_end:])
        elif mode == "PSI":            
            residual = []
            for nodes in self.mesh:
                for node in nodes:
                    if node is None:
                        continue
                    residual.append(node.residuals_psi[-n_end:])
        else:
            raise Exception("mode is not valid.")
        ax.plot(pd.DataFrame((residual)).mean(), label=label)
        if label_on:
            ax.set_xlabel('iteration')
            ax.set_ylabel('residual')
            ax.set_title(f"{mode} iteration-residual")

    def anime_residual(self, ax, fig, mode="W", t=0.1, n_end=None, label=None):
        """
        mode avilible -> "PSI", "W"
        """
        if n_end is None:
            n_end = len(self.mesh[0][0].residuals_w)
        if fig is None or ax is None:
            fig, ax = plt.subplots(1, 1)

        if fig is None or ax is None:
            fig, ax = plt.subplots(1, 1)
        x = range(max([len(self.mesh[0][0].residuals_w)-n_end, 0]), len(self.mesh[0][0].residuals_w))
        if mode == "W":
            residual = []
            for nodes in self.mesh:
                for node in nodes:
                    if node is None:
                        continue
                    residual.append(node.residuals_w[-n_end:])
        elif mode == "PSI":            
            residual = []
            for nodes in self.mesh:
                for node in nodes:
                    if node is None:
                        continue
                    residual.append(node.residuals_psi[-n_end:])
        else:
            raise Exception("mode is not valid.")
        ax.clear()
        ax.plot(x,pd.DataFrame(residual).mean(), label=label)
        ax.set_xlabel('iteration')
        ax.set_ylabel('residual')
        ax.set_title(f"{mode} iteration-residual")
        plt.pause(t)

    def anime_vector(self, ax, fig, t=0.1):
        
        df_U = pd.DataFrame([[None if node is None else node.calc_U() for node in nodes] for nodes in self.mesh])
        df_V = pd.DataFrame([[None if node is None else node.calc_V() for node in nodes] for nodes in self.mesh])
        ax.clear()
        ax.quiver(self.x, self.y, df_U, df_V)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f"vector U & V for Re={self.Re}")
        plt.pause(t)
        return
    
    def anime_contour(self, ax, fig, mode="U", t=0.1):
        """
        mode avilible -> "U", "V", "PSI", "W"
        """

        if fig is None or ax is None:
            fig, ax = plt.subplots(1, 1)

        if fig is None or ax is None:
            fig, ax = plt.subplots(1, 1)
        if mode == "U":
            df = pd.DataFrame([[None if node is None else node.calc_U() for node in nodes] for nodes in self.mesh])
        elif mode == "V":
            df = pd.DataFrame([[None if node is None else node.calc_V() for node in nodes] for nodes in self.mesh])
        elif mode == "PSI":
            df = pd.DataFrame([[None if node is None else node.psi for node in nodes] for nodes in self.mesh])
        elif mode == "W":
            df = pd.DataFrame([[None if node is None else node.w for node in nodes] for nodes in self.mesh])
        else:
            raise Exception("mode is not valid.")
        pc = ax.contour(self.x, self.y, df, 20, cmap='RdGy') 
        cb = fig.colorbar(pc)
        #plt.contourf(x, Y, Z, 20, cmap='RdGy')
        #plt.colorbar()
        ax.clabel(pc, fontsize=5)
        plt.pause(t)
        cb.remove()

    @property
    def info(self):
        mesh = pd.DataFrame(self.mesh)
        mesh_count_y = mesh.count(axis=0)
        mesh_count_x = mesh.count(axis=1)

        print(f"""
Number of node              : {mesh_count_x.sum()}
max & min number of node x  : {max(mesh_count_x)} - {min(mesh_count_x)}
max & min number of node y  : {max(mesh_count_y)} - {min(mesh_count_y)}
        """)
 
class MeshRectangular(Mesh2D):
    def __init__(self, fs_y: Callable[..., Any], fs_x: Callable[..., Any], ns_node_x: list, ns_node_y: list, boundrie_x: list, boundrie_y: list, info_default_w: dict, info_default_psi: dict, Length: float, pooling=None, MAX_NODE_EACH_DIM=1000000, MIN_DELTA_Y=0.001, MIN_DELTA_X=0.001, Re=1.0):
        super().__init__(fs_y, fs_x, ns_node_x, ns_node_y, boundrie_x, boundrie_y, info_default_w, info_default_psi, Length, pooling, MAX_NODE_EACH_DIM, MIN_DELTA_Y, MIN_DELTA_X)
        self.Re = Re

    def generate_guide_node_interpolate(self, value_base:list, boundrie_axis):
        guide_node = []
        delta_node = []
        for bn1, bn2 in zip([0.0]+boundrie_axis[:-1],boundrie_axis):
            for x in value_base[value_base.index(bn1):value_base.index(bn2):2]:
                guide_node.append(x)
        if self.Length != guide_node[-1]:
            guide_node.append(self.Length)
        for x1, x2 in zip(guide_node[:-1], guide_node[1:]):
            delta_node.append(x2-x1)
        return guide_node, delta_node

    def update_Residuals_w_psi(self):
        for nodes in self.mesh:
            for node in nodes:
                if node is not None:
                    node.update_Residual_w()
                    node.update_Residual_psi()

    def backward(self):
        self.update_delta()


    def create_mesh_interpolate(self, mesh_base:"MeshRectangular"):
        self.mesh_base = mesh_base
        self.x, self.delta_x = self.generate_guide_node_interpolate(value_base=mesh_base.x, boundrie_axis=mesh_base.boundrie_x)
        self.y, self.delta_y = self.generate_guide_node_interpolate(value_base=mesh_base.y, boundrie_axis=mesh_base.boundrie_y)
        x, y = self.x, self.y
        self.mesh:list[list[Node]] = []
        for j in tqdm(range(len(y))):
            self.mesh.append([])
            for i in range(len(x)):
                if y[j] < self.boundrie_y[1] and y[j] > self.boundrie_y[0] and (x[i] < self.boundrie_x[0] or x[i] > self.boundrie_x[1]):
                    self.mesh[-1].append(None)
                    continue
                self.mesh[-1].append(Node.__new__(Node))        
        for j in tqdm(range(len(y))):
            for i in range(len(x)):
                if y[j] < self.boundrie_y[1] and y[j] > self.boundrie_y[0] and (x[i] < self.boundrie_x[0] or x[i] > self.boundrie_x[1]):
                    continue
                N_walls = self.calc_N_wall(x[i], y[j])
                NBN = [
                    self.mesh[j+1][i] if j!=len(y)-1 else None, 
                    self.mesh[j][i+1] if i!=len(x)-1 else None, 
                    self.mesh[j-1][i] if j!=0 else None, 
                    self.mesh[j][i-1] if i!=0 else None, 
                ]
                dx = [
                    None if NBN[3] is None else self.delta_x[i-1], 
                    None if NBN[1] is None else self.delta_x[i]
                ]
                dy = [
                    None if NBN[2] is None else self.delta_y[j-1], 
                    None if NBN[0] is None else self.delta_y[j] 
                ]
                interpolate_node:Node = MeshRectangular.find_node(mesh_base.mesh, x=x[i], y=y[j], i=i, j=j)
                self.mesh[j][i].__init__(
                    i,j, x[i], y[j], NBN=NBN, delta_x=dx, delta_y=dy, N_walls=N_walls, default_val_psi=self.info_default_psi[N_walls[0]] , default_val_w=self.info_default_w[N_walls[0]], Re=self.Re, interpolate_node=interpolate_node, is_base_node=False
                )
                
        return self.mesh

    @staticmethod 
    def find_node(mesh, x, y, i=0, j=0):
        for nodes in mesh[j:]:
            for node in nodes[i:]:
                if node is not None and (node.x == x and node.y == y):
                    return node
        
    def create_mesh(self):
        self.x, self.delta_x = self.generate_guide_node(
            ns_node=self.ns_node_x,
            fs=self.fs_x, 
            boundrie=self.boundrie_x
        )
        self.y, self.delta_y = self.generate_guide_node(
            ns_node=self.ns_node_y,
            fs=self.fs_y, 
            boundrie=self.boundrie_y
        )
        x, y = self.x, self.y
        self.mesh:list[list[Node]] = []
        for j in tqdm(range(len(y))):
            self.mesh.append([])
            for i in range(len(x)):
                if y[j] < self.boundrie_y[1] and y[j] > self.boundrie_y[0] and (x[i] < self.boundrie_x[0] or x[i] > self.boundrie_x[1]):
                    self.mesh[-1].append(None)
                    continue
                self.mesh[-1].append(Node.__new__(Node))

        for j in tqdm(range(len(y))):
            for i in range(len(x)):
                if y[j] < self.boundrie_y[1] and y[j] > self.boundrie_y[0] and (x[i] < self.boundrie_x[0] or x[i] > self.boundrie_x[1]):
                    continue

                N_walls = self.calc_N_wall(x[i], y[j])
                NBN = [
                    self.mesh[j+1][i] if j!=len(y)-1 else None, 
                    self.mesh[j][i+1] if i!=len(x)-1 else None, 
                    self.mesh[j-1][i] if j!=0 else None, 
                    self.mesh[j][i-1] if i!=0 else None, 
                ]
                dx = [
                    None if NBN[3] is None else self.delta_x[i-1], 
                    None if NBN[1] is None else self.delta_x[i]
                ]
                dy = [
                    None if NBN[2] is None else self.delta_y[j-1], 
                    None if NBN[0] is None else self.delta_y[j] 
                ]

                
                self.mesh[j][i].__init__(
                    i,j, x[i], y[j], NBN=NBN, delta_x=dx, delta_y=dy, N_walls=N_walls, default_val_psi=self.info_default_psi[N_walls[0]] , default_val_w=self.info_default_w[N_walls[0]], Re=self.Re
                )
        return self.mesh

    def calc_N_wall(self, x, y):
        N_walls = []
        if y == 0.0:
            N_walls.append(12)
        if abs(y - self.boundrie_y[0]) < 1e-16:
            if x <= self.boundrie_x[0]:
                N_walls.append(2)
            if x >= self.boundrie_x[1]:
                N_walls.append(10)
        
        if abs(y - self.boundrie_y[1])<1e-16 :
            if x <= self.boundrie_x[0]:
                N_walls.append(4)
            if x >= self.boundrie_x[1]:
                N_walls.append(8)
        
        if x == 0.0:
            if y <= self.boundrie_y[0]:
                N_walls.append(1)
            if y >= self.boundrie_y[1]:
                N_walls.append(5)
        if abs(x - self.boundrie_x[0]) < 1e-16:
            if y >= self.boundrie_y[0] and y <= self.boundrie_y[1]:
                N_walls.append(3)
        if abs(x - self.boundrie_x[1]) < 1e-16:
            if y >= self.boundrie_y[0] and y <= self.boundrie_y[1]:
                N_walls.append(9)
        if abs(x - self.Length) < 1e-16:
            if y <= self.boundrie_y[0]:
                N_walls.append(11)
            if y >= self.boundrie_y[1]:
                N_walls.append(7)
                
        if abs(y - self.Length) < 1e-16:
            N_walls.append(6)
        if len(N_walls) == 0:
            return [-1]
        return N_walls

    def calc_N_wall_(self, x, y):
        N_walls = []
        if y == 0.0:
            N_walls.append(12)
        if y == self.boundrie_y[0] :
            N_walls.append(6)
        
        
        if x == 0.0:
            N_walls.append(1)
        if x == self.boundrie_x[0]:
            N_walls.append(7)
        if len(N_walls) == 0:
            return [-1]
        return N_walls


    def update_delta(self):
        m = np.full((len(self.mesh_base.y), len(self.mesh_base.x)), np.nan)
        for nodes in self.mesh_base.mesh:
            for node in nodes:
                if node is None:
                    continue
                inter:Node = MeshRectangular.find_node(self.mesh, x=node.x, y=node.y)
                if inter is not None:
                    m[node.j, node.i] = inter.w
        m_copy = m.copy()

        # پیدا کردن مکان‌هایی که مقدار آنها `NaN` است
        nan_indices = np.isnan(m)

        # جایگزینی مقادیر `NaN` با میانگین مقادیر دو طرف از آنها
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if nan_indices[i, j] and self.mesh_base.mesh[i][j] is not None:
                    # پیدا کردن مقادیر دو طرف از `NaN`
                    left_value = m_copy[i, j-1] if j > 0 else np.nan
                    right_value = m_copy[i, j+1] if j < m.shape[1]-1 else np.nan
                    down_value = m_copy[i-1, j] if i > 0 else np.nan
                    up_value = m_copy[i+1, j] if i < m.shape[0]-1 else np.nan
                    # محاسبه میانگین و جایگزینی `NaN`
                    m[i, j] = np.nanmean([up_value, down_value, left_value, right_value])
        nan_indices = np.isnan(m)

        # جایگزینی مقادیر `NaN` با میانگین مقادیر دو طرف از آنها
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if nan_indices[i, j] and self.mesh_base.mesh[i][j] is not None:
                    # پیدا کردن مقادیر دو طرف از `NaN`
                    left_value = m[i, j-1] if j > 0 else np.nan
                    right_value = m[i, j+1] if j < m.shape[1]-1 else np.nan
                    down_value = m[i-1, j] if i > 0 else np.nan
                    up_value = m[i+1, j] if i < m.shape[0]-1 else np.nan
                    # محاسبه میانگین و جایگزینی `NaN`
                    m[i, j] = np.nanmean([up_value, down_value, left_value, right_value])

        for nodes_base, row in zip(self.mesh_base.mesh, m):
            for node_b, delta_w in zip(nodes_base, row):
                if node_b is None:
                    continue
                node_b.update_w(node_b.w+delta_w)




        ########################################### psi ################################## 
        m = np.full((len(self.mesh_base.y), len(self.mesh_base.x)), np.nan)
        for nodes in self.mesh_base.mesh:
            for node in nodes:
                if node is None:
                    continue
                inter:Node = MeshRectangular.find_node(self.mesh, x=node.x, y=node.y)
                if inter is not None:
                    m[node.j, node.i] = inter.psi

        # پیدا کردن مکان‌هایی که مقدار آنها `NaN` است
        m_copy = m.copy()
        nan_indices = np.isnan(m)

        # جایگزینی مقادیر `NaN` با میانگین مقادیر دو طرف از آنها
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if nan_indices[i, j] and self.mesh_base.mesh[i][j] is not None:
                    # پیدا کردن مقادیر دو طرف از `NaN`
                    left_value = m_copy[i, j-1] if j > 0 else np.nan
                    right_value = m_copy[i, j+1] if j < m.shape[1]-1 else np.nan
                    down_value = m_copy[i-1, j] if i > 0 else np.nan
                    up_value = m_copy[i+1, j] if i < m.shape[0]-1 else np.nan
                    # محاسبه میانگین و جایگزینی `NaN`
                    m[i, j] = np.nanmean([up_value, down_value, left_value, right_value])

        nan_indices = np.isnan(m)

        # جایگزینی مقادیر `NaN` با میانگین مقادیر دو طرف از آنها
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if nan_indices[i, j] and self.mesh_base.mesh[i][j] is not None:
                    # پیدا کردن مقادیر دو طرف از `NaN`
                    left_value = m[i, j-1] if j > 0 else np.nan
                    right_value = m[i, j+1] if j < m.shape[1]-1 else np.nan
                    down_value = m[i-1, j] if i > 0 else np.nan
                    up_value = m[i+1, j] if i < m.shape[0]-1 else np.nan
                    # محاسبه میانگین و جایگزینی `NaN`
                    m[i, j] = np.nanmean([up_value, down_value, left_value, right_value])

        for nodes_base, row in zip(self.mesh_base.mesh, m):
            for node_b, delta_psi in zip(nodes_base, row):
                if node_b is None:
                    continue
                node_b.update_psi(node_b.psi+delta_psi)
        return 


class MeshCircular(Mesh2D):
    def __init__(self, fs_y: Callable[..., Any], fs_x: Callable[..., Any], ns_node_x: list, ns_node_y: list, boundrie_x: list, boundrie_y: list, info_default_w: dict, info_default_psi: dict, Length: float, pooling=None, MAX_NODE_EACH_DIM=1000000, MIN_DELTA_Y=0.001, MIN_DELTA_X=0.001, Re=1.0):
        super().__init__(fs_y, fs_x, ns_node_x, ns_node_y, boundrie_x, boundrie_y, info_default_w, info_default_psi, Length, pooling, MAX_NODE_EACH_DIM, MIN_DELTA_Y, MIN_DELTA_X)
        self.Re = Re
        self.Radius = self.Length/6.0

    def generate_guide_node_circular_x(self, ns_node, fs, boundrie, ys):
        guide_node = np.array([0.0])

        delta_node = []
        circule_mesh = np.vectorize(lambda y: np.sqrt(-(y-self.Length/2)**2.0+self.Radius**2.0))(ys[:len(ys)//2])
        guide_node = np.concatenate( (guide_node,  circule_mesh))
        bnd1, bnd2, n, f = boundrie[0], boundrie[1], ns_node[1], fs[1]
        l = bnd2 - bnd1
        batch_gn = np.vectorize(f)( ((np.linspace(0,1,n)))[:] )*l+bnd1 
        guide_node = np.concatenate( (guide_node,  batch_gn))
        
        guide_node = np.concatenate( (guide_node,  (boundrie[-1]-circule_mesh[::-1])))
        guide_node = np.concatenate( (guide_node,  [1.0]))
        for x1, x2 in zip(guide_node[:-1], guide_node[1:].copy()):
            delta_node.append(x2-x1)
        return guide_node.tolist(), delta_node

    def pos_circle(self, y):
        x = np.sqrt((self.Radius)**2.0-(y-self.Length/2.0)**2.0)
        return (x, self.Length-x)

    def create_mesh(self):
        self.y, self.delta_y = self.generate_guide_node(
            ns_node=self.ns_node_y,
            fs=self.fs_y, 
            boundrie=self.boundrie_y
        )
        self.x, self.delta_x = self.generate_guide_node_circular_x(
            ns_node=self.ns_node_x,
            fs=self.fs_x, 
            boundrie=self.boundrie_x,
            ys=self.y[self.y.index(self.boundrie_y[0])+1:self.y.index(self.boundrie_y[1])]
        )
        x, y = self.x, self.y
        self.mesh:list[list[Node]] = []
        for j in tqdm(range(len(y))):
            self.mesh.append([])
            for i in range(len(x)):
                x1, x2 = self.pos_circle(y[j])
                if y[j] < self.Length/2.0+self.Radius and y[j] > self.Length/2.0-self.Radius and (x[i] < x1 or x[i] > x2):
                    self.mesh[-1].append(None)
                    continue
                self.mesh[-1].append(Node.__new__(Node))

        for j in tqdm(range(len(y))):
            for i in range(len(x)):
                x1, x2 = self.pos_circle(y[j])
                if y[j] < self.Length/2.0+self.Radius and y[j] > self.Length/2.0-self.Radius and (x[i] < x1 or x[i] > x2):
                    continue

                N_walls = self.calc_N_wall(x[i], y[j], i)
                NBN = [
                    self.mesh[j+1][i] if j!=len(y)-1 else None, 
                    self.mesh[j][i+1] if i!=len(x)-1 else None, 
                    self.mesh[j-1][i] if j!=0 else None, 
                    self.mesh[j][i-1] if i!=0 else None, 
                ]
                dx = [
                    None if NBN[3] is None else self.delta_x[i-1], 
                    None if NBN[1] is None else self.delta_x[i]
                ]
                dy = [
                    None if NBN[2] is None else self.delta_y[j-1], 
                    None if NBN[0] is None else self.delta_y[j] 
                ]
                
                self.mesh[j][i].__init__(
                    i,j, x[i], y[j], NBN=NBN, delta_x=dx, delta_y=dy, N_walls=N_walls, default_val_psi=self.info_default_psi[N_walls[0]] , default_val_w=self.info_default_w[N_walls[0]], Re=self.Re
                )
        return self.mesh

    def calc_N_wall(self, x, y, i):
        N_walls = []
        if y == 0.0:
            N_walls.append(8)
        if y == self.Length:
            N_walls.append(4)
        if y <= self.Length/2.0-self.Radius:
            if x == 0.0:
                N_walls.append(1)
            if x == self.Length:
                N_walls.append(7)
        elif y >= self.Length/2.0+self.Radius:
            if x == 0.0:
                N_walls.append(3)
            if x == self.Length:
                N_walls.append(5)
        else :
            x1, x2 = self.pos_circle(y)
            if x1 == x:
                N_walls.append(2)
            if x2 == x:
                N_walls.append(6)
        if len(N_walls) == 0:
            N_walls.append(-1)
        return N_walls



class Node():
    def __init__(self, i, j, x, y, delta_x, delta_y, NBN:list["Node"], N_walls, default_val_w:float, default_val_psi:float, Re:float, interpolate_node:"Node"=None, is_base_node=True):
        self.i, self.j = i, j
        self.x, self.y = x, y
        self.N_walls = N_walls
        self.default_val_w, self.default_val_psi = default_val_w, default_val_psi
        self.Re = Re
        self.ws = [self.default_val_w]
        self.psis = [self.default_val_psi]
        self.is_base_node = is_base_node

        self.NBN:list[Node] = NBN
        self.delta_x, self.delta_y = delta_x, delta_y
        
        if is_base_node:
            self.calc_I_residual_w = lambda :0.0
            self.calc_I_residual_psi = lambda :0.0
        else:
            self.calc_I_residual_w = lambda :self.Residual_w
            self.calc_I_residual_psi = lambda :self.Residual_psi

        self.calc_w, self.calc_psi = self.set_w_psi_func()
        self.calc_coefficient_w_line, self.calc_coefficient_psi_line = self.set_coefficient_w_psi_line()
        self.calc_coefficient_w_col, self.calc_coefficient_psi_col = self.set_coefficient_w_psi_col()
        self.calc_U, self.calc_V = self.set_UW_func()
        self.residuals_w = []
        self.residuals_psi = []
        self.Residual_w = 0.0
        self.Residual_psi = 0.0
        
        self.interpolate_node = interpolate_node

    def update_delta_w(self, sub_mesh:MeshRectangular):
        pass
    def update_delta_psi(self):
        self.update_psi(self.psi + sum(map(lambda node: node.psi,self.childs_interpolate))/len(self.childs_interpolate))

    def update_Residual_w(self):
        return self.calc_w(R=self.calc_I_residual_w()) - self.w
    
    def update_Residual_psi(self):
        return self.calc_psi(R=self.calc_I_residual_psi()) - self.psi

    def update_w(self, w_new):
        self.residuals_w.append(abs(-w_new+self.ws[-1]))
        self.ws.append(w_new)
        return self.residuals_w[-1]
    
    def update_psi(self, psi_new):
        self.residuals_psi.append(abs(-psi_new+self.psis[-1]))
        self.psis.append(psi_new)
        return self.residuals_psi[-1]

    @property
    def w(self):
        return self.ws[-1]
    
    @property 
    def psi(self):
        return self.psis[-1]

    def set_UW_func(self):
        N_walls = self.N_walls
        info_wall = {
            -1: (None, None),

            1: (None, lambda : 0.0),
            3: (None, lambda : 0.0),
            5: (None, lambda : 0.0),

            11: (None, lambda : 0.0),
            9:  (None, lambda : 0.0),
            7:  (None, lambda : 0.0),

            10: (lambda : 0.0,None),
            6:  (lambda : 1.0,None),
            2:  (lambda : 0.0,None),

            12: (lambda : 0.0, None),
            4:  (lambda : 0.0, None),
            8:  (lambda : 0.0, None),
        }
        U, V = info_wall[N_walls[0]]
        for N_wall in N_walls[1:]:
            _U, _V = info_wall[N_wall]
            if _U is not None:
                U = _U
            if _V is not None:
                V = _V

        if U is None:
            U = self.calc_U
        if V is None:
            V = self.calc_V
        return (U, V)
    
    def set_w_psi_func(self):
        N_wall = self.N_walls[0]
        calc_w   = self.calc_W_general
        calc_psi = self.calc_psi_general
        if N_wall != -1:
            calc_psi = lambda R=0.0: 0.0
        if N_wall in [1,3,5]:
            calc_w   = self.calc_w_wall_left
        elif N_wall in [11,9,7]:
            calc_w   = self.calc_w_wall_right
        elif N_wall in [12,8,4]:
            calc_w   = self.calc_w_wall_down
        elif N_wall in [2,6,10]:
            calc_w   = self.calc_w_wall_up
        return calc_w, calc_psi

    def coefficient_w_general_line(self, omega):
    
        temp1 = omega*self.delta_x[0]*self.delta_x[1]*self.delta_y[0]*self.delta_y[1]/(-2.0*self.delta_y[0]*self.delta_y[1] - 2.0*self.delta_x[0]*self.delta_x[1])
        temp2 = 1.0/(self.delta_x[0]*self.delta_x[1])
        temp3 = self.Re*(self.NBN[0].psi-self.NBN[2].psi)/sum(self.delta_y)/sum(self.delta_x)
        a = temp1 * (temp2+temp3)
        c = temp1 * (temp2-temp3)
        b = 1.0
        d = self.w*(1-omega) - temp1*(
            self.Re*
            (self.NBN[1].psi-self.NBN[3].psi)/sum(self.delta_x)*
            (self.NBN[0].w-self.NBN[2].w)/sum(self.delta_y) +
            (self.NBN[0].w + self.NBN[2].w)/(self.delta_y[0]*self.delta_y[1])
        )
        return a,b,c,d
        
    def coefficient_w_boundry(self, omega):
        a = 0.0
        b = 1.0
        c = 0.0
        d = self.w*(1-omega) + omega*self.calc_w()
        return a, b, c, d
    
    def coefficient_psi_general_line(self, omega):
        temp1 = -omega/(2.0+2.0*self.delta_x[0]*self.delta_x[1]/(self.delta_y[0]*self.delta_y[1]))
        a = temp1
        b = 1.0
        c = temp1
        d = self.psi*(1-omega) + omega*(
                (self.w+(self.NBN[0].psi+self.NBN[2].psi)/(self.delta_y[0]*self.delta_y[1]))/(2.0/(self.delta_x[0]*self.delta_x[1]) + 2.0/(self.delta_y[0]*self.delta_y[1]))
            )
        return a,b,c,d
    
    def coefficient_psi_boundry(self, omega):
        a = 0.0
        b = 1.0
        c = 0.0
        d = 0.0
        return a,b,c,d


    def coefficient_w_general_col(self, omega):
    
        temp1 = omega*self.delta_x[0]*self.delta_x[1]*self.delta_y[0]*self.delta_y[1]/(-2.0*self.delta_y[0]*self.delta_y[1] - 2.0*self.delta_x[0]*self.delta_x[1])
        temp2 = 1.0/(self.delta_y[0]*self.delta_y[1])
        temp3 = self.Re*(self.NBN[1].psi-self.NBN[3].psi)/sum(self.delta_y)/sum(self.delta_x)
        a = temp1 * (temp2-temp3)
        c = temp1 * (temp2+temp3)
        b = 1.0
        d = self.w*(1-omega) - temp1*(
            self.Re*
            (self.NBN[0].psi-self.NBN[2].psi)/sum(self.delta_y)*
            (self.NBN[1].w-self.NBN[3].w)/sum(self.delta_x) -
            (self.NBN[1].w + self.NBN[3].w)/(self.delta_x[0]*self.delta_x[1])
        )
        return a,b,c,d
            
    def coefficient_psi_general_col(self, omega):
        temp1 = -omega/(2.0+2.0*self.delta_y[0]*self.delta_y[1]/(self.delta_x[0]*self.delta_x[1]))
        a = temp1
        b = 1.0
        c = temp1
        d = self.psi*(1-omega) + omega*(
            (self.w+(self.NBN[1].psi+self.NBN[3].psi)/(self.delta_x[0]*self.delta_x[1]))/(2.0/(self.delta_x[0]*self.delta_x[1]) + 2.0/(self.delta_y[0]*self.delta_y[1]))
        )
        return a,b,c,d


    def set_coefficient_w_psi_line(self):
        N_wall = self.N_walls[0]
        if N_wall == -1:
            self.calc_coefficient_w_line = self.coefficient_w_general_line
            self.calc_coefficient_psi_line = self.coefficient_psi_general_line
        else:
            self.calc_coefficient_w_line = self.coefficient_w_boundry
            self.calc_coefficient_psi_line = self.coefficient_psi_boundry
        return self.calc_coefficient_w_line, self.calc_coefficient_psi_line
    
    def set_coefficient_w_psi_col(self):
        N_wall = self.N_walls[0]
        if N_wall == -1:
            self.calc_coefficient_w_col = self.coefficient_w_general_col
            self.calc_coefficient_psi_col = self.coefficient_psi_general_col
        else:
            self.calc_coefficient_w_col = self.coefficient_w_boundry
            self.calc_coefficient_psi_col = self.coefficient_psi_boundry
        return self.calc_coefficient_w_col, self.calc_coefficient_psi_col
    
    def calc_W_general(self, R=0.0):
        """
                  NBN[0]
                    |
                    |
        NBN[3] ---- . ------NBN[1]
                    |
                    |
                  NBN[2]
       """
        return (
            self.delta_x[0]*self.delta_x[1]*self.delta_y[0]*self.delta_y[1]/
            (-2.0*self.delta_y[0]*self.delta_y[1] -2.0*self.delta_x[0]*self.delta_x[1]) *
            (
                self.Re*(
                    (self.NBN[0].psi-self.NBN[2].psi)/sum(self.delta_y) * (self.NBN[1].w-self.NBN[3].w)/sum(self.delta_x) -
                    (self.NBN[1].psi-self.NBN[3].psi)/sum(self.delta_x) * (self.NBN[0].w-self.NBN[2].w)/sum(self.delta_y)
                ) - 
                (self.NBN[1].w+self.NBN[3].w)/(self.delta_x[0]*self.delta_x[1]) -
                (self.NBN[0].w+self.NBN[2].w)/(self.delta_y[0]*self.delta_y[1])
            )
        )+R
    
    def calc_psi_general(self, R=0.0):
        """
                  NBN[0]
                    |
                    |
        NBN[3] ---- . ------NBN[1]
                    |
                    |
                  NBN[2]
       """
        return (
            (
                self.w + 
                (self.NBN[1].psi+self.NBN[3].psi)/(self.delta_x[0]*self.delta_x[1]) +
                (self.NBN[0].psi+self.NBN[2].psi)/(self.delta_y[0]*self.delta_y[1])
            ) / 
            (
                2.0/(self.delta_x[0]*self.delta_x[1]) + 2.0/(self.delta_y[0]*self.delta_y[1])
            )
        )+R

    def calc_w_wall_right(self, R=0.0):
        return (-self.NBN[3].psi*2.0/self.delta_x[0]**2.0 + self.calc_V()*2.0/self.delta_x[0])+R
        
    def calc_w_wall_left(self, R=0.0):
        return (-self.NBN[1].psi*2.0/self.delta_x[1]**2.0 - self.calc_V()*2.0/self.delta_x[1])+R
    
    def calc_w_wall_up(self, R=0.0):
        return (-self.NBN[2].psi*2.0/self.delta_y[0]**2.0 - self.calc_U()*2.0/self.delta_y[0])+R
    
    def calc_w_wall_down(self, R=0.0):
        return (-self.NBN[0].psi*2.0/self.delta_y[1]**2.0 + self.calc_U()*2.0/self.delta_y[1])+R

    def calc_U(self):
        return (self.NBN[0].psi - self.NBN[2].psi) / sum(self.delta_y)
    
    def calc_V(self):
        return (-self.NBN[1].psi + self.NBN[3].psi) / sum(self.delta_x)
    
    def calc_interpolate_w(self):
        return self.Residual_w
    def calc_interpolate_psi(self):
        return self.Residual_psi

    def stability(self):
        return self.calc_U()*self.delta_x[1] + self.calc_V()*self.delta_y[1]

