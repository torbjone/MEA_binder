#!/usr/bin/env python
'''
Cortical pyramidal cell model from Almog and Korngreen (2014) J Neurosci 34:1 182-196

This file is only supplimentary to the Ipython Notebook file index.ipynb
'''

import sys, os
import numpy as np
import pylab as plt
import neuron
import LFPy
from vimeapy import MoI

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

class Electrode:
    """ Class to make the electrodes.
    """
    def __init__(self, slice_thickness=300., elec_radius=15., elec_x=[0], elec_y=[0], elec_z=[0]):
        self.slice_thickness = slice_thickness
        self.elec_radius = elec_radius
        self.elec_x = np.array(elec_x, dtype=float)
        self.elec_y = np.array(elec_y, dtype=float)
        self.elec_z = np.array(elec_z, dtype=float)  # THIS IS AN ARRAY NOW!
        self.num_elecs = len(self.elec_x)
        self.elec_clr = [plt.cm.rainbow(1./(self.num_elecs - 1) * idx) for idx in range(self.num_elecs)]


class SimulateMEA:
    """ Class to investigate neural activity following synaptic input, and calculated the
    extracellular potential at a microelectrode array (MEA) plane.
    """
    def __init__(self, syn_soma_pos, syn_apic_pos, slice_thickness, cell_z_pos, elec_x, elec_y):

        self.syn_soma_pos = syn_soma_pos  # [x, y, z] position of synaptic input
        self.syn_apic_pos = syn_apic_pos  # [x, y, z] position of synaptic input
        self.syn_weight = 0.008  # Strength of synaptic input

        self.input_spike_train_soma = np.array([12., 15.])  # Set time (ms) of synaptic input
        self.input_spike_train_apic = np.array([12., 15.])  # Set time (ms) of synaptic input
        elec_params = {'slice_thickness': slice_thickness,
                   'elec_x': elec_x,
                   'elec_y': elec_y,
                   'elec_z': np.zeros(len(elec_x)),
                   }

        MEA = Electrode(**elec_params)
        ext_sim_dict = {'use_line_source': False,
                        'n_elecs': MEA.num_elecs,
                        'moi_steps': 20,
                        'elec_x': MEA.elec_x,
                        'elec_y': MEA.elec_y,
                        'elec_z': MEA.elec_z,
                        'slice_thickness': MEA.slice_thickness,
                        'include_elec': False,
                        'neural_input': '.',
                        }

        cell, synapse_soma, synapse_apic = self.make_cell(ext_sim_dict, cell_z_pos, MEA)
        self.plot_results(cell, synapse_soma, synapse_apic, MEA)

    def make_cell(self, ext_sim_dict, cell_z_pos, MEA):
        cell_parameters = {
            'morphology': 'A140612.hoc',  # File with cell morphology
            'v_init': -62,
            'passive': False,
            'nsegs_method': None,
            'timeres_NEURON': 2**-3,  # [ms] Should be a power of 2
            'timeres_python': 2**-3,
            'tstartms': -50,  # [ms] Simulation start time
            'tstopms': 50,  # [ms] Simulation end time
            'custom_code': ['cell_model.hoc']  # Loads model specific code
        }

        cell = LFPy.Cell(**cell_parameters)

        # Specify the position and rotation of the cell
        cell.set_rotation(z=-np.pi/2.2, y=np.pi, x=0.03)
        cell.set_pos(zpos=cell_z_pos)

        syn_idx = cell.get_closest_idx(x=self.syn_soma_pos[0], y=self.syn_soma_pos[1], z=self.syn_soma_pos[2])
        synapse_parameters_soma = {
            'idx': syn_idx,
            'e': 0., #  Change to -90 for inhibitory input, and 0 for excitatory
            'syntype': 'Exp2Syn',
            'tau1': 1.,
            'tau2': 2.,
            'weight': self.syn_weight,
            'record_current': False,
        }
        synapse_soma = LFPy.Synapse(cell, **synapse_parameters_soma)
        synapse_soma.set_spike_times(self.input_spike_train_soma)

        syn_idx_apic = cell.get_closest_idx(x=self.syn_apic_pos[0], y=self.syn_apic_pos[1], z=self.syn_apic_pos[2])
        synapse_parameters_apic = {
            'idx': syn_idx_apic,
            'e': 0., #  Change to -90 for inhibitory input, and 0 for excitatory
            'syntype': 'Exp2Syn',
            'tau1': 1.,
            'tau2': 2.,
            'weight': self.syn_weight * 4,
            'record_current': False,
        }
        synapse_apic = LFPy.Synapse(cell, **synapse_parameters_apic)
        synapse_apic.set_spike_times(self.input_spike_train_apic)
        cell.simulate(rec_imem=True, rec_vmem=True, rec_isyn=False)
        self.make_mapping(cell, MEA, ext_sim_dict)

        return cell, synapse_soma, synapse_apic

    def make_mapping(self, cell, MEA, ext_sim_dict):
        moi_params = {
            'sigma_G': 0.0,  # Below electrode
            'sigma_T': 0.3,  # Tissue
            'sigma_S': 1.5,  # Saline
            'h': MEA.slice_thickness,
            'steps': 20,
            }

        moi = MoI(**moi_params)
        mapping_normal_saline = moi.make_mapping_cython(ext_sim_dict, xmid=cell.xmid, ymid=cell.ymid, zmid=cell.zmid)
        MEA.phi = 1000 * np.dot(mapping_normal_saline, cell.imem)

    def plot_results(self, cell, synapse_soma, synapse_apic, MEA):

        time_window = [10, 40]
        syn_soma_idx = synapse_soma.idx
        syn_apic_idx = synapse_apic.idx

        cell_plot_idxs = [syn_soma_idx, syn_apic_idx]
        cell_plot_colors = {syn_soma_idx: 'y', syn_apic_idx: 'g'}
        num_cols = 4

        fig = plt.figure(figsize=[15, 5])
        plt.subplots_adjust(hspace=0.6, wspace=0.3, right=0.99, left=0.03, top=0.9)
        ax1 = fig.add_axes([0.05, 0.5, 0.37, 0.4], aspect=1, frameon=False, xticks=[], yticks=[], title='Top view')
        ax3 = fig.add_axes([0.05, 0.1, 0.37, 0.4], aspect=1, frameon=False, xticks=[], yticks=[], title='Side view')

        ax_ec = fig.add_subplot(1, num_cols, 3, xlim=time_window, xlabel='ms', ylabel='$\mu$V',
                                title='Extracellular\npotential')
        ax_v = plt.subplot(1, num_cols, 4, title='Membrane potential', xlabel='ms', ylabel='mV',
                           ylim=[-80, 20], xlim=time_window)

        l_elec, l_soma, l_syn = self._plot_recording_set_up(cell, ax1, ax3, MEA, syn_soma_idx, syn_apic_idx, cell_plot_colors)
        [ax_v.plot(cell.tvec, cell.vmem[idx, :], c=cell_plot_colors[idx], lw=2) for idx in cell_plot_idxs]
        for elec in range(MEA.num_elecs):
            ax_ec.plot(cell.tvec, MEA.phi[elec] - MEA.phi[elec, 0], lw=2, c=MEA.elec_clr[elec])

        fig.legend([l_soma, l_elec], ["Synapse", "MEA electrode"],
                   frameon=False, numpoints=1, ncol=3, loc=3)
        simplify_axes([ax_v, ax_ec])
        mark_subplots([ax1, ax3, ax_ec, ax_v], ypos=1.05, xpos=-0.1)

        # plt.savefig('test.png', dpi=150)

    def _plot_recording_set_up(self, cell, ax_neur, ax_side, MEA, syn_soma_idx, syn_apic_idx, cell_plot_colors):

        for comp in xrange(len(cell.xmid)):
            if comp == 0:
                ax_neur.scatter(cell.xmid[comp], cell.ymid[comp], s=cell.diam[comp],
                                edgecolor='none', color='gray', zorder=1)
            else:
                ax_neur.plot([cell.xstart[comp], cell.xend[comp]],
                             [cell.ystart[comp], cell.yend[comp]],
                             lw=cell.diam[comp]/2, color='gray', zorder=1)

        for comp in xrange(len(cell.xmid)):
            if comp == 0:
                ax_side.scatter(cell.xmid[comp], cell.zmid[comp], s=cell.diam[comp],
                                edgecolor='none', color='gray', zorder=1)
            else:
                ax_side.plot([cell.xstart[comp], cell.xend[comp]],
                             [cell.zstart[comp], cell.zend[comp]],
                             lw=cell.diam[comp]/2, color='gray', zorder=1)
        for idx in range(MEA.num_elecs):
            ax_side.plot(MEA.elec_x[idx], MEA.elec_z[idx] - 10, 's', clip_on=False,
                         c=MEA.elec_clr[idx], zorder=10, mec='none')
            ax_side.plot([MEA.elec_x[idx], MEA.elec_x[idx]], [MEA.elec_z[idx] - 10, MEA.elec_z[idx] - 150],
                         c=MEA.elec_clr[idx], zorder=10, lw=2)
            ax_neur.plot(MEA.elec_x[idx], MEA.elec_y[idx], 's', c=MEA.elec_clr[idx], zorder=10)

        ax_side.axhspan(-250, 0, facecolor='0.5', edgecolor='none')
        ax_side.axhspan(0, MEA.slice_thickness, facecolor='lightsalmon', edgecolor='none')
        ax_side.axhspan(MEA.slice_thickness, MEA.slice_thickness + 250, facecolor='aqua', edgecolor='none')
        ax_neur.axhspan(-500, 500, facecolor='lightsalmon', edgecolor='none')

        l_elec, = ax_neur.plot(MEA.elec_x[0], MEA.elec_y[0], 's', c=MEA.elec_clr[0], zorder=0)

        l_soma, = ax_neur.plot(cell.xmid[syn_soma_idx], cell.ymid[syn_soma_idx], '*', c=cell_plot_colors[syn_soma_idx], ms=15)

        l_apic, = ax_neur.plot(cell.xmid[syn_apic_idx], cell.ymid[syn_apic_idx], '*', c=cell_plot_colors[syn_apic_idx], ms=15)

        ax_side.plot(cell.xmid[syn_soma_idx], cell.zmid[syn_soma_idx], '*', c=cell_plot_colors[syn_soma_idx], ms=15)
        ax_side.plot(cell.xmid[syn_apic_idx], cell.zmid[syn_apic_idx], '*', c=cell_plot_colors[syn_apic_idx], ms=15)

        ax_neur.arrow(-220, -100, 30, 0, lw=1, head_width=12, color='k', clip_on=False)
        ax_neur.arrow(-220, -100, 0, 30, lw=1, head_width=12, color='k', clip_on=False)
        ax_neur.text(-150, -100, 'x', size=10, ha='center', va='center', clip_on=False)
        ax_neur.text(-220, -20, 'y', size=10, ha='center', va='center', clip_on=False)

        ax_side.arrow(-220, 20, 30, 0, lw=1, head_width=12, color='k', clip_on=False)
        ax_side.text(-140, 25, 'x', size=10, ha='center', va='center')
        ax_side.arrow(-220, 20, 0, 30, lw=1, head_width=12, color='k', clip_on=False)
        ax_side.text(-220, 100, 'z', size=10, ha='center', va='center')

        ax_side.plot([-500, 1250], [MEA.slice_thickness, MEA.slice_thickness], color='k')
        ax_side.plot([-500, 1250], [0, 0], 'k')  # PLOTTING BOTTOM OF MEA

        ax_side.plot([1280, 1280], [0, MEA.slice_thickness], '_-',
                     color='k', lw=2, clip_on=False, solid_capstyle='butt')
        ax_side.text(1300, MEA.slice_thickness / 2, '%g $\mu$m' % MEA.slice_thickness, size=8, va='center')

        ax_side.text(1200, -10, 'MEA', va='top', ha='right')
        ax_side.text(1200, MEA.slice_thickness, 'Saline', va='bottom', ha='right')
        ax_side.text(1200, MEA.slice_thickness - 50, 'Tissue', va='top', ha='right')

        ax_side.axis([-300, 1250, -70, MEA.slice_thickness + 70])
        ax_neur.axis([-300, 1250, -250, 250])

        return l_elec, l_soma, l_apic


def simplify_axes(axes):
    """
    :param axes: The axes object or list that is to be simplified. Right and top axis line is removed
    :return:
    """
    if not type(axes) is list:
        axes = [axes]

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()


def mark_subplots(axes, letters='ABCDEFGHIJKLMNOPQRSTUVWXYZ', xpos=-0.12, ypos=1.15):
    """ Marks subplots in axes (should be list or axes object) with capital letters
    """
    if not type(axes) is list:
        axes = [axes]

    for idx, ax in enumerate(axes):
        ax.text(xpos, ypos, letters[idx].capitalize(),
                horizontalalignment='center',
                verticalalignment='center',
                fontweight='demibold',
                fontsize=12,
                transform=ax.transAxes)

if __name__ == '__main__':

    #  Electrode position on the MEA plane. Arrays must be of equal length
    elec_x = [0, 400, 800]  # um
    elec_y = [0, 0, 0]

    slice_thickness = 300  # um
    cell_z_pos = 50  # z-position of cell, relative to MEA plane at 0 um
    synapse_soma_pos = [0, 0, cell_z_pos]  # Synapse is inserted at closest cell position to point (x,y,z)
    synapse_apic_pos = [800, 50, cell_z_pos]

    mea = SimulateMEA(synapse_soma_pos, synapse_apic_pos, slice_thickness, cell_z_pos, elec_x, elec_y)
