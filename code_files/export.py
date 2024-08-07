import numpy as np
import pandas as pd
import os
import pathlib
import datetime
import os.path
import matplotlib.pyplot as plt
from simulation import SimKeys, Simulation
import matplotlib.cm as cm


class ExpKeys:
    TIME = "Time (ms)"
    MSE_TOTAL = "mse_total"
    MAPE_TOTAL = "mape_total"
    MAE_TOTAL = "mae_total"


class Data:
    def __init__(self, plots_every, agent_config, sim_config):
        self.plots_every = plots_every
        self.last_plotted = None
        self.last_individual_error = None
        self.last_individual_time = 0
        self.total_errors = []
        self.mse_tsteps = None
        self.mape_tsteps = None
        self.mae_tsteps = None
        self.agent_config = agent_config
        self.sim_config = sim_config

        self.path, self.image_path = self._create_folder()
        self._create_meta_data()

    def _create_folder(self):
        parent_path = os.path.join(pathlib.Path().resolve(), "data")
        if not os.path.exists(parent_path):
            os.makedirs(parent_path)

        folder_name = "Simulation_" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        dir_name = os.path.join(parent_path, folder_name)
        os.makedirs(dir_name)

        im_path = os.path.join(dir_name, "Graphs")
        os.makedirs(im_path)

        return dir_name, im_path

    def _create_meta_data(self):
        path1 = os.path.join(self.path, "sim_config.txt")
        path2 = os.path.join(self.path, "agent_config.txt")

        with open(path1, "w") as f1, open(path2, "w") as f2:
            f1.write(''.join(["%s = %s\n" % (k, v) for k, v in self.sim_config.__dict__.items()]))
            f2.write(''.join(["%s = %s\n" % (k, v) for k, v in self.agent_config.__dict__.items()]))

    def finish(self, sim: Simulation):
        dir_name = os.path.join(self.image_path, "summary")
        os.makedirs(dir_name)

        create_sim_linechart(sim, dir_name)

        if len(self.total_errors) == 0:
            return
        total_error = pd.DataFrame.from_records(self.total_errors)
        time = total_error[[ExpKeys.TIME]].div(1000)
        if self.mse_tsteps is not None and self.mape_tsteps is not None and self.mae_tsteps is not None:
            create_error_linechart_steps(time.to_numpy(), self.mse_tsteps, "MSE [-]", dir_name)
            create_error_linechart_steps(time.to_numpy(), self.mape_tsteps, "MAPE [%]", dir_name)
            create_error_linechart_steps(time.to_numpy(), self.mae_tsteps, "MAE [-]", dir_name)

        create_error_linechart_total(time.to_numpy(), total_error[[ExpKeys.MSE_TOTAL]].to_numpy(), "MSE [-]", dir_name)
        create_error_linechart_total(time.to_numpy(), total_error[[ExpKeys.MAPE_TOTAL]].to_numpy(), "MAPE [%]",
                                     dir_name)
        create_error_linechart_total(time.to_numpy(), total_error[[ExpKeys.MAE_TOTAL]].to_numpy(), "MAE [-]", dir_name)

        if self.last_individual_error is not None:
            f_name_heatmap = "error_heatmap.png"
            f_name_scatter = "3d_scatter_error.png"
            create_error_scatter_3d(self.last_individual_error, self.last_individual_time,
                                    os.path.join(dir_name, f_name_scatter))
            create_error_heatmap(self.last_individual_error, self.last_individual_time,
                                 os.path.join(dir_name, f_name_heatmap))

    def record_errors(self, predictions: np.array, actual_values: np.array, time):
        mse_total = np.mean((predictions - actual_values) ** 2)
        mape_total = np.mean(np.abs((actual_values - predictions) / actual_values)) * 100
        mae_total = np.mean(np.abs(actual_values - predictions))

        total = {
            ExpKeys.TIME: time,
            ExpKeys.MSE_TOTAL: mse_total,
            ExpKeys.MAPE_TOTAL: mape_total,
            ExpKeys.MAE_TOTAL: mae_total
        }
        self.total_errors.append(total)

        mse_tsteps = np.mean((predictions - actual_values) ** 2, axis=0)
        mape_tsteps = np.mean(np.abs((actual_values - predictions) / actual_values), axis=0) * 100
        mae_tsteps = np.mean(np.abs(actual_values - predictions), axis=0)

        # Add different error measures for each time step to an array
        if self.mse_tsteps is None:
            self.mse_tsteps = np.copy(mse_tsteps)
            self.mse_tsteps = np.expand_dims(self.mse_tsteps, axis=0)
        else:
            self.mse_tsteps = np.vstack([self.mse_tsteps, mse_tsteps])

        if self.mape_tsteps is None:
            self.mape_tsteps = np.copy(mape_tsteps)
            self.mape_tsteps = np.expand_dims(self.mape_tsteps, axis=0)
        else:
            self.mape_tsteps = np.vstack([self.mape_tsteps, mape_tsteps])

        if self.mae_tsteps is None:
            self.mae_tsteps = np.copy(mae_tsteps)
            self.mae_tsteps = np.expand_dims(self.mae_tsteps, axis=0)
        else:
            self.mae_tsteps = np.vstack([self.mae_tsteps, mae_tsteps])

        quad_individual = (predictions - actual_values) ** 2
        self.last_individual_error = quad_individual
        self.last_individual_time = time

        # Graphs are only saved after a set intervall
        if self.last_plotted is None or time >= self.last_plotted + self.plots_every:
            self.last_plotted = 0 if self.last_plotted is None else time

            mse_strategies = np.mean((predictions - actual_values) ** 2, axis=1)
            mape_strategies = np.mean(np.abs((actual_values - predictions) / actual_values), axis=1) * 100
            mae_strategies = np.mean(np.abs(actual_values - predictions), axis=1)

            f_name = f"Time_{str(time / 1000)}s"
            dir_name = os.path.join(self.image_path, f_name)
            os.makedirs(dir_name)
            f_name_heatmap = "error_heatmap.png"
            f_name_scatter = "3d_scatter_error.png"

            create_error_heatmap(quad_individual, time, os.path.join(dir_name, f_name_heatmap))
            create_error_scatter_3d(quad_individual, time, os.path.join(dir_name, f_name_scatter))

            create_error_bar_chart_strategies(mse_strategies, "MSE [-]", time, dir_name)
            create_error_bar_chart_strategies(mape_strategies, "MAPE [%]", time, dir_name)
            create_error_bar_chart_strategies(mae_strategies, "MAE [-]", time, dir_name)

            create_error_bar_chart_steps(mse_tsteps, "MSE [-]", time, dir_name)
            create_error_bar_chart_steps(mape_tsteps, "MAPE [%]", time, dir_name)
            create_error_bar_chart_steps(mae_tsteps, "MAE [-]", time, dir_name)


def create_error_heatmap(errors: np.array, time, path):
    # First Axis are the different strategies, second axis are the future time step
    plt.ioff()
    x_labels = [f"t+{str(i + 1)}" for i in range(errors.shape[1])]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(errors, interpolation='nearest', cmap='viridis', aspect='auto')
    ax.set_xticks(np.arange(len(x_labels)), labels=x_labels)
    ax.set_xlabel("Zeitschritt [-]")
    ax.set_ylabel("Strategie Index [-]")
    plt.colorbar(im)
    plt.title(f"Individueller Quadratischer Fehler für t={str(time / 1000)}s")
    plt.savefig(path)
    plt.clf()
    plt.close()


def create_error_scatter_3d(errors: np.array, time, path):
    plt.ioff()
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    x_data = np.arange(0, errors.shape[0], 1)
    y_data = np.arange(0, errors.shape[1], 1)
    X, Y = np.meshgrid(x_data, y_data)
    y_labels = [f"t+{str(i + 1)}" for i in range(errors.shape[1])]
    ax.scatter(X, Y, errors, s=2, c=Y, vmin=0, vmax=errors.shape[1], cmap=cm.get_cmap("plasma"))
    ax.set_yticks(np.arange(len(y_labels)), labels=y_labels)
    ax.set_xlabel("Strategie Index [-]")
    ax.set_ylabel("Zeitschritt [-]")
    ax.set_zlabel("Quadratischer Fehler [-]")

    plt.title(f"Individueller Quadratischer Fehler für t={str(time / 1000)}s")

    plt.savefig(path)
    plt.clf()
    plt.close()


def create_error_bar_chart_strategies(errors: np.array, error_type, time, path):
    plt.ioff()
    plt.figure()
    plt.bar(np.arange(0, errors.shape[0], 1), errors)
    plt.ylabel(error_type)
    plt.xlabel("Strategie Index")
    plt.title(error_type + " für die Vorhersage aller Strategien für t=" + str(time / 1000) + "s")
    f_name = error_type + "_strategies.png"
    plt.savefig(os.path.join(path, f_name))
    plt.clf()
    plt.close()


def create_error_bar_chart_steps(errors: np.array, error_type, time, path):
    x_labels = [f"t+{str(i + 1)}" for i in range(errors.shape[0])]

    plt.ioff()
    plt.figure()
    plt.bar(x_labels, errors)
    plt.ylabel(error_type)
    plt.xlabel("Zeitschritt")
    plt.title(error_type + " für die Vorhersage aller Zeitschritte für t=" + str(time / 1000) + "s")
    f_name = error_type + "_steps.png"
    plt.savefig(os.path.join(path, f_name))
    plt.clf()
    plt.close()


def create_error_linechart_steps(time: np.array, errors: np.array, error_type, path, ylim=None):
    # Here errors is a 2d array that includes the error of every time step over all predictions
    plt.ioff()
    plt.figure()
    legend = [f"t+{str(i + 1)}" for i in range(errors.shape[1])]

    for tstep in errors.T:
        plt.plot(time, tstep)

    if ylim is not None:
        plt.ylim(ylim)
    plt.grid(visible=True)
    plt.ylabel(error_type)
    plt.xlabel("Zeits [s]")
    plt.title(error_type + " aller Vorhersagen für die einzelnen Zeitpunkte")
    plt.legend(legend)
    f_name = error_type + "_line_steps.png"
    plt.savefig(os.path.join(path, f_name))
    plt.clf()
    plt.close()


def create_error_linechart_total(time: np.array, errors: np.array, error_type, path, ylim=None):
    # Here errors is a 2d array that includes the error of every time step over all predictions
    plt.ioff()
    plt.figure()
    plt.plot(time, errors)
    plt.grid(visible=True)
    plt.ylabel(error_type)
    plt.xlabel("Zeit [s]")
    if ylim is not None:
        plt.ylim(ylim)
    plt.title(error_type + " Gesamtfehler über die gesamte Laufzeit")
    f_name = error_type + "_line_total.png"
    plt.savefig(os.path.join(path, f_name))
    plt.clf()
    plt.close()


def create_sim_linechart(sim: Simulation, path):
    plt.ioff()
    df = pd.DataFrame.from_records(sim.df)
    time = df[SimKeys.TIME].div(1000)

    fig, axs = plt.subplots(4, 1, sharex='row', figsize=(10, 8), gridspec_kw={'height_ratios': [4, 1, 1, 1]})

    axs[0].plot(time, df[SimKeys.VALUE])
    axs[0].plot(time, df[SimKeys.REFERENCE], c='k')
    axs[0].set_ylim(bottom=sim.min_val, top=sim.max_val)
    axs[0].set_xlim(left=0)
    axs[0].axhline(sim.stable_val, color='black', linestyle='dotted')
    axs[0].legend(["Regelgröße y", "Führungsgröße w"])
    axs[0].set_xlabel("Zeit [s]")
    axs[0].set_ylabel("y(t) [-]")
    axs[0].grid(visible=True)
    axs[0].set_title("Verlauf der Regelgröße über die Zeit")

    axs[1].plot(time, df[SimKeys.ACTION])
    axs[1].set_xlabel("Zeit [s]")
    axs[1].set_ylabel("u(t) [-]")
    axs[1].set_ylim(bottom=-0.1)
    axs[1].set_xlim(left=0)
    axs[1].grid(visible=True)
    axs[1].set_title("Verlauf der Stellgröße über die Zeit")

    axs[2].plot(time, df[SimKeys.Z])
    axs[2].set_xlabel("Zeit [s]")
    axs[2].set_ylabel("z(t) [-]")
    axs[2].set_xlim(left=0)
    axs[2].grid(visible=True)
    axs[2].set_title("Verlauf der Störungen über die Zeit")

    axs[3].plot(time, df[SimKeys.K1])
    axs[3].plot(time, df[SimKeys.K2])
    axs[3].set_ylim(bottom=0)
    axs[3].set_xlabel("Zeit [s]")
    axs[3].set_ylabel("k(t) [-]")
    axs[3].set_xlim(left=0)
    axs[3].legend(["k1", "k2"])
    axs[3].grid(visible=True)
    axs[3].set_title("Verlauf der Zustandsgrößen k1 und k2 über die Zeit")

    plt.tight_layout()

    fname = "sim_linechart.png"
    plt.savefig(os.path.join(path, fname))
    plt.clf()
    plt.close()

